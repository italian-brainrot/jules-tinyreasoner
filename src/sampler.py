import torch
import torch.nn.functional as F
from src.tokenizer import CharTokenizer
from src.model import TinyReasonerModel
from src.capabilities import dispatch_capability

class Sampler:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def sample(self, prompt, max_len=512, temperature=1.0, top_k=0, stop_at_eos=True):
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens]).long().to(self.device)

        generated = tokens
        hidden = None

        # Pre-fill hidden state
        logits, hidden = self.model(input_ids, hidden)

        while len(generated) < max_len:
            last_logit = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            if temperature == 0:
                next_token = torch.argmax(last_logit, dim=-1).item()
            else:
                probs = F.softmax(last_logit, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(next_token)

            token_str = self.tokenizer.itos.get(next_token, "")

            if token_str in ["[DEFINE]", "[SYMPY]"]:
                cap_type = token_str[1:-1]
                payload_tokens = []
                found_stop = False

                while len(generated) < max_len:
                    input_ids = torch.tensor([[generated[-1]]]).long().to(self.device)
                    logits, hidden = self.model(input_ids, hidden)

                    # Inside capability call, we usually want greedy or same temp
                    last_logit = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                    if temperature == 0:
                        nt = torch.argmax(last_logit, dim=-1).item()
                    else:
                        nt = torch.multinomial(F.softmax(last_logit, dim=-1), num_samples=1).item()

                    generated.append(nt)
                    if nt == self.tokenizer.stop_token_id:
                        found_stop = True
                        break
                    payload_tokens.append(nt)

                if found_stop:
                    payload = self.tokenizer.decode(payload_tokens)
                    result = dispatch_capability(cap_type, payload)
                    result_tokens = self.tokenizer.encode(result)
                    result_tokens.append(self.tokenizer.stop_token_id)

                    for r_token in result_tokens:
                        generated.append(r_token)
                        input_ids = torch.tensor([[generated[-1]]]).long().to(self.device)
                        logits, hidden = self.model(input_ids, hidden)
                else:
                    break
            elif next_token == self.tokenizer.eos_token_id:
                if stop_at_eos:
                    break
                else:
                    input_ids = torch.tensor([[generated[-1]]]).long().to(self.device)
                    logits, hidden = self.model(input_ids, hidden)
            else:
                input_ids = torch.tensor([[generated[-1]]]).long().to(self.device)
                logits, hidden = self.model(input_ids, hidden)

        return self.tokenizer.decode(generated)

    def grpo_rollout(self, prompt, num_rollouts=8, max_len=512, temperature=1.0):
        """Perform multiple rollouts for GRPO.
        Returns completions, log_probs, and mask for model-generated tokens.
        """
        prompt_tokens = self.tokenizer.encode(prompt)
        all_completions = []
        all_log_probs = []
        all_masks = [] # 1 for model generated, 0 for prompt or tool injected

        for _ in range(num_rollouts):
            generated_tokens = list(prompt_tokens)
            log_probs = [] # Only for tokens where mask == 1
            mask = [0] * (len(prompt_tokens) - 1) # Mask for prompt tokens (except BOS)

            hidden = None
            input_ids = torch.tensor([prompt_tokens]).long().to(self.device)
            logits, hidden = self.model(input_ids, hidden)

            curr_len = len(generated_tokens)
            while curr_len < max_len:
                last_logit = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                probs = F.softmax(last_logit, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1).item()
                lp = torch.log(probs[0, next_token] + 1e-10)

                generated_tokens.append(next_token)
                log_probs.append(lp)
                mask.append(1) # Model generated

                token_str = self.tokenizer.itos.get(next_token, "")
                if token_str in ["[DEFINE]", "[SYMPY]"]:
                    cap_type = token_str[1:-1]
                    payload_tokens = []
                    found_stop = False

                    while len(generated_tokens) < max_len:
                        input_ids = torch.tensor([[generated_tokens[-1]]]).long().to(self.device)
                        logits, hidden = self.model(input_ids, hidden)

                        last_logit = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                        probs = F.softmax(last_logit, dim=-1)
                        nt = torch.multinomial(probs, num_samples=1).item()
                        lp = torch.log(probs[0, nt] + 1e-10)

                        generated_tokens.append(nt)
                        # SPECIAL TOKENS handled by tokenizer might be multi-char but they are single tokens here.
                        # Wait, the tokenizer.encode handles [DEFINE] as one token.
                        log_probs.append(lp)
                        mask.append(1) # Model generated

                        if nt == self.tokenizer.stop_token_id:
                            found_stop = True
                            break
                        payload_tokens.append(nt)

                    if found_stop:
                        payload = self.tokenizer.decode(payload_tokens)
                        result = dispatch_capability(cap_type, payload)
                        result_tokens = self.tokenizer.encode(result)
                        result_tokens.append(self.tokenizer.stop_token_id)

                        for r_token in result_tokens:
                            generated_tokens.append(r_token)
                            mask.append(0) # Tool injected

                            input_ids = torch.tensor([[generated_tokens[-1]]]).long().to(self.device)
                            logits, hidden = self.model(input_ids, hidden)
                    else:
                        break
                elif next_token == self.tokenizer.eos_token_id:
                    break
                else:
                    input_ids = torch.tensor([[generated_tokens[-1]]]).long().to(self.device)
                    logits, hidden = self.model(input_ids, hidden)

                curr_len = len(generated_tokens)

            all_completions.append(self.tokenizer.decode(generated_tokens))
            all_log_probs.append(torch.stack(log_probs))
            all_masks.append(torch.tensor(mask).to(self.device))

        return all_completions, all_log_probs, all_masks

if __name__ == "__main__":
    import os
    import sys

    tokenizer = CharTokenizer()
    model = TinyReasonerModel(tokenizer.vocab_size)

    model_path = "models/sft_model.pt"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model {model_path} not found, using random weights.")

    sampler = Sampler(model, tokenizer)

    prompts = [
        "[BOS]What is the definition of apple?",
        "[BOS]What is the sum of 10 and 20?"
    ]

    for p in prompts:
        print(f"\nPrompt: {p}")
        output = sampler.sample(p, max_len=256, temperature=0)
        print(f"Output: {output}")
