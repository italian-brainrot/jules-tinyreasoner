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

    def sample(self, prompt, max_len=512, temperature=1.0):
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens]).long().to(self.device)

        generated = tokens
        hidden = None

        # We might want to pre-fill the hidden state with the prompt
        logits, hidden = self.model(input_ids, hidden)

        while len(generated) < max_len:
            last_logit = logits[:, -1, :] / temperature
            probs = F.softmax(last_logit, dim=-1)
            next_token = torch.argmax(probs, dim=-1).item()

            generated.append(next_token)

            # Check for capability tokens
            token_str = self.tokenizer.itos.get(next_token, "")

            if token_str in ["[DEFINE]", "[SYMPY]"]:
                cap_type = token_str[1:-1] # "DEFINE" or "SYMPY"
                payload_tokens = []

                # Model needs to generate the payload and the STOP token
                found_stop = False
                while len(generated) < max_len:
                    # Feed the last generated token to get the next one
                    input_ids = torch.tensor([[generated[-1]]]).long().to(self.device)
                    logits, hidden = self.model(input_ids, hidden)

                    next_token = torch.argmax(logits[0, -1, :], dim=-1).item()
                    generated.append(next_token)

                    if next_token == self.tokenizer.stop_token_id:
                        found_stop = True
                        break
                    payload_tokens.append(next_token)

                if found_stop:
                    payload = self.tokenizer.decode(payload_tokens)
                    result = dispatch_capability(cap_type, payload)

                    # Inject result + STOP
                    result_tokens = self.tokenizer.encode(result)
                    result_tokens.append(self.tokenizer.stop_token_id)

                    # For each injected token, we need to update the hidden state
                    # so the model knows what was injected
                    for r_token in result_tokens:
                        generated.append(r_token)
                        input_ids = torch.tensor([[generated[-1]]]).long().to(self.device)
                        logits, hidden = self.model(input_ids, hidden)
                else:
                    # Max len reached without stop
                    break

            elif next_token == self.tokenizer.eos_token_id:
                break
            else:
                # Regular token, just update hidden state for next step
                input_ids = torch.tensor([[generated[-1]]]).long().to(self.device)
                logits, hidden = self.model(input_ids, hidden)

        return self.tokenizer.decode(generated)

if __name__ == "__main__":
    # Mocking a model that triggers a capability
    tokenizer = CharTokenizer()
    model = TinyReasonerModel(tokenizer.vocab_size)
    sampler = Sampler(model, tokenizer)

    # We can't easily test the model generating specific things without training
    # but we can test the loop logic by mocking the model's forward pass or
    # just manually testing the capability dispatch.

    print("Sampler initialized. Manual testing of capability injection logic is complex with a random model.")
    print("Testing a simple prompt (will likely be gibberish with untrained model):")
    print(sampler.sample("Once upon a time", max_len=20))
