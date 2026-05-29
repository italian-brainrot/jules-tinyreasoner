import torch
from src.tokenizer import CharTokenizer
from src.model import TinyReasonerModel
from src.sampler import Sampler

def test():
    device = "cpu"
    tokenizer = CharTokenizer()
    model = TinyReasonerModel(tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load("models/rl_model.pt", map_location=device))
    sampler = Sampler(model, tokenizer, device=device)

    test_prompts = [
        "What is the definition of guitar?",
        "What is the sum of 5 and 7?",
        "Which word is longer: 'cat' or 'elephant'?",
        "What is a synonym for 'fast'?"
    ]

    for p in test_prompts:
        print(f"\nPrompt: {p}")
        output = sampler.sample(f"[BOS]{p}", max_len=256, temperature=0)
        print(f"Output: {output}")

if __name__ == "__main__":
    test()
