from src.sampler import Sampler
from src.tokenizer import CharTokenizer
from src.model import TinyReasonerModel
from src.prompts import get_random_prompt
import torch
import os

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    model = TinyReasonerModel(tokenizer.vocab_size).to(device)

    if os.path.exists("models/rl_model.pt"):
        model.load_state_dict(torch.load("models/rl_model.pt", map_location=device))
        print("Loaded RL model.")
    else:
        print("RL model not found.")
        return

    sampler = Sampler(model, tokenizer, device=device)

    for i in range(10):
        prompt_text, ref_answer, task_type = get_random_prompt()
        prompt = f"[BOS]{prompt_text}"
        print(f"\n--- Test {i+1} ---")
        print(f"Prompt: {prompt_text}")
        output = sampler.sample(prompt, max_len=256, temperature=0)
        print(f"Output: {output}")

if __name__ == "__main__":
    evaluate()
