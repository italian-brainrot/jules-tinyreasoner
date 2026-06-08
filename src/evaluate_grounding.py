import torch
import os
from src.tokenizer import CharTokenizer
from src.model import TinyReasonerModel
from src.sampler import Sampler
from src.prompts import get_random_prompt
from src.rewards import get_total_reward

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    model = TinyReasonerModel(tokenizer.vocab_size).to(device)

    model_path = "models/rl_model.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))

    sampler = Sampler(model, tokenizer, device=device)

    levels = [0]
    temps = [0.7, 1.0, 1.3]

    for level in levels:
        print(f"--- Level {level} ---")
        for _ in range(3):
            prompt_text, ref_answer, task_type = get_random_prompt(level=level)
            prompt = f"[BOS]{prompt_text}"
            print(f"\nPrompt: {prompt_text}")

            for temp in temps:
                output = sampler.sample(prompt, max_len=256, temperature=temp)
                reward = get_total_reward(prompt_text, output, ref_answer, task_type)
                print(f"  Temp {temp}: Reward {reward:.2f}")
                print(f"  Output: {output[:150]}...")

if __name__ == "__main__":
    evaluate()
