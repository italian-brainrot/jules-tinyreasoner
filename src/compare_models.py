import torch
import os
import argparse
from src.tokenizer import CharTokenizer
from src.model import TinyReasonerModel
from src.sampler import Sampler
from src.prompts import get_random_prompt
from src.rewards import get_total_reward, reward_grounding

def compare_models(model_paths=None, num_samples=20, level=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()

    if model_paths is None:
        models = {
            "SFT": "models/sft_model.pt",
            "RL": "models/rl_model.pt"
        }
    else:
        models = {os.path.basename(p): p for p in model_paths}

    results = {}

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue

        model = TinyReasonerModel(tokenizer.vocab_size).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        sampler = Sampler(model, tokenizer, device=device)

        total_reward = 0
        total_grounding = 0
        correct_tasks = 0

        print(f"\n--- Evaluating {name} Model ({path}) ---")

        for i in range(num_samples):
            prompt_text, ref_answer, task_type = get_random_prompt(level=level)
            prompt = f"[BOS]{prompt_text}"

            output = sampler.sample(prompt, max_len=256, temperature=0.7)

            reward = get_total_reward(prompt_text, output, ref_answer, task_type)
            grounding = reward_grounding(prompt_text, output)

            total_reward += reward
            total_grounding += grounding

            # Count as grounded if grounding reward is positive
            if grounding > 0:
                correct_tasks += 1

            if i < 3: # Print first 3 samples for inspection
                print(f"P: {prompt_text}")
                print(f"O: {output[:150]}...")
                print(f"R: {reward:.2f}, G: {grounding:.2f}")

        results[name] = {
            "avg_reward": total_reward / num_samples,
            "avg_grounding": total_grounding / num_samples,
            "grounding_rate": correct_tasks / num_samples
        }

    for name, stats in results.items():
        print(f"\nResults for {name}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="Paths to model checkpoints to compare")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()

    compare_models(args.models, args.num_samples, args.level)
