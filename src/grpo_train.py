import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_optimizer import SOAP
import os
import copy
from src.tokenizer import CharTokenizer
from src.model import TinyReasonerModel
from src.sampler import Sampler
from src.rewards import get_total_reward
from src.prompts import get_random_prompt

def compute_grpo_loss(model, ref_model, tokens, old_log_probs, mask, advantages, clip_eps=0.2, beta=0.0001):
    # tokens: (batch, seq_len)
    # old_log_probs: list of tensors (different lengths)
    # mask: list of tensors
    # advantages: (batch,)

    total_loss = 0

    for i in range(len(tokens)):
        t = torch.tensor([tokens[i]]).long().to(model.embedding.weight.device)
        m = mask[i].clone().detach().to(model.embedding.weight.device)
        adv = advantages[i]
        old_lp = old_log_probs[i]

        logits, _ = model(t)
        log_probs_full = F.log_softmax(logits[0, :-1, :], dim=-1)

        target_tokens = t[0, 1:]
        current_lp_all = log_probs_full[torch.arange(len(target_tokens)), target_tokens]

        # Ensure mask and current_lp_all have same length
        if len(m) > len(current_lp_all):
            m = m[:len(current_lp_all)]
        elif len(m) < len(current_lp_all):
            # This shouldn't happen based on sampler logic, but let's be safe
            current_lp_all = current_lp_all[:len(m)]

        current_lp = current_lp_all[m == 1]

        if len(current_lp) != len(old_lp):
            # print(f"Mismatch: current_lp {len(current_lp)}, old_lp {len(old_lp)}")
            if len(current_lp) > len(old_lp):
                current_lp = current_lp[:len(old_lp)]
            else:
                # Should not happen if sampler and mask are correct
                continue

        ratio = torch.exp(current_lp - old_lp)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL Penalty (optional, against ref_model)
        with torch.no_grad():
            ref_logits, _ = ref_model(t)
            ref_lp_full = F.log_softmax(ref_logits[0, :-1, :], dim=-1)
            ref_lp = ref_lp_full[torch.arange(len(target_tokens)), target_tokens][m == 1]

        kl = (torch.exp(ref_lp - current_lp) - (ref_lp - current_lp) - 1).mean()

        total_loss += policy_loss + beta * kl

    return total_loss / len(tokens)

def train_grpo(num_iterations=500, group_size=32, load_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = CharTokenizer()
    model = TinyReasonerModel(tokenizer.vocab_size).to(device)

    start_iteration = 0
    if load_model and os.path.exists(load_model):
        model.load_state_dict(torch.load(load_model, map_location=device))
        print(f"Loaded model from {load_model}.")
    elif os.path.exists("models/rl_model.pt"):
        model.load_state_dict(torch.load("models/rl_model.pt", map_location=device))
        print("Loaded existing RL model.")
        # Re-starting from level 0 to ensure grounding.
        start_iteration = 0
    elif os.path.exists("models/sft_model.pt"):
        model.load_state_dict(torch.load("models/sft_model.pt", map_location=device))
        print("Loaded SFT model.")
    else:
        print("Warning: No model found. Starting from scratch or pretrained.")
        if os.path.exists("models/pretrained.pt"):
            model.load_state_dict(torch.load("models/pretrained.pt", map_location=device))
            print("Loaded pretrained model.")

    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    embedding_params = list(model.embedding.parameters())
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n]
    param_groups = [
        {"params": other_params},
        {"params": embedding_params, "max_precond_dim": 1}
    ]
    optimizer = SOAP(param_groups, lr=1e-5) # Smaller LR for RL

    sampler = Sampler(model, tokenizer, device=device)

    for i in range(start_iteration, start_iteration + num_iterations):
        # Curriculum: Level 0 for first 300 iters, Level 1 for next 300, Level 2 after
        if i < 300:
            level = 0
        elif i < 600:
            level = 1
        else:
            level = 2

        prompt_text, ref_answer, task_type = get_random_prompt(level=level)
        prompt = f"[BOS]{prompt_text}"

        # 1. Rollout with exploration noise
        with torch.no_grad():
            # Alternate between noise and slightly higher temperature for variety
            use_noise = (i % 2 == 0)
            completions, log_probs, masks = sampler.grpo_rollout(
                prompt,
                num_rollouts=group_size,
                temperature=1.0 if use_noise else 1.1,
                noise_std=0.03 if use_noise else 0.0
            )

        # 2. Rewards
        rewards = []
        for completion in completions:
            r = get_total_reward(prompt_text, completion, ref_answer, task_type)
            rewards.append(r)

        rewards = torch.tensor(rewards).to(device)
        unique_completions = len(set(completions))
        print(f"Iter {i} (Level {level}), Prompt: {prompt}, Mean Reward: {rewards.mean().item():.4f}, Unique: {unique_completions}/{group_size}", flush=True)
        if i % 1 == 0:
            print(f"Sample Completion: {completions[0]}", flush=True)

        # 3. Advantages
        if len(rewards) > 1:
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            adv = rewards - rewards.mean()

        # 4. Update
        model.train()
        optimizer.zero_grad()

        # Re-encode completions to tokens
        all_tokens = [tokenizer.encode(c) for c in completions]

        loss = compute_grpo_loss(model, ref_model, all_tokens, log_probs, masks, adv)
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            torch.save(model.state_dict(), "models/rl_model.pt")
            print(f"Saved checkpoint at iter {i+1}")

    torch.save(model.state_dict(), "models/rl_model.pt")
    print("RL training complete. Model saved to models/rl_model.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--load_model", type=str, default=None)
    args = parser.parse_args()

    train_grpo(num_iterations=args.iterations, group_size=args.group_size, load_model=args.load_model)
