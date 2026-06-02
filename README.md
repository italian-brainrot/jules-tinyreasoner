# TinyReasoner

A reasoning model under 1 million parameters capable of tool calling.

## Architecture
- 2-layer LSTM
- Hidden size: 256
- Embedding size: 128
- Character-level tokenizer with special tokens for capabilities.

## Progress

### Session 1: Pretraining
- Implemented the model, tokenizer, and pretraining script.
- Pretrained on NLTK Gutenberg corpus (approx. 1M characters).
- Saved checkpoint: `models/pretrained.pt`.

### Session 2: Supervised Fine-Tuning (SFT)
- Implemented `src/capabilities.py` with dictionary (NLTK WordNet) and math (SymPy) tools.
- Implemented `src/sampler.py` with capability call detection and result injection.
- Created `src/generate_sft_data.py` to generate 2000 synthetic reasoning traces.
- Created `src/sft_train.py` for instruction tuning using the SOAP optimizer.
- Fine-tuned the model on reasoning traces.
- Saved checkpoint: `models/sft_model.pt`.
- Verified that the model starts to use `[DEFINE]` and `[SYMPY]` tokens correctly.

### Session 3: Reinforcement Learning (GRPO)
- Implemented GRPO training in `src/grpo_train.py`.
- Created multi-faceted reward functions in `src/rewards.py`.
- Expanded prompt generation in `src/prompts.py`.
- Updated `src/sampler.py` to support multi-rollout log-prob and mask tracking.
- Verified RL training loop and checkpointing.
- Saved initial RL checkpoint: `models/rl_model.pt`.

### Session 4: Extended RL and Task Complexity
- Enhanced `src/prompts.py` with multi-step math problems and word length comparisons.
- Refined `src/rewards.py` with length penalties and rewards for utilizing tool results.
- Improved `src/grpo_train.py` for continuous training and increased iterations to 500.
- Successfully completed 500 iterations of GRPO training.
- Verified that the model maintains reasoning traces and tool use even with increased task complexity.
- Synced all artifacts to Hugging Face bucket.

### Session 5: Addressing Mode Collapse and Real Data Integration
- Identified "mode collapse" where the model defaulted to `[DEFINE]elephant` for most prompts.
- Updated `src/generate_sft_data.py` to use a much larger vocabulary from NLTK.
- Created `src/generate_real_sft_data.py` to incorporate real tool-calling traces (from Hermes dataset) with injected capability calls.
- Retrained SFT model on combined synthetic and real data (3000 examples) for 10 epochs.
- Strengthened RL rewards in `src/rewards.py`:
    - Increased grounding reward to 0.25 per entity.
    - Added a -0.5 penalty for tool calls that don't match prompt entities (hallucinations).
    - Increased reward for utilizing tool results to 0.3.
- Continued GRPO training with the new reward structure.

### Session 6: Aggressive RL and Grounding Incentives
- Drastically increased grounding rewards (from 0.5 to 5.0) and hallucination penalties (from -1.0 to -10.0) in `src/rewards.py`.
- Introduced a specific -20.0 penalty for the `elephant` hallucination mode to force exploration.
- Reduced KL penalty (`beta` from 0.01 to 0.0001) in `grpo_train.py` to allow the model to move away from the collapsed SFT state.
- Increased GRPO `group_size` to 16 for better advantage estimation.
- Implemented alternating exploration strategies in `src/sampler.py` (noise vs. temperature).
- Observed the model starting to explore other words like `cat`, `banana`, and `jacket` in RL logs.
- Added detailed logging of sample completions and unique completion counts in the training loop.

### Session 7: Curriculum Learning and Dense Rewards
- Implemented a three-level curriculum strategy in `src/prompts.py`:
    - Level 0: Single-digit math, 3-4 letter words.
    - Level 1: Two-digit math, 4-6 letter words, simple comparisons and synonyms.
    - Level 2: Original complex tasks.
- Updated `src/rewards.py` for dense grounding rewards:
    - Rewards are now given for *each* unique number or word matched from the prompt (min length 3).
- Enhanced `src/grpo_train.py` with curriculum progression and persistence:
    - Automatically skips to Level 2 if an existing `rl_model.pt` is found to prevent catastrophic forgetting.
    - Increased total iterations to 1000.
- Optimized `src/prompts.py` by caching NLTK word lists at the module level.

## Next Steps
- Monitor the transition between curriculum levels to ensure the model maintains performance.
- Continue RL training with these aggressive rewards until grounding (using prompt words/numbers) becomes the dominant strategy.
- If the model still struggles with grounding, consider generating "near-miss" SFT data where only the grounding is changed.
- Monitor training diversity to ensure the model doesn't just collapse into a different fixed word (like `cat`).
- Fine-tune the reward balance as the model starts showing desired behaviors.

## Usage
To test the model:
```bash
PYTHONPATH=. python3 src/sampler.py models/sft_model.pt
```
