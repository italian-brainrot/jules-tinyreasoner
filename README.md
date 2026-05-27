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

## Next Steps
- Continue RL training for more iterations to improve accuracy.
- Experiment with more complex math and reasoning tasks.
- Fine-tune reward weights.

## Usage
To test the model:
```bash
PYTHONPATH=. python3 src/sampler.py models/sft_model.pt
```
