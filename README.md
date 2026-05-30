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

## Next Steps
- Continue RL training to further break the habit of hallucinating fixed tool calls.
- Monitor grounding rewards to ensure the model starts picking up words/numbers from the prompt.
- Experiment with adding noise to the hidden state during RL to encourage exploration.
- If performance stagnates, consider a larger model within the 1M parameter limit.

## Usage
To test the model:
```bash
PYTHONPATH=. python3 src/sampler.py models/sft_model.pt
```
