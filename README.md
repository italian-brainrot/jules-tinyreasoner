# Jules Tiny Reasoner

A project to train a tiny reasoning model (<1M parameters) capable of tool calling, using a linearly scaling architecture (LSTM).

## Current Status
- **Pretraining**: Stage 1 complete. The model has been pretrained on ~1M characters from the NLTK Gutenberg corpus.
- **Architecture**: 2-layer LSTM with hidden size 256 and embedding size 128 (~951k total parameters).
- **Tokenizer**: Custom character-level tokenizer with support for numbers, lowercase letters, an uppercase modifier token `[UPPER]`, and special capability tokens.
- **Capabilities**:
    - `[DEFINE]word[CAPABILITY_STOP]`: Look up word definitions using NLTK WordNet.
    - `[SYMPY]expression[CAPABILITY_STOP]`: Evaluate math expressions using SymPy.
- **Sampling**: Custom greedy sampler that handles capability call interception and result injection.

## Project Structure
- `src/`: Core logic
    - `model.py`: LSTM model definition.
    - `tokenizer.py`: Character-level tokenizer.
    - `capabilities.py`: Dictionary and SymPy tool implementations.
    - `sampler.py`: Greedy sampling loop with tool injection logic.
    - `train.py`: Pretraining script.
    - `setup.py`: Environment setup (NLTK downloads).
    - `integration_test.py`: End-to-end system test.
- `models/`: Model checkpoints (e.g., `pretrained.pt`).
- `data/`: Datasets (to be populated in SFT stage).

## Getting Started
1. Run setup to download dependencies:
   ```bash
   python3 src/setup.py
   ```
2. Run integration tests:
   ```bash
   PYTHONPATH=. python3 src/integration_test.py
   ```
3. Start pretraining (if needed):
   ```bash
   PYTHONPATH=. python3 src/train.py
   ```

## Next Steps
- Generate synthetic dataset of simple reasoning tasks for Supervised Fine-Tuning (SFT).
- Fine-tune the model to use capabilities for reasoning.
- Implement GRPO reinforcement learning stage.
