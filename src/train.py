import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_optimizer import SOAP
import nltk
from nltk.corpus import gutenberg
from src.tokenizer import CharTokenizer
from src.model import TinyReasonerModel
import os

class GutenbergDataset(Dataset):
    def __init__(self, tokenizer, seq_len=256, max_chars=1000000):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Load Gutenberg text
        text = ""
        files = gutenberg.fileids()[:5] # Use first 5 books for variety
        for fileid in files:
            text += gutenberg.raw(fileid)
            if len(text) > max_chars:
                text = text[:max_chars]
                break

        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = torch.tensor(self.tokens[start:end]).long()
        y = torch.tensor(self.tokens[start+1:end+1]).long()
        return x, y

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = CharTokenizer()
    dataset = GutenbergDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TinyReasonerModel(tokenizer.vocab_size).to(device)

    # Configure optimizers as requested:
    # SOAP/SPlus for recurrent layers, Adam (or diagonal) for embeddings.
    # SOAP with max_precond_size=1 is like Adam.

    # Separate parameters
    embedding_params = list(model.embedding.parameters())
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n]

    param_groups = [
        {"params": other_params}, # Default SOAP
        {"params": embedding_params, "max_precond_size": 1} # Adam-like behavior in SOAP
    ]

    optimizer = SOAP(param_groups, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")

            # For verification dry-run
            if os.environ.get("DRY_RUN") == "1" and i >= 5:
                break

        avg_loss = total_loss / (i + 1)
        print(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/pretrained.pt")
    print("Model saved to models/pretrained.pt")

if __name__ == "__main__":
    train()
