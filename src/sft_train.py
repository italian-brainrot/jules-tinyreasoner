import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_optimizer import SOAP
import json
import os
import sys
from src.tokenizer import CharTokenizer
from src.model import TinyReasonerModel

class SFTDataset(Dataset):
    def __init__(self, tokenizer, data_path, seq_len=512):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        with open(data_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = f"[BOS]{item['prompt']}\n{item['completion']}[EOS]"
        tokens = self.tokenizer.encode(full_text)

        # Pad or truncate
        if len(tokens) > self.seq_len + 1:
            tokens = tokens[:self.seq_len + 1]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len + 1 - len(tokens))

        x = torch.tensor(tokens[:-1]).long()
        y = torch.tensor(tokens[1:]).long()

        return x, y

def train_sft(data_path="data/sft_data.json", output_path="models/sft_model.pt", base_model_path=None, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = CharTokenizer()
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    dataset = SFTDataset(tokenizer, data_path)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TinyReasonerModel(tokenizer.vocab_size).to(device)

    if base_model_path and os.path.exists(base_model_path):
        model.load_state_dict(torch.load(base_model_path, map_location=device))
        print(f"Loaded base model from {base_model_path}.")
    elif os.path.exists("models/pretrained.pt"):
        model.load_state_dict(torch.load("models/pretrained.pt", map_location=device))
        print("Loaded pretrained model.")
    else:
        print("Warning: No base model found. Training from scratch.")

    embedding_params = list(model.embedding.parameters())
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n]

    param_groups = [
        {"params": other_params},
        {"params": embedding_params, "max_precond_dim": 1}
    ]

    optimizer = SOAP(param_groups, lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

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

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/sft_data.json")
    parser.add_argument("--output", type=str, default="models/sft_model.pt")
    parser.add_argument("--base", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    train_sft(data_path=args.data, output_path=args.output, base_model_path=args.base, num_epochs=args.epochs)
