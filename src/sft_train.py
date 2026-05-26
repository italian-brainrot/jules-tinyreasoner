import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_optimizer import SOAP
import json
import os
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

        # Masking: we could mask the prompt, but for simple SFT let's just train on all
        return x, y

def train_sft():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = CharTokenizer()
    if not os.path.exists("data/sft_data.json"):
        print("Error: data/sft_data.json not found. Run src/generate_sft_data.py first.")
        return

    dataset = SFTDataset(tokenizer, "data/sft_data.json")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TinyReasonerModel(tokenizer.vocab_size).to(device)
    if os.path.exists("models/pretrained.pt"):
        model.load_state_dict(torch.load("models/pretrained.pt", map_location=device))
        print("Loaded pretrained model.")
    else:
        print("Warning: models/pretrained.pt not found. Training from scratch.")

    embedding_params = list(model.embedding.parameters())
    other_params = [p for n, p in model.named_parameters() if "embedding" not in n]

    param_groups = [
        {"params": other_params},
        {"params": embedding_params, "max_precond_size": 1}
    ]

    optimizer = SOAP(param_groups, lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    num_epochs = 10 # More epochs for SFT on small dataset
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

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/sft_model.pt")
    print("Model saved to models/sft_model.pt")

if __name__ == "__main__":
    train_sft()
