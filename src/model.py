import torch
import torch.nn as nn

class TinyReasonerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        embeds = self.embedding(x) # (batch, seq_len, embed_dim)
        lstm_out, hidden = self.lstm(embeds, hidden) # (batch, seq_len, hidden_dim)
        logits = self.fc(lstm_out) # (batch, seq_len, vocab_size)
        return logits, hidden

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    from src.tokenizer import CharTokenizer
    tokenizer = CharTokenizer()
    model = TinyReasonerModel(tokenizer.vocab_size)
    params = count_parameters(model)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Total parameters: {params}")
    if params < 1000000:
        print("Model is under 1 million parameters.")
    else:
        print("Model is TOO BIG!")

    # Test forward pass
    test_input = torch.tensor([tokenizer.encode("test")]).long()
    logits, _ = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (1, 4, tokenizer.vocab_size)
    print("Model test passed!")
