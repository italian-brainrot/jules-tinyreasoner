import torch
from src.tokenizer import CharTokenizer
from src.model import TinyReasonerModel
from src.sampler import Sampler
from src.capabilities import dispatch_capability

def integration_test():
    print("Starting integration test...")
    tokenizer = CharTokenizer()
    model = TinyReasonerModel(tokenizer.vocab_size)
    sampler = Sampler(model, tokenizer)

    # Test 1: Basic encoding/decoding
    text = "Hello, world!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert text == decoded, f"Tokenizer failed: {text} != {decoded}"
    print("Test 1 (Tokenizer) passed.")

    # Test 2: Model forward pass
    x = torch.tensor([encoded]).long()
    logits, hidden = model(x)
    assert logits.shape == (1, len(encoded), tokenizer.vocab_size)
    print("Test 2 (Model) passed.")

    # Test 3: Capability dispatch
    res_def = dispatch_capability("DEFINE", "apple")
    assert "fruit" in res_def.lower()
    res_math = dispatch_capability("SYMPY", "1 + 1")
    assert res_math == "2"
    print("Test 3 (Capabilities) passed.")

    # Test 4: Sampler (smoke test)
    # We can't easily force the model to call a capability without training,
    # but we can check if it runs without error.
    output = sampler.sample("The quick brown fox", max_len=30)
    print(f"Sample output: {output}")
    print("Test 4 (Sampler) passed.")

    print("Integration test completed successfully!")

if __name__ == "__main__":
    integration_test()
