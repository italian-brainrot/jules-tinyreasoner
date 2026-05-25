import torch

class CharTokenizer:
    def __init__(self):
        # Base characters: numbers (10), lowercase english (26), punctuation/space (around 32)
        # Plus special tokens: [UPPER], [DEFINE], [SYMPY], [CAPABILITY_STOP], [PAD], [BOS], [EOS]

        self.chars = "0123456789abcdefghijklmnopqrstuvwxyz !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        self.special_tokens = ["[UPPER]", "[DEFINE]", "[SYMPY]", "[CAPABILITY_STOP]", "[PAD]", "[BOS]", "[EOS]"]

        self.stoi = {char: i for i, char in enumerate(self.chars)}
        start_idx = len(self.chars)
        for i, token in enumerate(self.special_tokens):
            self.stoi[token] = start_idx + i

        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

        self.pad_token_id = self.stoi["[PAD]"]
        self.bos_token_id = self.stoi["[BOS]"]
        self.eos_token_id = self.stoi["[EOS]"]
        self.upper_token_id = self.stoi["[UPPER]"]
        self.define_token_id = self.stoi["[DEFINE]"]
        self.sympy_token_id = self.stoi["[SYMPY]"]
        self.stop_token_id = self.stoi["[CAPABILITY_STOP]"]

    def encode(self, text):
        tokens = []
        i = 0
        while i < len(text):
            # Check for special tokens first
            found_special = False
            for token in self.special_tokens:
                if text.startswith(token, i):
                    tokens.append(self.stoi[token])
                    i += len(token)
                    found_special = True
                    break
            if found_special:
                continue

            char = text[i]
            if char.isupper():
                tokens.append(self.upper_token_id)
                char = char.lower()

            if char in self.stoi:
                tokens.append(self.stoi[char])
            else:
                # Unknown characters treated as space or ignored? Let's just skip or use a '?'
                tokens.append(self.stoi.get('?', self.stoi[' ']))
            i += 1
        return tokens

    def decode(self, tokens):
        res = ""
        upper_next = False
        i = 0
        while i < len(tokens):
            t = tokens[i]
            s = self.itos.get(t, "")
            if s == "[UPPER]":
                upper_next = True
            elif s in self.special_tokens:
                res += s
                upper_next = False
            else:
                if upper_next:
                    res += s.upper()
                    upper_next = False
                else:
                    res += s
            i += 1
        return res

if __name__ == "__main__":
    tokenizer = CharTokenizer()
    test_str = "Hello World! 123 [DEFINE]test[CAPABILITY_STOP]"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_str}")
    print(f"Encoded:  {encoded}")
    print(f"Decoded:  {decoded}")
    assert test_str == decoded, f"Mismatch: {test_str} != {decoded}"
    print("Tokenizer test passed!")
