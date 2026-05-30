import json
import random
import os
from src.capabilities import lookup_dictionary, evaluate_math

def generate_dictionary_task():
    import nltk
    try:
        word_list = nltk.corpus.words.words()
    except LookupError:
        nltk.download('words')
        word_list = nltk.corpus.words.words()

    # Filter for reasonably sized words that actually have definitions
    word = None
    definition = "No definition found."
    for _ in range(10):
        w = random.choice(word_list).lower()
        d = lookup_dictionary(w)
        if d != f"No definition found for {w}.":
            word = w
            definition = d
            break

    if word is None:
        word = "apple"
        definition = lookup_dictionary(word)

    prompt = f"What is the definition of {word}?"
    reasoning = f"Reasoning: I need to find the definition of {word}. [DEFINE]{word}[CAPABILITY_STOP]{definition}[CAPABILITY_STOP] The word {word} means {definition}."
    answer = f"Answer: {definition}"

    return {
        "prompt": prompt,
        "completion": f"{reasoning} {answer}"
    }

def generate_math_task():
    a = random.randint(1, 500)
    b = random.randint(1, 500)
    ops = [("+", "sum of"), ("-", "difference between"), ("*", "product of")]
    op, phrase = random.choice(ops)

    expression = f"{a} {op} {b}"
    result = evaluate_math(expression)

    prompt = f"What is the {phrase} {a} and {b}?"
    reasoning = f"Reasoning: I need to calculate {expression}. [SYMPY]{expression}[CAPABILITY_STOP]{result}[CAPABILITY_STOP] The result of {expression} is {result}."
    answer = f"Answer: {result}"

    return {
        "prompt": prompt,
        "completion": f"{reasoning} {answer}"
    }

def main():
    data = []
    for _ in range(2000):
        if random.random() < 0.5:
            data.append(generate_dictionary_task())
        else:
            data.append(generate_math_task())

    os.makedirs("data", exist_ok=True)
    with open("data/sft_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} SFT examples in data/sft_data.json")

if __name__ == "__main__":
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    main()
