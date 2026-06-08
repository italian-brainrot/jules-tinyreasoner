import json
import random
import os
import re
from src.capabilities import lookup_dictionary, evaluate_math

def generate_grounding_dictionary_task():
    import nltk
    try:
        word_list = nltk.corpus.words.words()
    except LookupError:
        nltk.download('words')
        word_list = nltk.corpus.words.words()

    # Level 0/1 style words: 3 to 6 letters
    filtered_words = [w for w in word_list if 3 <= len(w) <= 6]

    word = None
    definition = "No definition found."
    for _ in range(20):
        w = random.choice(filtered_words).lower()
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

def generate_grounding_math_task():
    # Level 0 style: single digits
    a = random.randint(1, 9)
    b = random.randint(1, 9)
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
    for _ in range(1000):
        data.append(generate_grounding_dictionary_task())
    for _ in range(1000):
        data.append(generate_grounding_math_task())

    random.shuffle(data)
    os.makedirs("data", exist_ok=True)
    with open("data/grounding_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} grounding-focused SFT examples in data/grounding_data.json")

if __name__ == "__main__":
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    main()
