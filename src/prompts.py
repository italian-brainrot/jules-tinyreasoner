import random
from src.capabilities import lookup_dictionary, evaluate_math

def generate_math_prompt():
    a = random.randint(1, 500)
    b = random.randint(1, 500)
    ops = [("+", "sum of"), ("-", "difference between"), ("*", "product of")]
    op, phrase = random.choice(ops)

    expression = f"{a} {op} {b}"
    result = evaluate_math(expression)
    prompt = f"What is the {phrase} {a} and {b}?"

    return prompt, result, "math"

def generate_dict_prompt():
    # Use NLTK words corpus for more variety
    import nltk
    try:
        word_list = nltk.corpus.words.words()
    except LookupError:
        nltk.download('words')
        word_list = nltk.corpus.words.words()

    # Filter for reasonably sized words
    word_list = [w for w in word_list if 3 < len(w) < 10]
    word = random.choice(word_list)
    definition = lookup_dictionary(word)

    prompt = f"What is the definition of {word}?"

    return prompt, definition, "dict"

def get_random_prompt():
    if random.random() < 0.5:
        return generate_math_prompt()
    else:
        return generate_dict_prompt()
