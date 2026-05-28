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

def generate_complex_math_prompt():
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    c = random.randint(1, 100)

    # (a + b) * c or (a - b) + c etc
    op1, symbol1 = random.choice([("+", "plus"), ("-", "minus")])
    op2, symbol2 = random.choice([("*", "times"), ("+", "plus")])

    prompt = f"What is ({a} {symbol1} {b}) {symbol2} {c}?"
    expression = f"({a} {op1} {b}) {op2} {c}"
    result = str(evaluate_math(expression))

    return prompt, result, "math"

def generate_comparison_prompt():
    import nltk
    try:
        word_list = nltk.corpus.words.words()
    except LookupError:
        nltk.download('words')
        word_list = nltk.corpus.words.words()

    word1 = random.choice(word_list).lower()
    word2 = random.choice(word_list).lower()

    if len(word1) > len(word2):
        result = word1
    elif len(word2) > len(word1):
        result = word2
    else:
        result = "both"

    prompt = f"Which word is longer: '{word1}' or '{word2}'? If they are equal, say 'both'."

    return prompt, result, "dict"

def get_random_prompt():
    r = random.random()
    if r < 0.3:
        return generate_math_prompt()
    elif r < 0.6:
        return generate_dict_prompt()
    elif r < 0.8:
        return generate_complex_math_prompt()
    else:
        return generate_comparison_prompt()
