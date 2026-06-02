import random
from src.capabilities import lookup_dictionary, evaluate_math
import nltk

_word_list = None
def get_word_list():
    global _word_list
    if _word_list is None:
        try:
            _word_list = nltk.corpus.words.words()
        except LookupError:
            nltk.download('words')
            _word_list = nltk.corpus.words.words()
    return _word_list

def generate_math_prompt(level=2):
    if level == 0:
        a = random.randint(1, 9)
        b = random.randint(1, 9)
    elif level == 1:
        a = random.randint(1, 99)
        b = random.randint(1, 99)
    else:
        a = random.randint(1, 500)
        b = random.randint(1, 500)

    ops = [("+", "sum of"), ("-", "difference between"), ("*", "product of")]
    op, phrase = random.choice(ops)

    expression = f"{a} {op} {b}"
    result = evaluate_math(expression)
    prompt = f"What is the {phrase} {a} and {b}?"

    return prompt, result, "math"

def generate_dict_prompt(level=2):
    word_list = get_word_list()

    # Filter for reasonably sized words
    if level == 0:
        word_list = [w for w in word_list if 3 <= len(w) <= 4]
    elif level == 1:
        word_list = [w for w in word_list if 4 <= len(w) <= 6]
    else:
        word_list = [w for w in word_list if 3 < len(w) < 10]

    if not word_list:
        word_list = ["apple", "dog", "cat", "ball"]

    word = random.choice(word_list).lower()
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
    word_list = get_word_list()

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

def generate_synonym_prompt():
    from nltk.corpus import wordnet
    word_list = get_word_list()

    try:
        wordnet.synsets("test")
    except LookupError:
        nltk.download('wordnet')

    word = random.choice(word_list).lower()
    synsets = wordnet.synsets(word)

    if not synsets:
        return generate_dict_prompt() # Fallback

    synonyms = set()
    for syn in synsets:
        for lemma in syn.lemmas():
            if lemma.name().lower() != word:
                synonyms.add(lemma.name().replace('_', ' '))

    if not synonyms:
        return generate_dict_prompt() # Fallback

    result = random.choice(list(synonyms))
    prompt = f"What is a synonym for '{word}'?"
    return prompt, result, "dict"

def generate_antonym_prompt():
    from nltk.corpus import wordnet
    word_list = get_word_list()

    try:
        wordnet.synsets("test")
    except LookupError:
        nltk.download('wordnet')

    # Try a few times to find a word with an antonym
    for _ in range(20):
        word = random.choice(word_list).lower()
        synsets = wordnet.synsets(word)
        for syn in synsets:
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    antonym = lemma.antonyms()[0].name().replace('_', ' ')
                    prompt = f"What is the antonym of '{word}'?"
                    return prompt, antonym, "dict"

    return generate_dict_prompt() # Fallback

def get_random_prompt(level=2):
    r = random.random()

    # At lower levels, stick to simple math and dict lookups
    if level == 0:
        if r < 0.5:
            return generate_math_prompt(level=0)
        else:
            return generate_dict_prompt(level=0)

    if level == 1:
        if r < 0.4:
            return generate_math_prompt(level=1)
        elif r < 0.8:
            return generate_dict_prompt(level=1)
        elif r < 0.9:
            return generate_comparison_prompt() # Simple comparison
        else:
            return generate_synonym_prompt()

    # Level 2+: All prompts
    if r < 0.25:
        return generate_math_prompt(level=2)
    elif r < 0.5:
        return generate_dict_prompt(level=2)
    elif r < 0.7:
        return generate_complex_math_prompt()
    elif r < 0.85:
        return generate_comparison_prompt()
    elif r < 0.92:
        return generate_synonym_prompt()
    else:
        return generate_antonym_prompt()
