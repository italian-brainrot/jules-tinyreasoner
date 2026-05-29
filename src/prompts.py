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

def generate_synonym_prompt():
    import nltk
    from nltk.corpus import wordnet
    try:
        word_list = nltk.corpus.words.words()
    except LookupError:
        nltk.download('words')
        word_list = nltk.corpus.words.words()

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
    import nltk
    from nltk.corpus import wordnet
    try:
        word_list = nltk.corpus.words.words()
    except LookupError:
        nltk.download('words')
        word_list = nltk.corpus.words.words()

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

def get_random_prompt():
    r = random.random()
    if r < 0.25:
        return generate_math_prompt()
    elif r < 0.5:
        return generate_dict_prompt()
    elif r < 0.7:
        return generate_complex_math_prompt()
    elif r < 0.85:
        return generate_comparison_prompt()
    elif r < 0.92:
        return generate_synonym_prompt()
    else:
        return generate_antonym_prompt()
