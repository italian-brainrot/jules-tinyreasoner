import nltk
from nltk.corpus import wordnet
import sympy
import re

def lookup_dictionary(word):
    word = word.strip().lower()
    synsets = wordnet.synsets(word)
    if not synsets:
        return f"No definition found for {word}."
    # Return the first definition
    return synsets[0].definition()

def evaluate_math(expression):
    expression = expression.strip()
    try:
        # Basic sanitization - sympy.sympify is relatively safe but we should be careful
        # Only allow numbers, basic operators, and some common functions
        # For a small reasoning model, we keep it simple.
        res = sympy.sympify(expression)
        return str(res)
    except Exception as e:
        return f"Error evaluating expression: {e}"

def dispatch_capability(cap_type, payload):
    if cap_type == "DEFINE":
        return lookup_dictionary(payload)
    elif cap_type == "SYMPY":
        return evaluate_math(payload)
    else:
        return f"Unknown capability: {cap_type}"

if __name__ == "__main__":
    # Ensure wordnet is loaded
    try:
        wordnet.synsets("test")
    except LookupError:
        nltk.download('wordnet')

    print(f"Define 'apple': {lookup_dictionary('apple')}")
    print(f"Math '2 + 2 * 5': {evaluate_math('2 + 2 * 5')}")
    print(f"Math 'expand((x+1)**2)': {evaluate_math('expand((x+1)**2)')}")

    assert "fruit" in lookup_dictionary("apple").lower()
    assert evaluate_math("2+2") == "4"
    print("Capabilities test passed!")
