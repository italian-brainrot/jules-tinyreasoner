from datasets import load_dataset
import json
import random
import re
import os
from src.capabilities import lookup_dictionary, evaluate_math

def inject_capabilities(text):
    # Find words to define (3-8 characters, lowercase only for simplicity)
    words = re.findall(r'\b[a-z]{3,8}\b', text)
    if words:
        word = random.choice(words)
        definition = lookup_dictionary(word)
        if "No definition found" not in definition:
            cap_call = f" [DEFINE]{word}[CAPABILITY_STOP]{definition}[CAPABILITY_STOP]"
            # Inject before a sentence that might use it
            sentences = text.split('. ')
            if len(sentences) > 1:
                idx = random.randint(0, len(sentences) - 1)
                # Insert the capability call and then a sentence explaining/using it
                sentences.insert(idx, f"I should check the meaning of {word}.{cap_call} The word {word} means {definition}")
                text = '. '.join(sentences)
            else:
                text = f"I need to define {word}.{cap_call} {text} (Note: {word} means {definition})"

    # Add a random math check if numbers are present
    nums = re.findall(r'\d+', text)
    if len(nums) >= 2:
        n1, n2 = random.sample(nums, 2)
        expr = f"{n1} + {n2}"
        res = evaluate_math(expr)
        cap_call = f" [SYMPY]{expr}[CAPABILITY_STOP]{res}[CAPABILITY_STOP]"
        text = f"First, let me check a calculation: {expr} = ?{cap_call} So {expr} is {res}. " + text

    return text

def main():
    print("Loading dataset...")
    # Using a subset of hermes-agent-reasoning-traces or similar
    try:
        ds = load_dataset("lambda/hermes-agent-reasoning-traces", "kimi", split="train", streaming=True)
    except:
        print("Failed to load hermes, trying a simpler one")
        ds = load_dataset("fka/awesome-chatgpt-prompts", split="train", streaming=True)

    data = []
    count = 0
    max_examples = 1000

    print(f"Processing up to {max_examples} examples...")
    for item in ds:
        if count >= max_examples:
            break

        # Extract prompt and response
        # Hermes has 'conversations' list. fka has 'prompt' and 'response' (or just prompt)
        if 'conversations' in item:
            convs = item['conversations']
            prompt = ""
            completion = ""
            for msg in convs:
                if msg['from'] == 'human':
                    prompt += msg['value'] + "\n"
                elif msg['from'] == 'gpt':
                    completion += msg['value'] + "\n"

            if not prompt or not completion:
                continue
        else:
            # Fallback for other datasets
            prompt = item.get('prompt', '')
            completion = item.get('response', item.get('completion', ''))
            if not prompt or not completion:
                continue

        # Inject capabilities into completion
        completion = inject_capabilities(completion)

        # Add Reasoning tag if not present
        if "Reasoning:" not in completion:
            completion = "Reasoning: " + completion

        data.append({
            "prompt": prompt.strip(),
            "completion": completion.strip()
        })
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}...")

    os.makedirs("data", exist_ok=True)
    with open("data/sft_real_data.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} examples in data/sft_real_data.json")

if __name__ == "__main__":
    import nltk
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
    main()
