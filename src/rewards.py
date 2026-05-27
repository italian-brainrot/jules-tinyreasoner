import re
from src.capabilities import lookup_dictionary, evaluate_math

def reward_reasoning_tag(completion):
    """Reward for including 'Reasoning:' tag."""
    if "Reasoning:" in completion:
        return 0.1
    return 0.0

def reward_answer_tag(completion):
    """Reward for including 'Answer:' tag."""
    if "Answer:" in completion:
        return 0.1
    return 0.0

def reward_capability_syntax(completion):
    """Reward for correct capability syntax: [CAP]payload[CAPABILITY_STOP]result[CAPABILITY_STOP]."""
    # This is tricky because the result and the second stop are injected by the sampler.
    # We want to check if the model generated [DEFINE]...[CAPABILITY_STOP] or [SYMPY]...[CAPABILITY_STOP].
    pattern = r"\[(DEFINE|SYMPY)\][^\]]+\[CAPABILITY_STOP\]"
    if re.search(pattern, completion):
        return 0.2
    return 0.0

def reward_correctness(completion, reference_answer, task_type):
    """Reward for matching the reference answer."""
    # Extract Answer: ...
    match = re.search(r"Answer:\s*(.*)", completion)
    if not match:
        return 0.0

    extracted_answer = match.group(1).strip()

    if task_type == "math":
        # Try to compare numerically if possible
        try:
            if float(extracted_answer) == float(reference_answer):
                return 1.0
        except:
            if extracted_answer == reference_answer:
                return 1.0
    else:
        # Dictionary: fuzzy match or exact?
        # SFT data uses the exact first definition.
        if reference_answer.lower() in extracted_answer.lower() or extracted_answer.lower() in reference_answer.lower():
            return 1.0

    return 0.0

def get_total_reward(completion, reference_answer, task_type):
    reward = 0.0
    reward += reward_reasoning_tag(completion)
    reward += reward_answer_tag(completion)
    reward += reward_capability_syntax(completion)
    reward += reward_correctness(completion, reference_answer, task_type)
    return reward
