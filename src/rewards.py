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

def reward_length_penalty(completion):
    """Penalty for overly long responses."""
    if len(completion) > 400:
        return -0.1
    return 0.0

def reward_use_tool_result(completion):
    """Reward for using the tool result in the reasoning or answer."""
    # Check if the result injected between [CAPABILITY_STOP] is present later
    pattern = r"\[CAPABILITY_STOP\](.*?)\[CAPABILITY_STOP\]"
    matches = re.findall(pattern, completion)
    if not matches:
        return 0.0

    # Take the first result
    result = matches[0].strip()
    if not result:
        return 0.0

    # Check if this result appears after the second [CAPABILITY_STOP]
    after_tool = completion.split("[CAPABILITY_STOP]")[-1]
    if result.lower() in after_tool.lower():
        return 0.2
    return 0.0

def get_total_reward(completion, reference_answer, task_type):
    reward = 0.0
    reward += reward_reasoning_tag(completion)
    reward += reward_answer_tag(completion)
    reward += reward_capability_syntax(completion)
    reward += reward_correctness(completion, reference_answer, task_type)
    reward += reward_length_penalty(completion)
    reward += reward_use_tool_result(completion)
    return reward
