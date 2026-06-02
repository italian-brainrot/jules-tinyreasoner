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
    reward = 0.0
    if re.search(pattern, completion):
        reward += 0.5

    if "Error evaluating expression" in completion:
        reward -= 0.5

    return reward

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

    reward = 0.0
    # Use the content after the last injected tool result for checking
    parts = completion.split("[CAPABILITY_STOP]")
    if len(parts) < 3:
        return 0.0

    after_all_tools = parts[-1].lower()

    for i in range(1, len(parts)-1, 2):
        result = parts[i].strip()
        if result and result != "No definition found." and result.lower() in after_all_tools:
            reward += 0.3

    return min(reward, 0.6)

def reward_grounding(prompt, completion):
    """Reward for using entities from the prompt in capability calls, and penalize hallucinations."""
    # Find capability calls
    pattern = r"\[(DEFINE|SYMPY)\](.*?)\[CAPABILITY_STOP\]"
    calls = re.findall(pattern, completion)
    if not calls:
        return 0.0

    reward = 0.0
    seen_calls = set()
    for cap_type, payload in calls:
        # Penalty for duplicate calls
        call_sig = f"{cap_type}:{payload}"
        if call_sig in seen_calls:
            reward -= 0.5
        seen_calls.add(call_sig)

        call_grounded = False
        if cap_type == "SYMPY":
            # Extract numbers from prompt
            prompt_nums = re.findall(r"\d+", prompt)
            num_matches = 0
            for num in set(prompt_nums):
                if num in payload:
                    reward += 5.0
                    num_matches += 1
            if num_matches > 0:
                call_grounded = True
        elif cap_type == "DEFINE":
            prompt_words = re.findall(r"\w+", prompt)
            # Skip very short words (keep length 3 for Level 0 curriculum)
            prompt_words = [w.lower() for w in prompt_words if len(w) >= 3]
            word_matches = 0
            for w in set(prompt_words):
                if w in payload.lower():
                    reward += 5.0
                    word_matches += 1
            if word_matches > 0:
                call_grounded = True

        if not call_grounded:
            reward -= 10.0 # Strong penalty for each hallucinated call

        # Specific anti-hallucination for 'elephant' mode collapse
        if "elephant" in payload.lower() and "elephant" not in prompt.lower():
            reward -= 20.0

    return min(max(reward, -100.0), 20.0)

def reward_correct_capability_type(completion, task_type):
    """Reward for calling the right tool type for the task."""
    pattern = r"\[(DEFINE|SYMPY)\]"
    matches = re.findall(pattern, completion)
    if not matches:
        return 0.0

    reward = 0.0
    for cap_type in matches:
        if task_type == "math":
            if cap_type == "SYMPY":
                reward += 0.5
            else:
                reward -= 1.0
        elif task_type == "dict":
            if cap_type == "DEFINE":
                reward += 0.5
            else:
                reward -= 1.0
    return reward

def get_total_reward(prompt, completion, reference_answer, task_type):
    reward = 0.0
    reward += reward_reasoning_tag(completion)
    reward += reward_answer_tag(completion)
    reward += reward_capability_syntax(completion)
    reward += reward_correctness(completion, reference_answer, task_type)
    reward += reward_length_penalty(completion)
    reward += reward_use_tool_result(completion)
    reward += reward_grounding(prompt, completion)
    reward += reward_correct_capability_type(completion, task_type)
    return reward
