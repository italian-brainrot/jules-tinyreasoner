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
            prompt_nums = set(re.findall(r"\d+", prompt))
            payload_nums = set(re.findall(r"\d+", payload))

            if not payload_nums:
                reward -= 10.0
            else:
                for num in payload_nums:
                    if num in prompt_nums:
                        reward += 10.0
                    else:
                        reward -= 20.0

                if any(num in prompt_nums for num in payload_nums):
                    call_grounded = True

        elif cap_type == "DEFINE":
            prompt_words = set([w.lower() for w in re.findall(r"\w+", prompt) if len(w) >= 3])
            payload_words = set([w.lower() for w in re.findall(r"\w+", payload) if len(w) >= 3])

            if not payload_words:
                reward -= 10.0
            else:
                for w in payload_words:
                    if w in prompt_words:
                        reward += 10.0
                    else:
                        reward -= 20.0

                if any(w in prompt_words for w in payload_words):
                    call_grounded = True

        if not call_grounded:
            reward -= 10.0 # Strong penalty for each hallucinated call

        # Specific anti-hallucination for known mode-collapse words
        for mode_word in ["elephant", "jacket", "moss", "bat", "sout", "guitar", "cat", "banana", "tomss", "seet"]:
            if mode_word in payload.lower() and mode_word not in prompt.lower():
                reward -= 20.0

        # Target Grounding Bonus: check if the primary target word/numbers are present
        if cap_type == "DEFINE":
            # Extract target word from prompt like "What is the definition of apple?"
            target_match = re.search(r"definition of ([\w'-]+)", prompt.lower())
            if target_match:
                target_word = target_match.group(1)
                if target_word in payload.lower():
                    reward += 10.0 # Extra bonus for target word grounding

        elif cap_type == "SYMPY":
            # Check if all numbers in prompt are in payload
            prompt_nums = re.findall(r"\d+", prompt)
            if all(num in payload for num in prompt_nums) and len(prompt_nums) > 0:
                reward += 10.0 # Extra bonus for full math grounding

    return min(max(reward, -100.0), 40.0)

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
