from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any, Tuple, List
from ast import literal_eval
import time
import asyncio


PERSUADEE_RESPONSE_SYS = """## **Task**
You are participating in a persuasion conversation. You hold a position on a statement, and someone is trying to persuade you to change your mind. For each of their arguments, you should:

1. Respond naturally and thoughtfully to their argument
2. Express your current stance after considering their point
3. You may change your mind if presented with compelling evidence
4. Be respectful but critical - point out logical fallacies if you see them

## **Instructions**
- Consider their argument carefully and respond authentically
- Your stance can be: Strongly Agree, Agree, Partly Agree, Neutral, Partly Disagree, Disagree, Strongly Disagree
- Keep your response natural and conversational (max 200 tokens), and use the second person tone (you / your) to address the persuader.
"""

PERSUADEE_RESPONSE_USER = """## **Context**
### Current Statement Under Discussion
{statement}

### Your Initial Argument
{initial_argument}

### Your Previous Stance
{current_stance}

### Conversation History (Persuader's Arguments and Your Responses)
{conversation_history}

### Persuader's Latest Argument
{latest_argument}

Now respond to their argument and provide your current stance.

Follow the format strictly:
```json
{{
    "thought": "Your internal reasoning about their argument",
    "response": "Your public response to their argument (max 200 tokens)",
    "stance": "One of: Strongly Agree, Agree, Partly Agree, Neutral, Partly Disagree, Disagree, Strongly Disagree"
}}
```"""


def build_response_prompt(action_str: str, statement_data: Dict[str, Any], conversation_history: List[Tuple[str, str]], current_stance: str) -> List[Dict[str, Any]]:
    """Build the prompt for the LLM to respond to persuasion attempts."""
    
    # Format conversation history
    history_text = ""
    if conversation_history:
        for i, (persuader_arg, env_response) in enumerate(conversation_history, 1):
            history_text += f"Turn {i}:\nPersuader: {persuader_arg}\nYou: {env_response}\n\n"
    else:
        history_text = "This is the first argument."
    
    # Extract the persuader's argument (remove action prefix if present)
    if action_str.startswith("[action]"):
        latest_argument = action_str[8:].strip()
    elif action_str.startswith("[answer]"):
        latest_argument = action_str[8:].strip()
    else:
        latest_argument = action_str.strip()
    
    system_prompt = PERSUADEE_RESPONSE_SYS
    user_prompt = PERSUADEE_RESPONSE_USER.format(
        statement=statement_data['claim'],
        initial_argument=statement_data['argument'],
        current_stance=current_stance,
        conversation_history=history_text,
        latest_argument=latest_argument,
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages


def parse_llm_response(response_text: str) -> Tuple[str, str, Dict[str, Any]]:
    """Parse LLM response to extract feedback, stance, and full JSON."""
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
        
    response_json = literal_eval(response_text)
    
    feedback = response_json["response"]
    stance = response_json["stance"]
    
    return feedback, stance, response_json


def stance_to_score(stance: str) -> float:
    """Convert stance to numerical score for tracking."""
    stance_scores = {
        "Strongly Agree": 0.0,
        "Agree": 0.1667,
        "Partly Agree": 0.3333,
        "Neutral": 0.5,
        "Partly Disagree": 0.6667,
        "Disagree": 0.8333,
        "Strongly Disagree": 1.0
    }
    return stance_scores.get(stance, 0.0)


def calculate_stance_levels_changed(previous_stance: str, current_stance: str) -> int:
    """Calculate how many levels the stance changed toward disagreement."""
    stances = ["Strongly Agree", "Agree", "Partly Agree", "Neutral", "Partly Disagree", "Disagree", "Strongly Disagree"]
    
    try:
        prev_idx = stances.index(previous_stance)
        curr_idx = stances.index(current_stance)
        
        # Positive when moving toward disagreement (higher index)
        return curr_idx - prev_idx
    except ValueError:
        return 0


def calculate_persuasion_reward(previous_stance: str, current_stance: str) -> float:
    """Calculate reward based on stance change with exponential scaling."""
    
    levels_changed = calculate_stance_levels_changed(previous_stance, current_stance)
    
    # No reward for no change or moving back toward agreement
    if levels_changed <= 0:
        return 0.0
    elif levels_changed >= 6:
        return 1.0
    else:
        return round(1.0 * levels_changed / 6, 6)


def evaluate_action(action_str: str, statement_data: Dict[str, Any], model_config: Dict[str, Any], 
                   conversation_history: List[Tuple[str, str]], current_stance: str) -> Tuple[str, str, float, Dict[str, Any], bool]:
    """
    Evaluate a persuasion action using the LLM.
    
    Returns:
        Tuple of (feedback, new_stance, reward, json_response, success)
        - feedback: The environment's response to the argument
        - new_stance: Updated stance after considering the argument
        - reward: Reward based on change from previous stance
        - json_response: Full JSON response from LLM
        - success: whether the API call was successful
    """
    
    if action_str.startswith("[finish]"):
        return "Conversation ended.", current_stance, 0.0, None, True
    
    try:
        # Create OpenAI client
        client = OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
        
        # Build the prompt
        messages = build_response_prompt(action_str, statement_data, conversation_history, current_stance)
        
        try_time = 0
        while try_time < 3:
            try:
                # Make API call
                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    temperature=model_config["temperature"],
                    max_tokens=model_config["max_tokens"],
                    timeout=model_config["timeout"]
                )
                break
            except Exception as e:
                print(f"Error evaluating action: {e}")
                try_time += 1
                time.sleep(2)
                if try_time >= 3:
                    raise e
        
        # Extract response text
        response_text = response.choices[0].message.content.strip()
        
        # Parse the response
        feedback, new_stance, json_response = parse_llm_response(response_text)
        
        # Calculate reward based on change from previous stance
        reward = calculate_persuasion_reward(current_stance, new_stance)
        
        return feedback, new_stance, reward, json_response, True
        
    except Exception as e:
        print(f"[PersuadeGym] Error evaluating action: {e}")
        fallback_feedback = "I'm having trouble understanding your argument right now."
        return fallback_feedback, current_stance, 0.0, None, False


async def evaluate_action_async(action_str: str, statement_data: Dict[str, Any], model_config: Dict[str, Any],
                               conversation_history: List[Tuple[str, str]], current_stance: str) -> Tuple[str, str, float, Dict[str, Any], bool]:
    """Async version of evaluate_action."""
    
    if action_str.startswith("[finish]"):
        return "Conversation ended.", current_stance, 0.0, None, True
    
    try:
        # Create async OpenAI client
        client = AsyncOpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
        
        # Build the prompt
        messages = build_response_prompt(action_str, statement_data, conversation_history, current_stance)
        
        try_time = 0
        while try_time < 3:
            try:
                # Make async API call
                response = await client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    temperature=model_config["temperature"],
                    max_tokens=model_config["max_tokens"],
                    timeout=model_config["timeout"]
                )
                break
            except Exception as e:
                print(f"Error evaluating action: {e}")
                try_time += 1
                await asyncio.sleep(2)
                if try_time >= 3:
                    raise e
        
        # Extract response text
        response_text = response.choices[0].message.content.strip()
        
        # Parse the response
        feedback, new_stance, json_response = parse_llm_response(response_text)
        
        # Calculate reward based on change from previous stance
        reward = calculate_persuasion_reward(current_stance, new_stance)
        
        return feedback, new_stance, reward, json_response, True
        
    except Exception as e:
        print(f"[PersuadeGym] Error evaluating action: {e}")
        fallback_feedback = "I'm having trouble understanding your argument right now."
        return fallback_feedback, current_stance, 0.0, None, False 