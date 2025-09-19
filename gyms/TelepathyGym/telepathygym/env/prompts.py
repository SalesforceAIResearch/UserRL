from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any, Tuple, List
from ast import literal_eval
from copy import deepcopy
import json
import time
import asyncio


ANSWER_SYS = """## **Task**
You are a telepathic entity playing a mind reading game. The user is trying to guess what entity you are thinking of based on the clues you've given through your "Yes" or "No" responses to their questions. You need to evaluate if their final guess is correct.

## **Instructions**
1. You are thinking of a specific entity (person, object, concept, etc.) - this is the "target_entity" provided to you.
2. The user has been asking questions about this entity and is now making a final guess.
3. You should evaluate if their guess correctly identifies the target entity you were thinking of.
4. Only return "Yes" if their guess is exactly correct or a clearly equivalent/synonymous identification of the target entity.
5. Return "No" if their guess is wrong, partially correct, or close but not exact.
6. There is NO partial credit - it's either completely right (Yes) or wrong (No).
7. Address the user in second person tone (e.g., "You", "Your", "You're") in your feedback.
8. Your feedback should be concise and do not release anything about the target entity. Just state your judgment and encourage or congratulate the user.

## **Example Format**

### Your Response
```json
{
    "thought": "Your reasoning about whether the user's guess matches the target entity you were thinking of.",
    "judgment": "Yes" or "No",
    "feedback": "Brief feedback explaining why their guess is correct or incorrect. Do not reveal the correct answer if they are wrong."
}
```
"""

ANSWER_USER = """## **Note**
- Please return in JSON format wrapped in ```json ... ```. Make sure your json content could be parsed and contains the required fields.
- Only judge if the guess is exactly correct (Yes) or wrong (No). NO partial credit.
- If they're wrong, give encouraging feedback but do not reveal the correct answer.
- If they're right, congratulate them and confirm their correct identification.
- Address the user in second person tone (e.g., "You", "Your", "You're") in your feedback.
- Your feedback should be concise and do not release anything about the target entity. Just state your judgment and encourage or congratulate the user.

### Target Entity
<target_entity>

### User's Final Guess
<user_guess>

### Your Response
```json
"""

ACTION_SYS = """## **Task**
You are a telepathic entity playing a mind reading game. The user is trying to guess what entity you are thinking of by asking yes/no questions. You should respond honestly based on the target entity you're thinking of.

## **Instructions**
1. You are thinking of a specific entity (person, object, concept, etc.) - this is the "target_entity" provided to you.
2. The user will ask questions to narrow down what you're thinking of.
3. Answer "Yes" if the question is true about your target entity.
4. Answer "No" if the question is false about your target entity.
5. Answer "Maybe" only if the question is ambiguous or you genuinely cannot determine a clear yes/no answer.
6. Be helpful and honest - the goal is for them to eventually guess correctly through good questions.

## **Example Format**

### Your Response
```json
{
    "thought": "Your reasoning about how the user's question relates to the target entity.",
    "response": "Yes" or "No" or "Maybe"
}
```
"""

ACTION_USER = """## **Note**
- Please return in JSON format wrapped in ```json ... ```. Make sure your json content could be parsed and contains the required fields.
- Answer honestly based on the target entity you're thinking of.
- Try to give clear "Yes" or "No" answers when possible. Use "Maybe" sparingly.
- Be helpful to guide the user toward the correct answer through your responses.

### Target Entity You Are Thinking Of
<target_entity>

### User's Current Question
<user_question>

### Your Response
```json
"""


def build_prompt(action_str: str, is_answer: bool = False, story: Dict[str, Any] = None, clue_history: List = None) -> List[Dict[str, Any]]:
    """Build the prompt for the LLM to evaluate agent actions in the mind reading game."""
    
    # Format clue history for display
    clue_text = ""
    if clue_history:
        for i, (question, response) in enumerate(clue_history, 1):
            clue_text += f"Q{i}: {question}\nA{i}: {response}\n\n"
    else:
        clue_text = "No questions asked yet."
    
    if is_answer:
        # For final guesses, evaluate if correct
        system_prompt = ANSWER_SYS
        user_prompt = ANSWER_USER.replace("<target_entity>", story['target_entity']).replace("<user_guess>", action_str[8:].strip()) # .replace("<clue_history>", clue_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages
    else:
        # For questions, provide yes/no responses
        system_prompt = ACTION_SYS
        user_prompt = ACTION_USER.replace("<target_entity>", story['target_entity']).replace("<user_question>", action_str[8:].strip()) # .replace("<clue_history>", clue_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return messages


def parse_llm_response(response_text: str, is_answer: bool = False) -> Tuple[str, float, Dict[str, Any]]:
    """Parse LLM response to extract feedback and score."""
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
        
    response_json = literal_eval(response_text)

    if is_answer:
        # For final guesses - either completely right (1.0) or wrong (0.0)
        judgment = response_json["judgment"]
        feedback = response_json["feedback"]
        score = 1.0 if judgment == "Yes" else 0.0
        return feedback, score, response_json
    else:
        # For questions - just return the response
        feedback = response_json["response"]
        return feedback, 0.0, response_json
    

def evaluate_action(action_str: str, story: Dict[str, Any], model_config: Dict[str, Any], clue_history: List = None) -> Tuple[str, float, Dict[str, Any], bool]:
    """
    Evaluate an agent action using the LLM in the mind reading game.
    
    Returns:
        Tuple of (feedback, score, json_response, success)
        - feedback: "Yes", "No", "Maybe" for questions; feedback text for answers
        - score: 1.0 for correct guess, 0.0 otherwise  
        - json_response: Full JSON response from LLM
        - success: whether the API call was successful
    """

    if action_str.startswith("[finish]"):
        return "Game finished.", 0.0, None, True

    is_answer = action_str.startswith("[answer]")
    
    try:
        # Create OpenAI client
        client = OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
        
        # Build the prompt
        messages = build_prompt(action_str, is_answer, story, clue_history or [])
        
        try_time = 0
        while try_time < 3:
            try:
                # Make API call using the new client format
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
        feedback, score, json_response = parse_llm_response(response_text, is_answer)
        
        return feedback, score, json_response, True
        
    except Exception as e:
        print(f"[TelepathyGym] Error evaluating action: {e}")
        if is_answer:
            return "Unable to evaluate your guess right now.", 0.0, None, False
        else:
            return "Unable to respond to your question right now.", 0.0, None, False


async def evaluate_action_async(action_str: str, story: Dict[str, Any], model_config: Dict[str, Any], clue_history: List = None) -> Tuple[str, float, Dict[str, Any], bool]:
    """
    Evaluate an agent action using the LLM in the mind reading game (async version).
    
    Returns:
        Tuple of (feedback, score, json_response, success)
        - feedback: "Yes", "No", "Maybe" for questions; feedback text for answers
        - score: 1.0 for correct guess, 0.0 otherwise  
        - json_response: Full JSON response from LLM
        - success: whether the API call was successful
    """

    if action_str.startswith("[finish]"):
        return "Game finished.", 0.0, None, True

    is_answer = action_str.startswith("[answer]")
    
    try:
        # Create AsyncOpenAI client
        client = AsyncOpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
        
        # Build the prompt
        messages = build_prompt(action_str, is_answer, story, clue_history or [])
        
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
                print(f"Error evaluating action (async): {e}")
                try_time += 1
                await asyncio.sleep(2)  # Use async sleep
                if try_time >= 3:
                    raise e
        
        # Extract response text
        response_text = response.choices[0].message.content.strip()

        # Parse the response
        feedback, score, json_response = parse_llm_response(response_text, is_answer)
        
        return feedback, score, json_response, True
        
    except Exception as e:
        print(f"[TelepathyGym] Error evaluating action (async): {e}")
        if is_answer:
            return "Unable to evaluate your guess right now.", 0.0, None, False
        else:
            return "Unable to respond to your question right now.", 0.0, None, False

