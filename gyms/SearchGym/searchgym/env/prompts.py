from openai import OpenAI, AsyncOpenAI
from typing import Dict, Any, Tuple, List
from ast import literal_eval
from copy import deepcopy
import json
import time
import asyncio


ANSWER_SYS = """## **Task**
You are asked to judge whether the answer for a question is correct or not.

## **Instructions**
1. You will be provided with the question, the model's answer, and the correct answer.
2. If the answer is exactly the same, or a clearly equivalent/synonymous identification of the correct answer, return "Yes". Please base your answer judgment on the given question scenario, instead of just comparing the answers.
3. If the answer is wrong, return "No".
4. In your feedback, you could provide a succinct explanation for your judgment, but you should never reveal the correct answer.
5. In your feedback, please use second person tone (e.g., "You", "Your", "You're").
6. In your feedback please do not give any hint or any information about the correct answer.

## **Example Format**

### Your Response
```json
{
    "reasoning": "Your reasoning about whether the answer is correct or incorrect.",
    "judgment": "Yes" or "No",
    "feedback": "Brief feedback explaining why the answer is correct or incorrect. Do not reveal the correct answer if they are wrong."
}
```
"""

ANSWER_USER = """### Question
<question>

### Model's Answer
<model_answer>

### Correct Answer
<correct_answer>

### Your Response
```json
"""



def parse_llm_response(response_text: str):
    """Parse LLM response to extract feedback and score."""
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
        
    response_json = literal_eval(response_text)

    reasoning = response_json["reasoning"]
    judgment = response_json["judgment"]
    feedback = response_json["feedback"]

    return feedback, judgment, reasoning
    

def evaluate_answer(correct_answer: str, model_answer: str, question: str, model_config: Dict[str, Any]):
    """
    Evaluate the answer using the LLM.
    
    Returns:
        Tuple of (feedback, score, reasoning)
        - feedback: feedback text for answers
        - score: 
        - reasoning: reasoning for the answer
    """

    try:
        # Create OpenAI client
        client = OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])

        user_prompt = ANSWER_USER.replace("<question>", question).replace("<model_answer>", model_answer).replace("<correct_answer>", correct_answer)
        messages = [{"role": "system", "content": ANSWER_SYS}, {"role": "user", "content": user_prompt}]
        
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
        feedback, judgment, reasoning = parse_llm_response(response_text)
        
        return feedback, judgment, reasoning
        
    except Exception as e:
        print(f"[SearchGym] Error evaluating answer: {e}")
        return "Unable to evaluate your answer right now.", "Error", "Unable to evaluate your answer right now."


async def evaluate_answer_async(correct_answer: str, model_answer: str, question: str, model_config: Dict[str, Any]):
    """
    Evaluate the answer using the LLM (async version).
    
    Returns:
        Tuple of (feedback, score, reasoning)
        - feedback: feedback text for answers
        - score: 
        - reasoning: reasoning for the answer
    """

    try:
        # Create AsyncOpenAI client
        client = AsyncOpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])

        user_prompt = ANSWER_USER.replace("<question>", question).replace("<model_answer>", model_answer).replace("<correct_answer>", correct_answer)
        messages = [{"role": "system", "content": ANSWER_SYS}, {"role": "user", "content": user_prompt}]
        
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
        feedback, judgment, reasoning = parse_llm_response(response_text)
        
        return feedback, judgment, reasoning
        
    except Exception as e:
        print(f"[SearchGym] Error evaluating answer (async): {e}")
        return "Unable to evaluate your answer right now.", "Error", "Unable to evaluate your answer right now."

