"""
Prompts and evaluation functions for IntentionGym.

This module contains the prompts and evaluation logic for determining
how well user questions clarify missing details in vague tasks.
"""

import json
import openai
import asyncio
from typing import Dict, Any, List, Tuple, Optional
import re

# Response generation prompt - DOES NOT know about missing details to prevent hacking
RESPONSE_GENERATION_PROMPT = """
You are a person who has posted a vague request for help and is now responding to someone who is trying to help clarify your needs.

ORIGINAL TASK REQUEST: {task}
CATEGORY: {category}

CURRENT QUESTION FROM HELPER: {question}

CONVERSATION HISTORY:
{conversation_history}

Your job is to respond naturally as the person who originally made the request. Follow these guidelines:

1. If the question is asking about your specific preferences for this task:
   - Provide an authentic and coherent response
   - Share realistic preferences that someone might have for this type of task
   - Be conversational and natural

2. If the question is NOT directly about your preferences for this task:
   - Try to answer helpfully if you can
   - Guide the conversation back to clarifying what you need for your task
   - Be polite but redirect: "That's interesting, but what I'm really trying to figure out is..."
   - Do NOT provide what missing details need to be clarified or give any examples.
   - Do NOT provide concrete help or solutions - you're the one seeking help!

Please respond in the following json format:
{{
    "thought": "Your thought process about whether the question is about your preferences and how to respond",
    "response": "Your natural conversational response"
}}


IMPORTANT: 
- Respond only as the person seeking help, not as an evaluator
- Be natural and conversational
- Don't reveal any "ground truth" or act like you know what details are missing
- Just respond authentically as someone who made this request
"""

# Evaluation prompt - ONLY calculates reward, does not generate response
EVALUATION_PROMPT = """
You are evaluating how well a user's question addresses missing details in a vague task.

TASK: {task}
CATEGORY: {category}

MISSING DETAILS THAT NEED TO BE CLARIFIED:
{details_list}

CONVERSATION HISTORY:
{conversation_history}

LATEST USER'S QUESTION: {question}

Your job is to evaluate which (if any) of the missing details are addressed by the latest question.

Rules for evaluation:
- If the question is NOT related to clarifying the user's intent or task requirements, covered_detail_indices should be an empty list
- For each missing detail that is directly addressed by the question, note its index
- A question "addresses" a detail if it would help reveal the information needed for that detail

Please respond in the following json format:
{{
    "analysis": "Brief explanation of what missing details (if any) were covered by this question",
    "is_task_related": true/false (whether the question is about clarifying the task requirements)
    "covered_detail_indices": [list of indices from the missing details list that this question addresses],
}}

IMPORTANT: You are ONLY evaluating, not generating responses. Focus solely on which details are addressed.
"""

def build_response_generation_prompt(
    question: str,
    task: Dict[str, Any],
    conversation_history: List[Tuple[str, str]]
) -> str:
    """
    Build the response generation prompt (without missing details knowledge).
    
    Args:
        question: The user's question
        task: The current task dictionary
        conversation_history: Previous (question, response) pairs
        
    Returns:
        Formatted prompt for response generation
    """
    
    # Build conversation context
    context = ""
    if conversation_history:
        context = "\n\nPrevious conversation:\n"
        for i, (prev_q, prev_r) in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
            context += f"Helper: {prev_q}\nYou: {prev_r}\n"
    
    prompt = RESPONSE_GENERATION_PROMPT.format(
        task=task,
        category=task.get('category', 'General'),
        question=question,
        conversation_history=context if context.strip() else "This is the first question in our conversation."
    )

    return prompt

def build_evaluation_prompt(
    question: str,
    task: Dict[str, Any],
    conversation_history: List[Tuple[str, str]],
    remaining_missing_details: List[Dict[str, Any]]
) -> str:
    """
    Build the evaluation prompt for analyzing a user's question.
    
    Args:
        question: The user's question
        task: The current task dictionary
        conversation_history: Previous (question, response) pairs
        remaining_missing_details: List of missing details not yet covered
        
    Returns:
        Formatted prompt for evaluation
    """
    
    # Build conversation context
    context = ""
    if conversation_history:
        context = "\n\nPrevious conversation:\n"
        for i, (prev_q, prev_r) in enumerate(conversation_history[-3:], 1):  # Last 3 exchanges
            context += f"Q{i}: {prev_q}\nA{i}: {prev_r}\n"
    
    # Format remaining missing details with indices
    details_list = ""
    for i, detail in enumerate(remaining_missing_details):
        importance_label = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}[int(detail.get("importance", 1))]
        details_list += f"{i}. [{importance_label}] {detail['description']}\n"
        if "inquiry" in detail:
            details_list += f"   Example question type expected: {detail['inquiry']}\n"
        details_list += "\n"
    
    prompt = EVALUATION_PROMPT.format(
        task=task,
        category=task.get('category', 'General'),
        details_list=details_list if details_list.strip() else "None - all details have been covered!",
        question=question,
        conversation_history=context if context.strip() else "None"
    )

    return prompt

def generate_response(
    question: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    conversation_history: List[Tuple[str, str]]
) -> str:
    """
    Generate a natural response to the user's question without knowing missing details.
    
    Args:
        question: The user's question
        task: Current task dictionary
        model_config: LLM configuration
        conversation_history: Previous conversation
        
    Returns:
        Natural conversational response
    """
    try:
        if question.startswith("[action]") or question.startswith("[answer]"):
            question = question[8:].strip()
        
        # Build response generation prompt (no missing details)
        prompt = build_response_generation_prompt(question, task, conversation_history)
        
        # Set up OpenAI client
        client = openai.OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
        
        # Make API call
        response = client.chat.completions.create(
            model=model_config.get("model_name", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are helping someone respond naturally to questions about their request for help."},
                {"role": "user", "content": prompt}
            ],
            temperature=model_config.get("temperature", 0.7),  # Slightly higher temp for natural responses
            max_tokens=model_config.get("max_tokens", 1024),
            timeout=model_config.get("timeout", 10)
        )
        
        response_text = response.choices[0].message.content.strip()
        return parse_response_generation_response(response_text)
    
    except Exception as e:
        # Fallback response
        fallback_responses = [
            "That's an interesting question. Let me think about that.",
            "Good point! I hadn't considered that aspect.",
            "Thanks for asking - that helps me clarify what I need.",
            "That's definitely something I should specify.",
            "I appreciate you helping me think through this."
        ]
        import random
        return random.choice(fallback_responses)

async def generate_response_async(
    question: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    conversation_history: List[Tuple[str, str]]
) -> str:
    """
    Async version of generate_response.
    
    Args:
        question: The user's question
        task: Current task dictionary
        model_config: LLM configuration
        conversation_history: Previous conversation
        
    Returns:
        Natural conversational response
    """
    try:
        if question.startswith("[action]") or question.startswith("[answer]"):
            question = question[8:].strip()
        
        # Build response generation prompt (no missing details)
        prompt = build_response_generation_prompt(question, task, conversation_history)
        
        # Set up OpenAI async client
        client = openai.AsyncOpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
        
        # Make async API call
        response = await client.chat.completions.create(
            model=model_config.get("model_name", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are helping someone respond naturally to questions about their request for help."},
                {"role": "user", "content": prompt}
            ],
            temperature=model_config.get("temperature", 0.7),  # Slightly higher temp for natural responses
            max_tokens=model_config.get("max_tokens", 1024),
            timeout=model_config.get("timeout", 10)
        )
        
        response_text = response.choices[0].message.content.strip()
        return parse_response_generation_response(response_text)
    
    except Exception as e:
        # Fallback response
        fallback_responses = [
            "That's an interesting question. Let me think about that.",
            "Good point! I hadn't considered that aspect.",
            "Thanks for asking - that helps me clarify what I need.",
            "That's definitely something I should specify.",
            "I appreciate you helping me think through this."
        ]
        import random
        return random.choice(fallback_responses)

def evaluate_question_coverage(
    question: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    conversation_history: List[Tuple[str, str]],
    remaining_missing_details: List[Dict[str, Any]]
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Evaluate which missing details are covered by the user's question (reward calculation only).
    
    Args:
        question: The user's question
        task: Current task dictionary
        model_config: LLM configuration
        conversation_history: Previous conversation
        remaining_missing_details: Uncovered missing details
        
    Returns:
        Tuple of (covered_detail_indices, reward_info)
    """
    try:
        if question.startswith("[action]") or question.startswith("[answer]"):
            question = question[8:].strip()
        
        # Build evaluation prompt
        prompt = build_evaluation_prompt(question, task, conversation_history, remaining_missing_details)
        
        # Set up OpenAI client
        client = openai.OpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
        
        # Make API call
        response = client.chat.completions.create(
            model=model_config.get("model_name", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are evaluating question quality for task clarification. Focus only on evaluation, not response generation."},
                {"role": "user", "content": prompt}
            ],
            temperature=model_config.get("temperature", 0.0),  # Low temp for consistent evaluation
            max_tokens=model_config.get("max_tokens", 1024),
            timeout=model_config.get("timeout", 10)
        )
        
        response_text = response.choices[0].message.content
        return parse_evaluation_response(response_text)
    
    except Exception as e:
        print(f"[IntentionGym] Error evaluating question coverage: {e}")
        return build_fallback_evaluation(question, task)


async def evaluate_question_coverage_async(
    question: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    conversation_history: List[Tuple[str, str]],
    remaining_missing_details: List[Dict[str, Any]]
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Async version of evaluate_question_coverage.
    
    Args:
        question: The user's question
        task: Current task dictionary
        model_config: LLM configuration
        conversation_history: Previous conversation
        remaining_missing_details: Uncovered missing details
        
    Returns:
        Tuple of (covered_detail_indices, reward_info)
    """
    try:
        if question.startswith("[action]") or question.startswith("[answer]"):
            question = question[8:].strip()
        
        # Build evaluation prompt
        prompt = build_evaluation_prompt(question, task, conversation_history, remaining_missing_details)
        
        # Set up OpenAI async client
        client = openai.AsyncOpenAI(api_key=model_config["api_key"], base_url=model_config["base_url"])
        
        # Make async API call
        response = await client.chat.completions.create(
            model=model_config.get("model_name", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are evaluating question quality for task clarification. Focus only on evaluation, not response generation."},
                {"role": "user", "content": prompt}
            ],
            temperature=model_config.get("temperature", 0.0),  # Low temp for consistent evaluation
            max_tokens=model_config.get("max_tokens", 1024),
            timeout=model_config.get("timeout", 10)
        )
        
        response_text = response.choices[0].message.content
        return parse_evaluation_response(response_text)
    
    except Exception as e:
        print(f"[IntentionGym] Error evaluating question coverage: {e}")
        return build_fallback_evaluation(question, task)


def parse_response_generation_response(response_text: str) -> str:
    """
    Parse the LLM's response generation response.
    """
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            return data.get("response", "")
        else:
            return response_text
    except (json.JSONDecodeError, ValueError) as e:
        return response_text


def parse_evaluation_response(response_text: str) -> Tuple[List[int], Dict[str, Any]]:
    """
    Parse the LLM's evaluation response.
    
    Args:
        response_text: Raw response from the LLM
        
    Returns:
        Tuple of (covered_indices, reward_info)
    """
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            data = json.loads(json_str)
            
            covered_indices = data.get("covered_detail_indices", [])
            analysis = data.get("analysis", "")
            is_task_related = data.get("is_task_related", True)
            
            # Validate covered_indices
            if not isinstance(covered_indices, list):
                covered_indices = []
            else:
                covered_indices = [int(i) for i in covered_indices if isinstance(i, (int, str)) and str(i).isdigit()]
            
            # If not task related, set covered_indices to empty (reward = 0)
            if not is_task_related:
                covered_indices = []
            
            reward_info = {
                "method": "llm_evaluation",
                "analysis": analysis,
                "is_task_related": is_task_related,
                "raw_response": response_text[:200] + "..." if len(response_text) > 200 else response_text
            }
            
            return covered_indices, reward_info
        else:
            # No JSON found, assume no coverage
            return [], {"method": "plain_text", "raw_response": response_text}
    
    except (json.JSONDecodeError, ValueError) as e:
        # JSON parsing failed, assume no coverage
        return [], {"method": "json_parse_error", "error": str(e)}


# Legacy functions for backward compatibility
def evaluate_question(
    question: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    conversation_history: List[Tuple[str, str]],
    remaining_missing_details: List[Dict[str, Any]]
) -> Tuple[str, List[int], Dict[str, Any]]:
    """
    Legacy function - now uses the new two-step approach.
    
    Args:
        question: The user's question
        task: Current task dictionary
        model_config: LLM configuration
        conversation_history: Previous conversation
        remaining_missing_details: Uncovered missing details
        
    Returns:
        Tuple of (response, covered_detail_indices, reward_info)
    """
    try:
        # Step 1: Generate response (without knowing missing details)
        response = generate_response(question, task, model_config, conversation_history)
        
        # Step 2: Evaluate coverage (with missing details knowledge)
        covered_indices, reward_info = evaluate_question_coverage(
            question, task, model_config, conversation_history, remaining_missing_details
        )
        
        return response, covered_indices, reward_info
    
    except Exception as e:
        print(f"[IntentionGym] Error evaluating question coverage: {e}")
        return build_fallback_response(question, task)


async def evaluate_question_async(
    question: str,
    task: Dict[str, Any],
    model_config: Dict[str, Any],
    conversation_history: List[Tuple[str, str]],
    remaining_missing_details: List[Dict[str, Any]]
) -> Tuple[str, List[int], Dict[str, Any]]:
    """
    Legacy async function - now uses the new two-step approach.
    
    Args:
        question: The user's question
        task: Current task dictionary
        model_config: LLM configuration
        conversation_history: Previous conversation
        remaining_missing_details: Uncovered missing details
        
    Returns:
        Tuple of (response, covered_detail_indices, reward_info)
    """
    try:
        # Step 1: Generate response (without knowing missing details)
        response = await generate_response_async(question, task, model_config, conversation_history)
        
        # Step 2: Evaluate coverage (with missing details knowledge)
        covered_indices, reward_info = await evaluate_question_coverage_async(
            question, task, model_config, conversation_history, remaining_missing_details
        )
        
        return response, covered_indices, reward_info
    
    except Exception as e:
        print(f"[IntentionGym] Error evaluating question coverage: {e}")
        return build_fallback_response(question, task)


def build_fallback_evaluation(question: str, task: Dict[str, Any]) -> Tuple[List[int], Dict[str, Any]]:
    """
    Generate a fallback evaluation when LLM is not available.
    
    Args:
        question: The user's question
        task: The current task dictionary
        
    Returns:
        Tuple of (covered_indices, reward_info)
    """
    covered_indices = []
    question_lower = question.lower()
    
    # Basic keyword matching (very simple fallback)
    keywords_found = []
    if any(word in question_lower for word in ["when", "time", "duration", "long"]):
        keywords_found.append("timing")
    if any(word in question_lower for word in ["budget", "cost", "money", "price"]):
        keywords_found.append("budget")
    if any(word in question_lower for word in ["where", "location", "place"]):
        keywords_found.append("location")
    
    reward_info = {
        "method": "fallback",
        "keywords_found": keywords_found,
        "fallback_used": True,
        "is_task_related": len(keywords_found) > 0
    }
    
    return covered_indices, reward_info


def build_fallback_response(question: str, task: Dict[str, Any]) -> Tuple[str, List[int], Dict[str, Any]]:
    """
    Generate a fallback response when LLM is not available (legacy function for backward compatibility).
    
    Args:
        question: The user's question
        task: The current task dictionary
        
    Returns:
        Tuple of (response, covered_indices, reward_info)
    """
    fallback_responses = [
        "That's an interesting question. Let me think about that.",
        "Good point! I hadn't considered that aspect.",
        "Thanks for asking - that helps me clarify what I need.",
        "That's definitely something I should specify.",
        "I appreciate you helping me think through this."
    ]
    
    import random
    response = random.choice(fallback_responses)
    covered_indices, reward_info = build_fallback_evaluation(question, task)
    
    return response, covered_indices, reward_info