"""
Data loader for AlfworldGym.

Unlike SearchGym which pre-loads questions, AlfworldGym generates new scenarios
on each reset() call from the alfworld environment. This module provides
utilities for task information and scenario management.
"""

from typing import Dict, Any, Optional
import re


def parse_task_description(observation: str) -> Dict[str, Any]:
    """
    Parse the task description from alfworld observation.
    
    Args:
        observation: Raw observation string from alfworld environment
        
    Returns:
        Dictionary containing parsed task information
    """
    # Extract task description
    task = ""
    if "Your task is to:" in observation:
        task = observation.split("Your task is to:")[-1].strip()
        # Remove any trailing punctuation or whitespace
        task = task.split('.')[0].strip()
    
    # Extract the initial observation (everything after "ALFRED!")
    initial_obs = ""
    if "ALFRED! =-" in observation:
        initial_obs = observation.split("ALFRED! =-")[1].strip()
        # Remove the task description from initial observation
        if "Your task is to:" in initial_obs:
            initial_obs = initial_obs.split("Your task is to:")[0].strip()
    
    return {
        "task": task,
        "initial_observation": initial_obs,
        "full_observation": observation
    }


def extract_action_from_text(action_text: str) -> str:
    """
    Extract and validate action from text input.
    
    Args:
        action_text: Raw action text that should start with [action] or [finish]
        
    Returns:
        Clean action string for alfworld
        
    Raises:
        ValueError: If action format is invalid
    """
    action_text = action_text.strip()
    
    if action_text.startswith("[action]"):
        action = action_text[8:].strip()  # Remove "[action]" prefix
        if not action:
            raise ValueError("Action is empty. Please provide a valid action.")
        return action
    elif action_text.startswith("[finish]"):
        # For finish actions, we can return a special marker or the text after [finish]
        finish_text = action_text[8:].strip()
        return finish_text if finish_text else "finish"
    else:
        raise ValueError("Invalid action format. Actions must start with [action] or [finish].")


def is_valid_alfworld_action(action: str) -> bool:
    """
    Check if an action is a valid alfworld action.
    
    Args:
        action: Action string to validate
        
    Returns:
        True if action appears to be valid for alfworld
    """
    if not action or not isinstance(action, str):
        return False
    
    action = action.strip().lower()
    
    # List of common alfworld action prefixes
    valid_prefixes = [
        "go to", "take", "move", "open", "close", "examine", "look",
        "use", "heat", "clean", "cool", "slice", "put"
    ]
    
    # Check if action starts with any valid prefix
    for prefix in valid_prefixes:
        if action.startswith(prefix):
            return True
    
    # Special cases
    if action in ["look", "inventory", "help"]:
        return True
    
    return False


def normalize_action(action: str) -> str:
    """
    Normalize action text for consistency.
    
    Args:
        action: Raw action string
        
    Returns:
        Normalized action string
    """
    if not action:
        return ""
    
    # Basic cleanup
    action = action.strip()
    
    # Remove extra whitespace
    action = re.sub(r'\s+', ' ', action)
    
    # Convert to lowercase for consistency (alfworld seems to handle this)
    # action = action.lower()
    
    return action


def format_alfworld_feedback(observation: str, score: float, done: bool, info: Dict[str, Any]) -> str:
    """
    Format alfworld environment feedback into a user-friendly string.
    
    Args:
        observation: Raw observation from alfworld
        score: Current score (0.0 or 1.0 typically)
        done: Whether episode is done
        info: Additional info from alfworld
        
    Returns:
        Formatted feedback string
    """
    feedback = observation.strip()
    
    # Add score information if relevant
    if done and score > 0:
        feedback += f"\n\n✅ Task completed successfully! (Score: {score})"
    elif done and score == 0:
        feedback += f"\n\n❌ Task failed or incomplete. (Score: {score})"
    
    return feedback 