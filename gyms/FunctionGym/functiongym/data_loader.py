import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_functions() -> List[Dict[str, Any]]:
    """Load all function problems from the data file."""
    data_path = Path(__file__).parent / "data" / "functions.json"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Functions data file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        functions = json.load(f)
    
    return functions


def get_function_by_id(function_id: str) -> Dict[str, Any]:
    """Get a specific function problem by its ID."""
    functions = load_functions()
    
    for function in functions:
        if function["id"] == function_id:
            return function
    
    raise ValueError(f"Function with ID '{function_id}' not found")


def get_random_function() -> Dict[str, Any]:
    """Get a random function problem."""
    functions = load_functions()
    return random.choice(functions)


def evaluate_function(numbers: List[float], rule: str) -> float:
    """Evaluate the function rule with given numbers."""
    try:
        # Replace a, b, c, d with actual numbers
        a, b, c, d = numbers
        # Use eval safely with only the numbers in scope
        local_vars = {'a': a, 'b': b, 'c': c, 'd': d}
        return eval(rule, {"__builtins__": {}}, local_vars)
    except Exception as e:
        raise ValueError(f"Error evaluating function rule '{rule}' with numbers {numbers}: {e}")


def compare_answers(submitted_answer: float, correct_answer: float, tolerance: float = 1e-6) -> bool:
    """
    Compare a submitted numerical answer with the correct answer.
    
    Args:
        submitted_answer: The answer submitted by the agent
        correct_answer: The ground truth answer
        tolerance: Numerical tolerance for floating point comparison
        
    Returns:
        True if answers match within tolerance, False otherwise
    """
    try:
        return abs(float(submitted_answer) - float(correct_answer)) <= tolerance
    except (ValueError, TypeError):
        return False 