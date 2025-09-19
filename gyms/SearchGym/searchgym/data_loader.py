import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_questions() -> List[Dict[str, Any]]:
    """Load all questions from the data file."""
    data_path = Path(__file__).parent / "data" / "questions.json"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Questions data file not found: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    return questions


def get_question_by_id(question_id: str) -> Dict[str, Any]:
    """Get a specific question by its ID."""
    questions = load_questions()
    
    for question in questions:
        if question["id"] == question_id:
            return question
    
    raise ValueError(f"Question with ID '{question_id}' not found")


def get_random_question(category: Optional[str] = None, difficulty: Optional[str] = None) -> Dict[str, Any]:
    """Get a random question, optionally filtered by category or difficulty."""
    questions = load_questions()
    
    # Filter by category if specified
    if category:
        questions = [q for q in questions if q.get("category") == category]
    
    # Filter by difficulty if specified
    if difficulty:
        questions = [q for q in questions if q.get("difficulty") == difficulty]
    
    if not questions:
        raise ValueError(f"No questions found matching the criteria (category: {category}, difficulty: {difficulty})")
    
    return random.choice(questions)


def answer_normalize(answer: str) -> str:
    """Normalize the answer to a standard format."""
    # lower, without space, only keep alphanumeric characters
    return "".join(char.lower() for char in answer if char.isalnum())


def compare_answers(submitted_answer: str, correct_answer: str) -> bool:
    """
    Compare a submitted answer with the correct answer.
    
    Args:
        submitted_answer: The answer submitted by the agent
        correct_answer: The ground truth answer
        case_sensitive: Whether to perform case-sensitive comparison
        
    Returns:
        True if answers match, False otherwise
    """
    submitted_answer = answer_normalize(submitted_answer)
    correct_answer = answer_normalize(correct_answer)
    
    # Check for exact match
    if submitted_answer == correct_answer:
        return True
    
    # # Check if submitted answer is contained in correct answer (for partial matches)
    # # This allows for some flexibility in answer matching
    # if submitted_answer in correct_answer or correct_answer in submitted_answer:
    #     return True
    
    return False 