import json
import os
from typing import List, Dict, Any
from pathlib import Path


def get_data_path() -> Path:
    """Get the path to the data directory."""
    current_dir = Path(__file__).parent.parent
    return current_dir / "data"

def load_data_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load a data file from the data directory."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        
        # Validate task structure
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError("Each task must be a dictionary")
            
            required_fields = ["id", "task", "missing_details"]
            for field in required_fields:
                if field not in task:
                    raise ValueError(f"Task missing required field: {field}")
            
            # Validate missing_details structure
            if not isinstance(task["missing_details"], list):
                raise ValueError("missing_details must be a list")
            
            for detail in task["missing_details"]:
                if not isinstance(detail, dict):
                    raise ValueError("Each missing detail must be a dictionary")
                
                required_detail_fields = ["description", "importance"]
                for field in required_detail_fields:
                    if field not in detail:
                        raise ValueError(f"Missing detail missing required field: {field}")
                
                # Validate importance is a valid number
                try:
                    importance = int(detail["importance"])
                    if importance not in [1, 2, 3]:
                        raise ValueError(f"Importance must be 1, 2, or 3, got: {importance}")
                except (ValueError, TypeError):
                    raise ValueError(f"Invalid importance value: {detail['importance']}")
        
        return tasks
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in data file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading tasks: {e}")


def load_tasks() -> List[Dict[str, Any]]:
    """
    Load all tasks from the refined intentions JSON file.
    
    Returns:
        List of task dictionaries with missing details and metadata
    """
    data_path = get_data_path() / "all_intentions_train.json"
    test_path = get_data_path() / "all_intentions_test.json"
    
    data = load_data_file(data_path)
    test = load_data_file(test_path)

    return data + test


def get_task_by_id(task_id: str) -> Dict[str, Any]:
    """
    Get a specific task by ID.
    
    Args:
        task_id: The unique identifier for the task
        
    Returns:
        Task dictionary
        
    Raises:
        ValueError: If task ID is not found
    """
    if not task_id:
        raise ValueError("Task ID cannot be empty")
    
    tasks = load_tasks()
    
    for task in tasks:
        if task["id"] == task_id:
            return task
    
    raise ValueError(f"Task with ID '{task_id}' not found")


def get_tasks_by_category(category: str) -> List[Dict[str, Any]]:
    """
    Get all tasks in a specific category.
    
    Args:
        category: The category name to filter by
        
    Returns:
        List of task dictionaries in the specified category
    """
    if not category:
        raise ValueError("Category cannot be empty")
    
    tasks = load_tasks()
    return [task for task in tasks if task.get("category", "").lower() == category.lower()]


def get_task_statistics() -> Dict[str, Any]:
    """
    Get statistics about the loaded tasks.
    
    Returns:
        Dictionary with task statistics
    """
    tasks = load_tasks()
    
    # Count by category
    categories = {}
    importance_counts = {1: 0, 2: 0, 3: 0}
    total_missing_details = 0
    
    for task in tasks:
        # Category stats
        category = task.get("category", "Unknown")
        categories[category] = categories.get(category, 0) + 1
        
        # Missing details stats
        missing_details = task.get("missing_details", [])
        total_missing_details += len(missing_details)
        
        for detail in missing_details:
            importance = int(detail.get("importance", 1))
            importance_counts[importance] += 1
    
    return {
        "total_tasks": len(tasks),
        "categories": categories,
        "total_missing_details": total_missing_details,
        "average_details_per_task": total_missing_details / len(tasks) if tasks else 0,
        "importance_distribution": importance_counts
    } 