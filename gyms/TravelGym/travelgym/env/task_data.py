"""
Task data management for TravelGym.

This module handles loading and managing travel planning scenarios
with user preferences and evaluation data.
"""

import json
import os
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

np.random.seed(42)  # For reproducibility in shuffling options

def get_data_path() -> Path:
    """Get the path to the data directory."""
    current_dir = Path(__file__).parent.parent
    return current_dir / "data"


def load_data_file(file_path: Path) -> Dict[str, Any]:
    """Load a data file from the data directory."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            scenarios = json.load(f)
        return scenarios
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in data file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading scenarios: {e}")


def load_tasks(config=None) -> List[Dict[str, Any]]:
    """
    Load all travel planning scenarios from the data file.
    
    Returns:
        List of scenario dictionaries with preferences and metadata
    """

    all_scenarios = {}

    for wanted_num in ["22", "33", "44", "2222", "233", "334", "444", "333"]:
        data_path = get_data_path() / f"travelgym_data_{wanted_num}.json"
        scenarios = load_data_file(data_path)
        # incorporate the dict into the large dictionary
        all_scenarios.update(scenarios)

    # Convert to list format for compatibility
    tasks = []
    for scenario_key, scenario_data in all_scenarios.items():
        task = {
            "id": scenario_key,
            "scenario": scenario_data["scenario"],
            "category": "travel_planning",
            "dimensions": scenario_data["dimensions"],
            "wanted_num": scenario_data["wanted_num"],
            "initial_desc": scenario_data["initial_description"],
            "preferences": scenario_data,  # Keep full scenario data for preference access
            "all_options": {},  # Will be populated below
            "arguments": {}  # Will be populated below
        }
        
        # Preprocess data for each dimension
        dimensions = scenario_data["dimensions"]
        preferences_data = scenario_data
        
        for dimension in dimensions:
            if dimension in preferences_data:
                dim_data = preferences_data[dimension]
                
                # Create all_options by combining and shuffling correct, wrong, noise options
                if "options" in dim_data:
                    options_data = dim_data["options"]
                    correct_options = options_data.get("correct", [])
                    wrong_options = options_data.get("wrong", [])
                    noisy_options = options_data.get("noise", [])

                    np.random.shuffle(correct_options)
                    np.random.shuffle(wrong_options)
                    np.random.shuffle(noisy_options)
                    # Limit the number of wrong and noise options if specified in config
                    noisy_options = noisy_options[:config.noise_choice_number] if config and hasattr(config, 'noise_choice_number') else noisy_options
                    wrong_options = wrong_options[:config.wrong_choice_number] if config and hasattr(config, 'wrong_choice_number') else wrong_options
                    
                    all_options = correct_options + wrong_options + noisy_options
                    for option in all_options:
                        option.pop("type", None)
                        option.pop("reason", None)
                    
                    # randomly shuffle the options
                    np.random.shuffle(all_options)
                    task["all_options"][dimension] = all_options
                else:
                    task["all_options"][dimension] = []
                
                # Set up arguments structure for search judging
                if "arguments" in dim_data:
                    task["arguments"][dimension] = dim_data["arguments"]
        
        tasks.append(task)
    
    return tasks


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


def get_task_statistics() -> Dict[str, Any]:
    """
    Get statistics about the loaded tasks.
    
    Returns:
        Dictionary with task statistics
    """
    tasks = load_tasks()
    
    # Count by category
    categories = {}
    difficulty_counts = {}
    total_preferences = 0
    dimension_counts = {}
    
    for task in tasks:
        # Category stats
        category = task.get("category", "Unknown")
        categories[category] = categories.get(category, 0) + 1
        
        # Difficulty stats
        difficulty = task.get("difficulty", "unknown")
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        # Preferences stats
        preferences_data = task.get("preferences", {})
        for dimension in task.get("dimensions", []):
            if dimension in preferences_data:
                prefs = preferences_data[dimension].get("preferences", [])
                total_preferences += len(prefs)
                dimension_counts[dimension] = dimension_counts.get(dimension, 0) + len(prefs)
    
    return {
        "total_tasks": len(tasks),
        "categories": categories,
        "difficulty_distribution": difficulty_counts,
        "total_preferences": total_preferences,
        "average_preferences_per_task": total_preferences / len(tasks) if tasks else 0,
        "dimension_distribution": dimension_counts
    }