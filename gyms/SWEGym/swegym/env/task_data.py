import json
from pathlib import Path
from typing import Any, Dict, List

REQUIRED_FIELDS = ("id", "instruction", "oracle_intents", "completeness_goals")


def get_data_path() -> Path:
    """Get the path to the data directory."""
    return Path(__file__).parent.parent / "data"


def load_data_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load and validate one split of the task dataset."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in data file: {e}")
    except OSError as e:
        raise ValueError(f"Error loading tasks: {e}")

    tasks = list(payload.values()) if isinstance(payload, dict) else payload

    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Each task must be a dictionary")

        for field in REQUIRED_FIELDS:
            if field not in task:
                raise ValueError(f"Task missing required field: {field}")

        if not isinstance(task["oracle_intents"], list):
            raise ValueError("oracle_intents must be a list")
        for intent in task["oracle_intents"]:
            for field in ("description", "importance"):
                if field not in intent:
                    raise ValueError(f"Oracle intent missing required field: {field}")
            if int(intent["importance"]) not in (1, 2, 3):
                raise ValueError(f"Importance must be 1, 2, or 3, got: {intent['importance']}")

        if not isinstance(task["completeness_goals"], list) or not task["completeness_goals"]:
            raise ValueError("completeness_goals must be a non-empty list")
        for goal in task["completeness_goals"]:
            for field in ("statement", "weight"):
                if field not in goal:
                    raise ValueError(f"Completeness goal missing required field: {field}")

    return tasks


def load_tasks(split: str = None) -> List[Dict[str, Any]]:
    """
    Load tasks from the SWE-Together export.

    Args:
        split: "train", "test", or None for both

    Returns:
        List of task dictionaries with instruction, oracle intents and rubric
    """
    splits = [split] if split in ("train", "test") else ["train", "test"]
    tasks = []
    for name in splits:
        tasks.extend(load_data_file(get_data_path() / f"swe_tasks_{name}.json"))
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

    for task in load_tasks():
        if task["id"] == task_id:
            return task

    raise ValueError(f"Task with ID '{task_id}' not found")


def get_task_statistics() -> Dict[str, Any]:
    """
    Get statistics about the loaded tasks.

    Returns:
        Dictionary with task statistics
    """
    stats = {}
    for split in ("train", "test"):
        tasks = load_tasks(split)
        intents = [i for t in tasks for i in t["oracle_intents"]]
        importance = {1: 0, 2: 0, 3: 0}
        for intent in intents:
            importance[int(intent["importance"])] += 1
        stats[split] = {
            "total_tasks": len(tasks),
            "total_intents": len(intents),
            "average_intents_per_task": len(intents) / len(tasks) if tasks else 0,
            "importance_distribution": importance,
            "average_goals_per_task": (
                sum(len(t["completeness_goals"]) for t in tasks) / len(tasks) if tasks else 0
            ),
        }
    return stats
