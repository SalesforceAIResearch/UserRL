import json
from pathlib import Path
from typing import List, Dict, Any
import random


def load_statements() -> List[Dict[str, Any]]:
    """Load all statements from the JSON file."""
    data_path = Path(__file__).parent.parent / "data" / "all_statements_refined.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        statements = json.load(f)
    
    return statements


def get_statement_by_id(statement_id: str) -> Dict[str, Any]:
    """Get a specific statement by its ID."""
    statements = load_statements()
    
    for statement in statements:
        if statement["id"] == statement_id:
            return statement
    
    raise ValueError(f"Statement with id '{statement_id}' not found")


def get_random_statement() -> Dict[str, Any]:
    """Get a random statement."""
    statements = load_statements()
    return random.choice(statements)


def get_statements_by_ids(statement_ids: List[str]) -> List[Dict[str, Any]]:
    """Get multiple statements by their IDs."""
    statements = load_statements()
    result = []
    
    for statement_id in statement_ids:
        for statement in statements:
            if statement["id"] == statement_id:
                result.append(statement)
                break
        else:
            raise ValueError(f"Statement with id '{statement_id}' not found")
    
    return result 