import json
import os
from typing import List, Dict, Any
import random

def load_entities() -> List[Dict[str, Any]]:
    """Load mind reading game entities."""
    # Create a collection of entities for the mind reading game
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, '..', 'data', 'all_entities_refined.json')

    with open(data_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    return entities


def get_random_entity() -> Dict[str, Any]:
    """Get a random entity from the collection."""
    entities = load_entities()
    return random.choice(entities)


def get_entity_by_title(title: str) -> Dict[str, Any]:
    """Get a specific entity by its title."""
    entities = load_entities()
    for entity in entities:
        if entity["title"].lower() == title.lower() or entity["id"].lower() == title.lower():
            return entity
    raise ValueError(f"Entity with title '{title}' not found")


# Legacy aliases for backward compatibility
load_stories = load_entities
get_random_story = get_random_entity
get_story_by_title = get_entity_by_title
