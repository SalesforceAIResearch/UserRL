import json
import os
from typing import List, Dict, Any
import random

def load_stories() -> List[Dict[str, Any]]:
    """Load story scenarios from the refined JSON file."""
    # Get the path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, '..', 'data', 'all_stories_refined.json')

    with open(data_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    stories = []
    for story_id, story_data in raw_data.items():
        # Skip stories that don't have proper judgment
        if story_data.get('judgment') != 'Yes':
            continue
        
        # We do not have difficulty level for now for this story
        story = {
            "id": story_id,
            "title": story_data.get('new_title', f'Story {story_id}'),
            "description": story_data.get('new_surface', ''),  # Surface for the model
            "goal": "Explain what really happened in this story.",
            "ground_truth": story_data.get('new_bottom', ''),  # Bottom for evaluation
            "evaluation_criteria": story_data.get('evaluation', []),
        }
            
        stories.append(story)
        
    return stories


def get_random_story() -> Dict[str, Any]:
    """Get a random story from the collection."""
    stories = load_stories()
    return random.choice(stories)


def get_story_by_title(title: str) -> Dict[str, Any]:
    """Get a specific story by its title."""
    stories = load_stories()
    for story in stories:
        if story["title"].lower() == title.lower():
            return story
    raise ValueError(f"Story with title '{title}' not found")
