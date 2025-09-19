import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List
import random
import asyncio

from .prompts import evaluate_action, evaluate_action_async
from .entity_data import load_entities, get_entity_by_title
from ..config import TelepathyGymConfig, get_default_config


class TelepathyEnv(gym.Env):
    """
    Custom Gymnasium environment for mind reading games.
    
    The agent asks yes/no questions to guess what entity the AI is thinking of,
    receiving feedback from an LLM that responds based on the target entity.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: TelepathyGymConfig = None):
        """
        Initialize the Telepathy Environment.
        
        Args:
            config: TelepathyGymConfig instance with all configuration settings
        """
        super().__init__()
        
        # Use provided config or default
        self.config = config or get_default_config()
        self.config.validate()
        
        # Load entities based on data configuration
        self._load_entities()
        
        # Environment state
        self.current_entity = None
        self.current_entity_index = 0
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.response_history = []
        self.clue_history = []  # Track questions and responses for mind reading
        self.best_score = 0.0
        self.best_json_response = None
        self.total_reward = 0.0
        
        # Set random seed if provided
        if self.config.seed is not None:
            self.seed(self.config.seed)
        
        # Action space: discrete actions representing different action types
        self.action_space = spaces.Discrete(3)
        
        # Observation space: dictionary containing session state and feedback
        self.observation_space = spaces.Dict({
            "entity_title": spaces.Text(max_length=1024),
            "description": spaces.Text(max_length=1024),
            "goal": spaces.Text(max_length=256),
            "feedback": spaces.Text(max_length=1024),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=np.int32),
            "episode_complete": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            "current_score": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "best_score": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        })
    
    def _load_entities(self):
        """Load entities based on configuration."""
        all_entities = load_entities()
        
        if self.config.data_mode == "random":
            self.entities = all_entities
        elif self.config.data_mode == "single":
            # Find the specific entity
            entity = get_entity_by_title(self.config.data_source)
            self.entities = [entity]
        elif self.config.data_mode == "list":
            # Load multiple specific entities
            self.entities = []
            for title in self.config.data_source:
                try:
                    entity = get_entity_by_title(title)
                    self.entities.append(entity)
                except ValueError:
                    if self.config.verbose:
                        print(f"Warning: Entity '{title}' not found, skipping")
            
            if not self.entities:
                raise ValueError("No valid entities found in data_source list")
        
        if self.config.verbose:
            print(f"Loaded {len(self.entities)} entities in {self.config.data_mode} mode")
    
    def _get_next_entity(self):
        """Get the next entity based on data mode."""
        if self.config.data_mode == "random":
            return random.choice(self.entities)
        elif self.config.data_mode == "single":
            return self.entities[0]
        elif self.config.data_mode == "list":
            # Cycle through the list
            entity = self.entities[self.current_entity_index]
            self.current_entity_index = (self.current_entity_index + 1) % len(self.entities)
            return entity
    
    def reset(self, seed=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to start a new telepathy session."""
        super().reset(seed=seed)
        
        # Get next entity
        self.current_entity = self._get_next_entity()
        
        # Reset episode state
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.response_history = []
        self.clue_history = []
        self.best_score = 0.0
        self.best_json_response = None
        self.total_reward = 0.0
        self.guessed = False

        # Create initial observation
        observation = {
            "entity_title": self.current_entity["title"],
            "description": self.current_entity["description"],
            "goal": self.current_entity["goal"],
            "feedback": "Welcome to TelepathyGym! I'm thinking of something... Ask me yes/no questions to figure out what it is!",
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "current_score": 0.0,
            "best_score": 0.0
        }
        
        info = {
            "target_entity": self.current_entity.get("target_entity", ""),
            "category": self.current_entity.get("category", ""),
            "action_history": self.action_history.copy(),
            "response_history": self.response_history.copy(),
            "clue_history": self.clue_history.copy()
        }
        
        if self.config.verbose:
            print(f"🧠 New telepathy session: {self.current_entity['title']}")
        
        return observation, info
    
    def step(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        action_str = str(action_input)
        
        # Add action to history
        self.action_history.append(action_str)
        
        # Create model config dict for evaluation
        model_config = {
            "api_key": self.config.api_key,
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }
        
        # Evaluate the action using LLM with clue history
        feedback, score, json_response, judge_success = evaluate_action(
            action_str, self.current_entity, model_config, self.clue_history
        )

        self.guessed = (score == 1.0)
        
        # Update clue history for questions
        if action_str.startswith("[action]") or action_str.startswith("[answer]"):
            question = action_str[8:].strip()
            self.clue_history.append((question, feedback))
        
        base_reward = 0.0
        # Update best score if this is an answer
        if action_str.startswith("[answer]"):
            # calculate base reward as the delta between current score and best score
            base_reward = score - self.best_score
            if base_reward < 0:
                base_reward = 0
            if score > self.best_score:
                self.best_score = score
                self.best_json_response = json_response
        
        # Calculate reward - simplified for mind reading
        base_reward = base_reward * self.config.reward_scale
        step_penalty = self.config.step_penalty * self.step_count
        
        reward = base_reward - step_penalty
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        self.response_history.append(json_response)
        
        # Check termination conditions
        terminated = False
        if action_str.startswith("[answer]") and self.guessed:
            terminated = True
        elif action_str.startswith("[finish]"):
            terminated = True
        
        truncated = self.step_count >= self.config.max_steps
        
        if terminated or truncated:
            self.episode_complete = True
        
        # Create observation
        observation = {
            "entity_title": self.current_entity["title"],
            "description": self.current_entity["description"],
            "goal": self.current_entity["goal"],
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "current_score": score,
            "best_score": self.best_score
        }
        
        # Create info dictionary
        info = {
            "raw_action": action_str,
            "judge_success": judge_success,
            "current_score": score,
            "best_score": self.best_score,
            "best_json_response": self.best_json_response,
            "target_entity": self.current_entity.get("target_entity", ""),
            "category": self.current_entity.get("category", ""),
            "action_history": self.action_history.copy(),
            "response_history": self.response_history.copy(),
            "clue_history": self.clue_history.copy(),
            "total_reward": self.total_reward
        }
        
        if terminated and self.config.verbose:
            print(f"🎉 Episode completed successfully! Score: {self.best_score:.2f}")
        elif truncated and self.config.verbose:
            print(f"⏰ Episode ended due to max steps. Best score: {self.best_score:.2f}")
        
        return observation, reward, terminated, truncated, info

    async def step_async(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment (async version)."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        action_str = str(action_input)
        
        # Add action to history
        self.action_history.append(action_str)
        
        # Create model config dict for evaluation
        model_config = {
            "api_key": self.config.api_key,
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }
        
        # Evaluate the action using LLM with clue history (async version)
        feedback, score, json_response, judge_success = await evaluate_action_async(
            action_str, self.current_entity, model_config, self.clue_history
        )
        
        # Update clue history for questions
        if action_str.startswith("[action]") or action_str.startswith("[answer]"):
            question = action_str[8:].strip()
            self.clue_history.append((question, feedback))
        
        self.guessed = (score == 1.0)

        base_reward = 0.0
        # Update best score if this is an answer
        if action_str.startswith("[answer]"):
            # calculate base reward as the delta between current score and best score
            base_reward = score - self.best_score
            if base_reward < 0:
                base_reward = 0
            if score > self.best_score:
                self.best_score = score
                self.best_json_response = json_response
        
        # Calculate reward - simplified for mind reading
        base_reward = base_reward * self.config.reward_scale
        step_penalty = self.config.step_penalty * self.step_count
        
        reward = base_reward - step_penalty
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        self.response_history.append(json_response)
        
        # Check termination conditions
        terminated = False
        if action_str.startswith("[answer]") and self.guessed:
            terminated = True
        elif action_str.startswith("[finish]"):
            terminated = True
        
        truncated = self.step_count >= self.config.max_steps
        
        if terminated or truncated:
            self.episode_complete = True
        
        # Create observation
        observation = {
            "entity_title": self.current_entity["title"],
            "description": self.current_entity["description"],
            "goal": self.current_entity["goal"],
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "current_score": score,
            "best_score": self.best_score
        }
        
        # Create info dictionary
        info = {
            "raw_action": action_str,
            "judge_success": judge_success,
            "current_score": score,
            "best_score": self.best_score,
            "best_json_response": self.best_json_response,
            "target_entity": self.current_entity.get("target_entity", ""),
            "category": self.current_entity.get("category", ""),
            "action_history": self.action_history.copy(),
            "response_history": self.response_history.copy(),
            "clue_history": self.clue_history.copy(),
            "total_reward": self.total_reward
        }
        
        if terminated and self.config.verbose:
            print(f"🎉 Episode completed successfully! Score: {self.best_score:.2f}")
        elif truncated and self.config.verbose:
            print(f"⏰ Episode ended due to max steps. Best score: {self.best_score:.2f}")
        
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the current state of the environment."""
        if not self.current_entity:
            print("No entity loaded. Call reset() first.")
            return
        
        print("\n" + "="*50)
        print(f"TELEPATHY SESSION: {self.current_entity['title']}")
        print("="*50)
        print(f"Category: {self.current_entity.get('category', 'Unknown')}")
        print(f"Goal: {self.current_entity['goal']}")
        print(f"Steps taken: {self.step_count}/{self.config.max_steps}")
        print(f"Best Score: {self.best_score:.2f}")
        
        # Show clue history
        if self.clue_history:
            print(f"\n🧠 Clues so far:")
            for i, (question, response) in enumerate(self.clue_history, 1):
                print(f"  Q{i}: {question}")
                print(f"  A{i}: {response}")
        
        if self.action_history:
            print(f"\nLast action: {self.action_history[-1]}")
        
        print("="*50)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
