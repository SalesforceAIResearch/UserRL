import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List
import random
import asyncio

from .prompts import evaluate_action, evaluate_action_async
from .story_data import load_stories, get_story_by_title
from ..config import TurtleGymConfig, get_default_config


class StoryEnv(gym.Env):
    """
    Custom Gymnasium environment for interactive story-based reinforcement learning.
    
    The agent interacts with narrative scenarios by taking actions or providing answers,
    receiving feedback from an LLM that evaluates the appropriateness of their choices.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: TurtleGymConfig = None):
        """
        Initialize the Story Environment.
        
        Args:
            config: TurtleGymConfig instance with all configuration settings
        """
        super().__init__()
        
        # Use provided config or default
        self.config = config or get_default_config()
        self.config.validate()
        
        # Load stories based on data configuration
        self._load_stories()
        
        # Environment state
        self.current_story = None
        self.current_story_index = 0
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.judge_history = []
        self.best_score = 0.0
        self.best_json_response = None
        self.total_reward = 0.0
        
        # Set random seed if provided
        if self.config.seed is not None:
            self.seed(self.config.seed)
        
        # Action space: discrete actions representing different action types
        self.action_space = spaces.Discrete(3)
        
        # Observation space: dictionary containing story state and feedback
        self.observation_space = spaces.Dict({
            "story_title": spaces.Text(max_length=1024),
            "story_text": spaces.Text(max_length=1024),
            "goal": spaces.Text(max_length=256),
            "feedback": spaces.Text(max_length=1024),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=np.int32),
            "episode_complete": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            "current_score": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "best_score": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        })
    
    def _load_stories(self):
        """Load stories based on configuration."""
        all_stories = load_stories()
        
        if self.config.data_mode == "random":
            self.stories = all_stories
        elif self.config.data_mode == "single":
            # Find the specific story
            story = get_story_by_title(self.config.data_source)
            self.stories = [story]
        elif self.config.data_mode == "list":
            # Load multiple specific stories
            self.stories = []
            for title in self.config.data_source:
                try:
                    story = get_story_by_title(title)
                    self.stories.append(story)
                except ValueError:
                    if self.config.verbose:
                        print(f"Warning: Story '{title}' not found, skipping")
            
            if not self.stories:
                raise ValueError("No valid stories found in data_source list")
        
        if self.config.verbose:
            print(f"Loaded {len(self.stories)} stories in {self.config.data_mode} mode")
    
    def _get_next_story(self):
        """Get the next story based on data mode."""
        if self.config.data_mode == "random":
            return random.choice(self.stories)
        elif self.config.data_mode == "single":
            return self.stories[0]
        elif self.config.data_mode == "list":
            # Cycle through the list
            story = self.stories[self.current_story_index]
            self.current_story_index = (self.current_story_index + 1) % len(self.stories)
            return story
    
    def reset(self, seed=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        # Get next story
        self.current_story = self._get_next_story()
        
        # Reset episode state
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.judge_history = []
        self.best_score = 0.0
        self.best_json_response = None
        self.total_reward = 0.0
        
        # Create initial observation
        observation = {
            "story_title": self.current_story["title"],
            "story_text": self.current_story["description"],
            "goal": self.current_story["goal"],
            "feedback": "Welcome to the Turtle Soup Gym. Choose your action.",
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "current_score": 0.0,
            "best_score": 0.0
        }
        
        info = {
            "total_criteria": len(self.current_story.get("evaluation_criteria", [])),
            "ground_truth": self.current_story.get("ground_truth", ""),
            "action_history": self.action_history.copy(),
            "judge_history": self.judge_history.copy()
        }
        
        if self.config.verbose:
            print(f"🎭 New episode: {self.current_story['title']}")
        
        return observation, info
    
    def step(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        # if self.step_count != 0:
        #     print("~~~~~~~~ This is not the first step !!! I am currently in the evironment step function ~~~~~~~~")

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
        
        # Evaluate the action using LLM
        feedback, score, json_response, judge_success = evaluate_action(
            action_str, self.current_story, model_config
        )
        
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
        
        # Calculate reward
        base_reward = base_reward * self.config.reward_scale
        step_penalty = self.config.step_penalty * self.step_count
        
        reward = base_reward - step_penalty
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        self.judge_history.append(json_response)
        
        # Check termination conditions
        terminated = False
        if action_str.startswith("[answer]"):
            terminated = score >= self.config.success_threshold
        elif action_str.startswith("[finish]"):
            terminated = True
        
        truncated = self.step_count >= self.config.max_steps
        
        if terminated or truncated:
            self.episode_complete = True
        
        # Create observation
        observation = {
            "story_title": self.current_story["title"],
            "story_text": self.current_story["description"],
            "goal": self.current_story["goal"],
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
            "total_criteria": len(self.current_story.get("evaluation_criteria", [])),
            "ground_truth": self.current_story.get("ground_truth", ""),
            "action_history": self.action_history.copy(),
            "judge_history": self.judge_history.copy(),
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
        
        # Evaluate the action using LLM (async version)
        feedback, score, json_response, judge_success = await evaluate_action_async(
            action_str, self.current_story, model_config
        )
        
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
        
        # Calculate reward
        base_reward = base_reward * self.config.reward_scale
        step_penalty = self.config.step_penalty * self.step_count
        
        reward = base_reward - step_penalty
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        self.judge_history.append(json_response)
        
        # Check termination conditions
        terminated = False
        if action_str.startswith("[answer]"):
            terminated = score >= self.config.success_threshold
        elif action_str.startswith("[finish]"):
            terminated = True
        
        truncated = self.step_count >= self.config.max_steps
        
        if terminated or truncated:
            self.episode_complete = True
        
        # Create observation
        observation = {
            "story_title": self.current_story["title"],
            "story_text": self.current_story["description"],
            "goal": self.current_story["goal"],
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
            "total_criteria": len(self.current_story.get("evaluation_criteria", [])),
            "ground_truth": self.current_story.get("ground_truth", ""),
            "action_history": self.action_history.copy(),
            "judge_history": self.judge_history.copy(),
            "total_reward": self.total_reward
        }
        
        if terminated and self.config.verbose:
            print(f"🎉 Episode completed successfully! Score: {self.best_score:.2f}")
        elif truncated and self.config.verbose:
            print(f"⏰ Episode ended due to max steps. Best score: {self.best_score:.2f}")
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        """Render the current state of the environment."""
        if not self.config.verbose and mode == "human":
            return
            
        if not self.current_story:
            print("No story loaded. Call reset() first.")
            return
        
        print("\n" + "="*50)
        print(f"STORY: {self.current_story['title']}")
        print("="*50)
        print(f"Description: {self.current_story['description']}")
        print(f"Goal: {self.current_story['goal']}")
        print(f"Steps taken: {self.step_count}/{self.config.max_steps}")
        print(f"Best Score: {self.best_score:.2f}")
        print(f"Total Reward: {self.total_reward:.2f}")
        
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