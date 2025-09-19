"""
TravelEnv: A Gymnasium environment for travel planning preference elicitation.

This module provides a Gymnasium environment where agents interact with simulated
users to elicit travel preferences and provide recommendations.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List, Set
import random
import asyncio
import json

from .prompts import evaluate_action
from .prompt_async import async_evaluate_action
from .task_data import load_tasks, get_task_by_id
from ..config import TravelGymConfig, get_default_config


class TravelEnv(gym.Env):
    """
    TravelGym Environment for travel planning preference elicitation.
    
    This environment simulates a conversation between an agent and a user
    where the agent needs to elicit travel preferences and provide recommendations.
    """
    
    def __init__(self, config: TravelGymConfig = None):
        """Initialize the TravelGym environment."""
        super().__init__()
        
        # Set configuration
        self.config = config if config is not None else get_default_config()
        self.config.validate()
        
        # Set random seed if provided
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        # Load available tasks based on configuration
        self.current_task_index = 0
        self._load_tasks()
        
        # Define action and observation spaces
        # Action space is text input (we'll use Box for simplicity)
        self.action_space = spaces.Text(max_length=1000)
        
        # Observation space is a dictionary
        self.observation_space = spaces.Dict({
            "task_description": spaces.Text(max_length=5000),
            "goal": spaces.Text(max_length=500),
            "feedback": spaces.Text(max_length=5000),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=int),
            "episode_complete": spaces.Discrete(2),
            "total_preferences": spaces.Box(low=0, high=100, shape=(), dtype=int),
            "remaining_preferences": spaces.Box(low=0, high=100, shape=(), dtype=int),
            "elicitation_ratio": spaces.Box(low=0.0, high=1.0, shape=(), dtype=float),
            "last_reward": spaces.Box(low=-10.0, high=10.0, shape=(), dtype=float)
        })
        
        # Initialize state variables
        self.reset()
    
    def _load_tasks(self):
        """Load tasks based on configuration."""
        if self.config.data_mode == "random":
            self.tasks = load_tasks(self.config)
        elif self.config.data_mode == "single":
            # Find the specific task
            task = get_task_by_id(self.config.data_source)
            self.tasks = [task]
        elif self.config.data_mode == "list":
            # Load multiple specific tasks
            self.tasks = []
            for task_id in self.config.data_source:
                try:
                    task = get_task_by_id(task_id)
                    self.tasks.append(task)
                except ValueError:
                    if self.config.verbose:
                        print(f"Warning: Task '{task_id}' not found, skipping")
            
            if not self.tasks:
                raise ValueError("No valid tasks found in data_source list")
        
        if self.config.verbose:
            print(f"Loaded {len(self.tasks)} tasks in {self.config.data_mode} mode")
    
    def _get_next_task(self):
        """Get the next task based on data mode."""
        if self.config.data_mode == "random":
            return random.choice(self.tasks)
        elif self.config.data_mode == "single":
            return self.tasks[0]
        elif self.config.data_mode == "list":
            # Cycle through the list
            task = self.tasks[self.current_task_index]
            self.current_task_index = (self.current_task_index + 1) % len(self.tasks)
            return task
    
    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Returns:
            Tuple of (observation, info)
        """
        # Set seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset episode state
        self.episode_complete = False
        self.step_count = 0
        self.total_reward = 0.0
        
        # Reset history tracking
        self.action_history = []
        self.conversation_history = []
        self.elicited_preferences = []
        
        # Select task based on configuration
        self.current_task = self._get_next_task()
        
        # Process task data and initialize preferences
        self._initialize_preferences()
        
        # Initialize state tracking for search and choices
        self._initialize_state_list()
        
        # Create initial observation
        observation = {
            "task_description": self.current_task.get("scenario", ""),
            "goal": "Elicit travel preferences and provide appropriate recommendations.",
            "feedback": self.current_task.get("initial_desc", "Let's start the conversation!"),
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "total_preferences": len(self.remaining_preferences) + len(self.elicited_preferences),
            "remaining_preferences": len(self.remaining_preferences),
            "elicitation_ratio": self._calculate_elicitation_ratio(),
            "last_reward": 0.0
        }
        
        # Create info dictionary
        info = {
            "task_id": self.current_task.get("id", ""),
            "preferences_summary": self._get_remaining_preferences_summary(),
            "action_history": self.action_history.copy(),
            "conversation_history": self.conversation_history.copy(),
            "elicited_preferences": self.elicited_preferences.copy(),
            "total_reward": self.total_reward
        }
        
        if self.config.verbose:
            print(f"🎯 New Episode Started")
            print(f"Task ID: {info['task_id']}")
            print(f"Total Preferences: {observation['total_preferences']}")
            print(f"Scenario: {observation['task_description'][:100]}...")
        
        return observation, info
    
    def _initialize_preferences(self):
        """Initialize preferences from the current task."""
        self.remaining_preferences = []
        
        # Extract preferences from task data
        preferences_data = self.current_task.get("preferences", {})
        dimensions = self.current_task.get("dimensions", [])
        
        pref_id_counter = 1
        for dimension in dimensions:
            if dimension in preferences_data:
                dim_data = preferences_data[dimension]
                if "preferences" in dim_data:
                    for pref_list in dim_data["preferences"]:
                        if isinstance(pref_list, list) and len(pref_list) >= 4:
                            # Format: [aspect, subcategory, preference_text, implicit_elicitation, ...]
                            preference = {
                                "id": f"P{pref_id_counter}",
                                "aspect": pref_list[0],
                                "subcategory": pref_list[1], 
                                "preference": pref_list[2],
                                "implicit_elicitation": pref_list[3],
                                "dimension": dimension,
                                "elicited": False
                            }
                            self.remaining_preferences.append(preference)
                            pref_id_counter += 1
        
        # If no preferences found, create empty list
        if not self.remaining_preferences:
            self.remaining_preferences = []
    
    def _initialize_state_list(self):
        """Initialize state tracking for search and answer evaluation."""
        self.state_list = {
            "search_times": 0,
            "nonpreference_times": 0,
            "search_arguments": [],
            "remaining_best_options": [],
            "remaining_correct_options": [],
            "choice_initials": [],
            "active_elicited_preferences": 0,
            "passive_elicited_preferences": 0,
        }
        
        # Initialize search arguments from task dimensions
        dimensions = self.current_task.get("dimensions", [])
        self.state_list["search_arguments"] = dimensions.copy()
        
        # Initialize best and correct options from task data
        preferences_data = self.current_task.get("preferences", {})
        for dimension in dimensions:
            if dimension in preferences_data:
                dim_data = preferences_data[dimension]
                # Add correct options
                if "correct_ids" in dim_data:
                    self.state_list["remaining_correct_options"].extend(dim_data["correct_ids"])
                # Add best options using best_id
                if "best_id" in dim_data:
                    self.state_list["remaining_best_options"].append(dim_data["best_id"])
    
    def _calculate_elicitation_ratio(self) -> float:
        """Calculate the ratio of elicited preferences."""
        total_prefs = len(self.remaining_preferences) + len(self.elicited_preferences)
        if total_prefs == 0:
            return 0.0
        return len(self.elicited_preferences) / total_prefs
    
    def _get_remaining_preferences_summary(self) -> List[str]:
        """Get a summary of remaining preferences."""
        return [
            f"{pref['aspect']}-{pref['subcategory']}: {pref['preference'][:50]}..."
            for pref in self.remaining_preferences
        ]
    
    def close(self):
        """Clean up the environment."""
        pass

    def step(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        action = str(action_input).strip()
        
        # Add action to history
        self.action_history.append(action)
        
        # Check for finish command
        if action.lower().startswith("[finish]"):
            self.episode_complete = True
            reward = 0.0  # No reward for finishing
            
            # Final observation
            observation = {
                "task_description": self.current_task.get("scenario", ""),
                "goal": "Elicit travel preferences and provide appropriate recommendations.",
                "feedback": f"Session ended. You elicited {len(self.elicited_preferences)}/{len(self.remaining_preferences) + len(self.elicited_preferences)} preferences. Final elicitation ratio: {self._calculate_elicitation_ratio():.2%}",
                "step_count": self.step_count,
                "episode_complete": self.episode_complete,
                "total_preferences": len(self.remaining_preferences) + len(self.elicited_preferences),
                "remaining_preferences": len(self.remaining_preferences),
                "elicitation_ratio": self._calculate_elicitation_ratio(),
                "last_reward": reward
            }
            
            info = {
                "task_id": self.current_task.get("id", ""),
                "preferences_summary": self._get_remaining_preferences_summary(),
                "action_history": self.action_history.copy(),
                "conversation_history": self.conversation_history.copy(),
                "elicited_preferences": self.elicited_preferences.copy(),
                "final_elicitation_ratio": self._calculate_elicitation_ratio(),
                "total_reward": self.total_reward
            }
            
            return observation, reward, True, False, info
        
        # Create model config dict for evaluation
        model_config = {
            "api_key": self.config.api_key,
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        # state_config
        state_config = {
            "search_correct_reward": self.config.search_correct_reward,
            "preference_correct_reward": self.config.preference_correct_reward,
            "choice_correct_reward": self.config.choice_correct_reward,
            "choice_best_reward": self.config.choice_best_reward,
            "search_failure_interval": self.config.search_failure_interval,
            "elicitation_interval": self.config.elicitation_interval,
            "wrong_choice_penalty": self.config.wrong_choice_penalty,
            "one_choice_per_aspect": self.config.one_choice_per_aspect,
        }
        
        # Evaluate the action using LLM
        response, elicited_preference_ids, reward = evaluate_action(
            action, self.current_task, state_config, model_config, 
            self.conversation_history, self.remaining_preferences, self.state_list
        )
        
        # Process elicited preferences and calculate reward
        newly_elicited_preferences = []
        for pref_id in elicited_preference_ids:
            for pref in self.remaining_preferences:
                if pref['id'] == pref_id:
                    pref['elicited'] = True
                    newly_elicited_preferences.append(pref)
                    self.elicited_preferences.append(pref)
                    break
        
        # Remove elicited preferences from remaining list
        self.remaining_preferences = [
            pref for pref in self.remaining_preferences
            if pref['id'] not in elicited_preference_ids
        ]
        
        # Calculate reward based on elicited preferences and action quality
        reward = reward * self.config.reward_scale - self.config.step_penalty
        # Normalize reward to [0, 1] if configured
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        
        # Check termination conditions
        terminated = self.state_list["remaining_best_options"] == []
        truncated = self.step_count >= self.config.max_steps
        # Check if one choice per aspect is enforced to terminate
        if self.config.one_choice_per_aspect and len(self.current_task["dimensions"]) == len(self.state_list["choice_initials"]):
            terminated = True
        
        if terminated or truncated:
            self.episode_complete = True
        
        # Create observation
        observation = {
            "task_description": self.current_task.get("scenario", ""),
            "goal": "Elicit travel preferences and provide appropriate recommendations.",
            "feedback": response,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "total_preferences": len(self.remaining_preferences) + len(self.elicited_preferences),
            "remaining_preferences": len(self.remaining_preferences),
            "elicitation_ratio": self._calculate_elicitation_ratio(),
            "active_elicited_preferences": self.state_list["active_elicited_preferences"],
            "passive_elicited_preferences": self.state_list["passive_elicited_preferences"],
            "last_reward": reward
        }
        
        info = {
            "task_id": self.current_task.get("id", ""),
            "preferences_summary": self._get_remaining_preferences_summary(),
            "action_history": self.action_history.copy(),
            "conversation_history": self.conversation_history.copy(),
            "elicited_preferences": self.elicited_preferences.copy(),
            "newly_elicited_preferences": newly_elicited_preferences,
            "total_reward": self.total_reward
        }
        
        if self.config.verbose:
            print(f"A: {action}")
            print(f"U: {response}")
            print(f"Elicited preferences: {len(newly_elicited_preferences)}")
            print(f"Reward: {reward:.3f}")
            print(f"Remaining: {len(self.remaining_preferences)}/{len(self.remaining_preferences) + len(self.elicited_preferences)}")
        
        return observation, reward, terminated, truncated, info
        
    async def step_async(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment asynchronously."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        action = str(action_input).strip()
        
        # Add action to history
        self.action_history.append(action)
        
        # Check for finish command
        if action.lower().startswith("[finish]"):
            self.episode_complete = True
            reward = 0.0  # No reward for finishing
            
            # Final observation
            observation = {
                "task_description": self.current_task.get("scenario", ""),
                "goal": "Elicit travel preferences and provide appropriate recommendations.",
                "feedback": f"Session ended. You elicited {len(self.elicited_preferences)}/{len(self.remaining_preferences) + len(self.elicited_preferences)} preferences. Final elicitation ratio: {self._calculate_elicitation_ratio():.2%}",
                "step_count": self.step_count,
                "episode_complete": self.episode_complete,
                "total_preferences": len(self.remaining_preferences) + len(self.elicited_preferences),
                "remaining_preferences": len(self.remaining_preferences),
                "elicitation_ratio": self._calculate_elicitation_ratio(),
                "last_reward": reward
            }
            
            info = {
                "task_id": self.current_task.get("id", ""),
                "preferences_summary": self._get_remaining_preferences_summary(),
                "action_history": self.action_history.copy(),
                "conversation_history": self.conversation_history.copy(),
                "elicited_preferences": self.elicited_preferences.copy(),
                "final_elicitation_ratio": self._calculate_elicitation_ratio(),
                "total_reward": self.total_reward
            }
            
            return observation, reward, True, False, info
        
        # Create model config dict for evaluation
        model_config = {
            "api_key": self.config.api_key,
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }

        # state_config
        state_config = {
            "search_correct_reward": self.config.search_correct_reward,
            "preference_correct_reward": self.config.preference_correct_reward,
            "choice_correct_reward": self.config.choice_correct_reward,
            "choice_best_reward": self.config.choice_best_reward,
            "search_failure_interval": self.config.search_failure_interval,
            "elicitation_interval": self.config.elicitation_interval,
            "wrong_choice_penalty": self.config.wrong_choice_penalty,
            "one_choice_per_aspect": self.config.one_choice_per_aspect,
        }
        
        # Evaluate the action using async LLM
        response, elicited_preference_ids, reward = await async_evaluate_action(
            action, self.current_task, state_config, model_config, 
            self.conversation_history, self.remaining_preferences, self.state_list
        )
        
        # Process elicited preferences and calculate reward
        newly_elicited_preferences = []
        for pref_id in elicited_preference_ids:
            for pref in self.remaining_preferences:
                if pref['id'] == pref_id:
                    pref['elicited'] = True
                    newly_elicited_preferences.append(pref)
                    self.elicited_preferences.append(pref)
                    break
        
        # Remove elicited preferences from remaining list
        self.remaining_preferences = [
            pref for pref in self.remaining_preferences
            if pref['id'] not in elicited_preference_ids
        ]
        
        # Calculate reward based on elicited preferences and action quality
        reward = reward * self.config.reward_scale - self.config.step_penalty
        # Normalize reward to [0, 1] if configured
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        
        # Check termination conditions
        terminated = self.state_list["remaining_best_options"] == []
        truncated = self.step_count >= self.config.max_steps
        
        if terminated or truncated:
            self.episode_complete = True
        
        # Create observation
        observation = {
            "task_description": self.current_task.get("scenario", ""),
            "goal": "Elicit travel preferences and provide appropriate recommendations.",
            "feedback": response,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "total_preferences": len(self.remaining_preferences) + len(self.elicited_preferences),
            "remaining_preferences": len(self.remaining_preferences),
            "elicitation_ratio": self._calculate_elicitation_ratio(),
            "active_elicited_preferences": self.state_list["active_elicited_preferences"],
            "passive_elicited_preferences": self.state_list["passive_elicited_preferences"],
            "last_reward": reward
        }
        
        info = {
            "task_id": self.current_task.get("id", ""),
            "preferences_summary": self._get_remaining_preferences_summary(),
            "action_history": self.action_history.copy(),
            "conversation_history": self.conversation_history.copy(),
            "elicited_preferences": self.elicited_preferences.copy(),
            "newly_elicited_preferences": newly_elicited_preferences,
            "total_reward": self.total_reward
        }
        
        if self.config.verbose:
            print(f"A: {action}")
            print(f"U: {response}")
            print(f"Elicited preferences: {len(newly_elicited_preferences)}")
            print(f"Reward: {reward:.3f}")
            print(f"Remaining: {len(self.remaining_preferences)}/{len(self.remaining_preferences) + len(self.elicited_preferences)}")
        
        return observation, reward, terminated, truncated, info
