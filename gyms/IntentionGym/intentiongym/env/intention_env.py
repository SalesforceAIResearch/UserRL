import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List, Set
import random
import asyncio
import json

from .prompts import evaluate_question, evaluate_question_async
from .task_data import load_tasks, get_task_by_id
from ..config import IntentionGymConfig, get_default_config


class IntentionEnv(gym.Env):
    """
    Custom Gymnasium environment for intention guessing simulation.
    
    The agent attempts to clarify vague user tasks by asking targeted questions
    to uncover missing details. The environment responds with simulated user
    responses and tracks how well the questions cover the missing details.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: IntentionGymConfig = None):
        """
        Initialize the Intention Environment.
        
        Args:
            config: IntentionGymConfig instance with all configuration settings
        """
        super().__init__()
        
        # Use provided config or default
        self.config = config or get_default_config()
        self.config.validate()
        
        # Load tasks based on data configuration
        self._load_tasks()
        
        # Environment state
        self.current_task = None
        self.current_task_index = 0
        self.step_count = 0
        self.episode_complete = False
        self.question_history = []
        self.conversation_history = []  # Track (question, response) pairs
        self.remaining_missing_details = []  # List of missing details not yet covered
        self.covered_details = []  # List of covered missing details
        self.total_reward = 0.0
        
        # Set random seed if provided
        if self.config.seed is not None:
            self.seed(self.config.seed)
        
        # Action space: discrete actions representing different question types
        self.action_space = spaces.Discrete(3)
        
        # Observation space: dictionary containing task state and feedback
        self.observation_space = spaces.Dict({
            "task_category": spaces.Text(max_length=256),
            "task_description": spaces.Text(max_length=1024),
            "goal": spaces.Text(max_length=256),
            "feedback": spaces.Text(max_length=2048),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=np.int32),
            "episode_complete": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            "total_missing_details": spaces.Box(low=0, high=50, shape=(), dtype=np.int32),
            "remaining_missing_details": spaces.Box(low=0, high=50, shape=(), dtype=np.int32),
            "coverage_ratio": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "last_reward": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        })
    
    def _load_tasks(self):
        """Load tasks based on configuration."""
        all_tasks = load_tasks()
        
        if self.config.data_mode == "random":
            self.tasks = all_tasks
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
    
    def _calculate_coverage_ratio(self) -> float:
        """Calculate the ratio of covered missing details."""
        if not self.current_task or not self.current_task.get("missing_details"):
            return 0.0
        
        total_details = len(self.current_task["missing_details"])
        covered_count = len(self.covered_details)
        return covered_count / total_details if total_details > 0 else 1.0
    
    def _get_remaining_details_summary(self) -> str:
        """Get a summary of remaining uncovered details for observation."""
        if not self.remaining_missing_details:
            return "All missing details have been covered!"
        
        # Group by importance for better summary
        by_importance = {1: [], 2: [], 3: []}
        for detail in self.remaining_missing_details:
            importance = int(detail.get("importance", 1))
            by_importance[importance].append(detail["description"])
        
        summary_parts = []
        for importance in [3, 2, 1]:  # Highest importance first
            if by_importance[importance]:
                level_name = {3: "High", 2: "Medium", 1: "Low"}[importance]
                summary_parts.append(f"{level_name} priority: {', '.join(by_importance[importance])}")
        
        return f"Remaining details - {'; '.join(summary_parts)}"
    
    def reset(self, seed=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to start a new intention guessing session."""
        super().reset(seed=seed)
        
        # Get next task
        self.current_task = self._get_next_task()
        
        # Initialize missing details tracking
        self.remaining_missing_details = self.current_task.get("missing_details", []).copy()
        self.covered_details = []
        
        # Reset episode state
        self.step_count = 0
        self.episode_complete = False
        self.question_history = []
        self.conversation_history = []
        self.total_reward = 0.0
        
        # Create initial observation
        observation = {
            "task_category": self.current_task.get("category", ""),
            "task_description": self.current_task.get("task", ""),
            "goal": "Ask targeted questions to clarify the missing details in this vague task.",
            "feedback": f"Welcome to IntentionGym! Here's a vague task that needs clarification: '{self.current_task.get('task', '')}'. Ask me questions to understand what exactly I need help with!",
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "total_missing_details": len(self.current_task.get("missing_details", [])),
            "remaining_missing_details": len(self.remaining_missing_details),
            "coverage_ratio": self._calculate_coverage_ratio(),
            "last_reward": 0.0
        }
        
        info = {
            "task_id": self.current_task.get("id", ""),
            "missing_details_summary": self._get_remaining_details_summary(),
            "question_history": self.question_history.copy(),
            "conversation_history": self.conversation_history.copy(),
            "covered_details": self.covered_details.copy()
        }
        
        if self.config.verbose:
            print(f"🎯 New intention guessing session: {self.current_task.get('id', 'unknown')}")
            print(f"Task: {self.current_task.get('task', '')}")
            print(f"Missing details to uncover: {len(self.remaining_missing_details)}")
        
        return observation, info
    
    def step(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        question = str(action_input).strip()
        
        # Add question to history
        self.question_history.append(question)
        
        # Check for finish command
        if question.lower().startswith("[finish]"):
            self.episode_complete = True
            reward = 0.0  # No reward for finishing
            
            # Final observation
            observation = {
                "task_category": self.current_task.get("category", ""),
                "task_description": self.current_task.get("task", ""),
                "goal": "Ask targeted questions to clarify the missing details in this vague task.",
                "feedback": f"Session ended. You covered {len(self.covered_details)}/{len(self.current_task.get('missing_details', []))} missing details. Final coverage: {self._calculate_coverage_ratio():.2%}",
                "step_count": self.step_count,
                "episode_complete": self.episode_complete,
                "total_missing_details": len(self.current_task.get("missing_details", [])),
                "remaining_missing_details": len(self.remaining_missing_details),
                "coverage_ratio": self._calculate_coverage_ratio(),
                "last_reward": reward
            }
            
            info = {
                "task_id": self.current_task.get("id", ""),
                "missing_details_summary": self._get_remaining_details_summary(),
                "question_history": self.question_history.copy(),
                "conversation_history": self.conversation_history.copy(),
                "covered_details": self.covered_details.copy(),
                "final_coverage": self._calculate_coverage_ratio(),
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
        
        # Evaluate the question using LLM
        response, covered_detail_indices, reward_info = evaluate_question(
            question, self.current_task, model_config, 
            self.conversation_history, self.remaining_missing_details
        )
        
        # Process covered details and calculate reward
        newly_covered_details = []
        for idx in covered_detail_indices:
            if 0 <= idx < len(self.remaining_missing_details):
                detail = self.remaining_missing_details[idx]
                newly_covered_details.append(detail)
                self.covered_details.append(detail)
        
        # Remove covered details from remaining list
        self.remaining_missing_details = [
            detail for i, detail in enumerate(self.remaining_missing_details)
            if i not in covered_detail_indices
        ]
        
        # Calculate reward based on covered details
        reward = self._calculate_reward(newly_covered_details, reward_info)
        
        # Apply step penalty
        reward -= self.config.step_penalty
        
        # Normalize reward to [0, 1] if configured
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        
        # Update conversation history
        self.conversation_history.append((question, response))
        
        # Check termination conditions
        terminated = len(self.remaining_missing_details) == 0 or self.step_count >= self.config.max_steps
        truncated = self.step_count >= self.config.max_steps and len(self.remaining_missing_details) > 0
        
        if terminated or truncated:
            self.episode_complete = True
        
        # Create observation
        observation = {
            "task_category": self.current_task.get("category", ""),
            "task_description": self.current_task.get("task", ""),
            "goal": "Ask targeted questions to clarify the missing details in this vague task.",
            "feedback": response,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "total_missing_details": len(self.current_task.get("missing_details", [])),
            "remaining_missing_details": len(self.remaining_missing_details),
            "coverage_ratio": self._calculate_coverage_ratio(),
            "last_reward": reward
        }
        
        info = {
            "task_id": self.current_task.get("id", ""),
            "missing_details_summary": self._get_remaining_details_summary(),
            "question_history": self.question_history.copy(),
            "conversation_history": self.conversation_history.copy(),
            "covered_details": self.covered_details.copy(),
            "newly_covered_details": newly_covered_details,
            "reward_breakdown": reward_info,
            "total_reward": self.total_reward
        }
        
        if self.config.verbose:
            print(f"Q: {question}")
            print(f"A: {response}")
            print(f"Covered details: {len(newly_covered_details)}")
            print(f"Reward: {reward:.3f}")
            print(f"Remaining: {len(self.remaining_missing_details)}/{len(self.current_task.get('missing_details', []))}")
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, covered_details: List[Dict], reward_info: Dict[str, Any]) -> float:
        """Calculate reward based on covered missing details."""
        if not covered_details:
            return 0.0
        
        # Find the highest importance among covered details
        max_importance = max(int(detail.get("importance", 1)) for detail in covered_details)
        
        # Base reward based on highest importance
        importance_rewards = {3: 1.0, 2: 0.7, 1: 0.4}
        base_reward = importance_rewards.get(max_importance, 0.4)
        
        # Apply penalty for multiple details (we want focused questions)
        if len(covered_details) > 1:
            penalty = self.config.multi_detail_penalty * (len(covered_details) - 1)
            base_reward = max(0.0, base_reward - penalty)
        
        return base_reward * self.config.reward_scale
    
    async def step_async(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Async version of step method."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        question = str(action_input).strip()
        
        # Add question to history
        self.question_history.append(question)
        
        # Check for finish command
        if question.lower().startswith("[finish]"):
            self.episode_complete = True
            reward = 0.0
            
            observation = {
                "task_category": self.current_task.get("category", ""),
                "task_description": self.current_task.get("task", ""),
                "goal": "Ask targeted questions to clarify the missing details in this vague task.",
                "feedback": f"Session ended. You covered {len(self.covered_details)}/{len(self.current_task.get('missing_details', []))} missing details.",
                "step_count": self.step_count,
                "episode_complete": self.episode_complete,
                "total_missing_details": len(self.current_task.get("missing_details", [])),
                "remaining_missing_details": len(self.remaining_missing_details),
                "coverage_ratio": self._calculate_coverage_ratio(),
                "last_reward": reward
            }
            
            info = {
                "task_id": self.current_task.get("id", ""),
                "missing_details_summary": self._get_remaining_details_summary(),
                "question_history": self.question_history.copy(),
                "conversation_history": self.conversation_history.copy(),
                "covered_details": self.covered_details.copy(),
                "final_coverage": self._calculate_coverage_ratio(),
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
        
        # Evaluate the question using async LLM
        response, covered_detail_indices, reward_info = await evaluate_question_async(
            question, self.current_task, model_config, 
            self.conversation_history, self.remaining_missing_details
        )
        
        # Process covered details and calculate reward (same as sync version)
        newly_covered_details = []
        for idx in covered_detail_indices:
            if 0 <= idx < len(self.remaining_missing_details):
                detail = self.remaining_missing_details[idx]
                newly_covered_details.append(detail)
                self.covered_details.append(detail)
        
        self.remaining_missing_details = [
            detail for i, detail in enumerate(self.remaining_missing_details)
            if i not in covered_detail_indices
        ]
        
        reward = self._calculate_reward(newly_covered_details, reward_info)
        reward -= self.config.step_penalty
        
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        self.conversation_history.append((question, response))
        
        terminated = len(self.remaining_missing_details) == 0 or self.step_count >= self.config.max_steps
        truncated = self.step_count >= self.config.max_steps and len(self.remaining_missing_details) > 0
        
        if terminated or truncated:
            self.episode_complete = True
        
        observation = {
            "task_category": self.current_task.get("category", ""),
            "task_description": self.current_task.get("task", ""),
            "goal": "Ask targeted questions to clarify the missing details in this vague task.",
            "feedback": response,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "total_missing_details": len(self.current_task.get("missing_details", [])),
            "remaining_missing_details": len(self.remaining_missing_details),
            "coverage_ratio": self._calculate_coverage_ratio(),
            "last_reward": reward
        }
        
        info = {
            "task_id": self.current_task.get("id", ""),
            "missing_details_summary": self._get_remaining_details_summary(),
            "question_history": self.question_history.copy(),
            "conversation_history": self.conversation_history.copy(),
            "covered_details": self.covered_details.copy(),
            "newly_covered_details": newly_covered_details,
            "reward_breakdown": reward_info,
            "total_reward": self.total_reward
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        """Render the environment state."""
        if mode == "human":
            print(f"\n=== IntentionGym Step {self.step_count} ===")
            print(f"Task: {self.current_task.get('task', 'N/A')}")
            print(f"Coverage: {len(self.covered_details)}/{len(self.current_task.get('missing_details', []))} ({self._calculate_coverage_ratio():.1%})")
            print(f"Remaining details: {len(self.remaining_missing_details)}")
            if self.conversation_history:
                last_q, last_a = self.conversation_history[-1]
                print(f"Last Q: {last_q}")
                print(f"Last A: {last_a}")
            print(f"Total reward: {self.total_reward:.3f}")
    
    def close(self):
        """Clean up environment resources."""
        pass
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return seed 