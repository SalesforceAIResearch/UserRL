import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import numpy as np
import random
from ..config import TemplateGymConfig, get_default_config

class TemplateEnv(gym.Env):
    """Template environment for [your domain description]."""
    
    def __init__(self, config: Optional[TemplateGymConfig] = None):
        super().__init__()
        
        self.config = config or get_default_config()
        self.config.validate()
        
        # Define action and observation spaces
        self.action_space = spaces.Text(max_length=1000)
        self.observation_space = spaces.Dict({
            "description": spaces.Text(max_length=5000),
            "goal": spaces.Text(max_length=1000),
            "feedback": spaces.Text(max_length=2000),
            "step_count": spaces.Discrete(1000),
            "current_score": spaces.Box(0.0, 1.0, dtype=np.float32),
        })
        
        # Environment state
        self.step_count = 0
        self.episode_complete = False
        self.current_data = None
        self.total_reward = 0.0
        self.best_score = 0.0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_complete = False
        self.total_reward = 0.0
        self.best_score = 0.0
        
        # Load new data
        self.current_data = self._load_data()
        
        observation = {
            "description": self.current_data.get("description", ""),
            "goal": self.current_data.get("goal", "Complete the task successfully."),
            "feedback": "Environment ready.",
            "step_count": self.step_count,
            "current_score": 0.0,
        }
        
        info = {"data_id": self.current_data.get("id", "")}
        return observation, info
    
    def step(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        action_str = str(action_input).strip()
        
        # Process action
        feedback, reward, terminated, truncated = self._process_action(action_str)
        
        if terminated or truncated:
            self.episode_complete = True
        
        self.total_reward += reward
        
        observation = {
            "description": self.current_data.get("description", ""),
            "goal": self.current_data.get("goal", "Complete the task successfully."),
            "feedback": feedback,
            "step_count": self.step_count,
            "current_score": self.best_score,
        }
        
        info = {"total_reward": self.total_reward}
        return observation, reward, terminated, truncated, info
    
    async def step_async(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment asynchronously."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        action_str = str(action_input).strip()
        
        # Process action asynchronously
        feedback, reward, terminated, truncated = await self._process_action_async(action_str)
        
        if terminated or truncated:
            self.episode_complete = True
        
        self.total_reward += reward
        
        observation = {
            "description": self.current_data.get("description", ""),
            "goal": self.current_data.get("goal", "Complete the task successfully."),
            "feedback": feedback,
            "step_count": self.step_count,
            "current_score": self.best_score,
        }
        
        info = {"total_reward": self.total_reward}
        return observation, reward, terminated, truncated, info
    
    def _process_action(self, action_str: str) -> Tuple[str, float, bool, bool]:
        """Process the agent's action."""
        if action_str.startswith("[finish]"):
            return "Episode ended.", 0.0, True, False
        
        elif action_str.startswith("[action]"):
            question = action_str[8:].strip()
            feedback, reward = self._handle_action(question)
            return feedback, reward, False, False
        
        elif action_str.startswith("[answer]"):
            answer = action_str[8:].strip()
            feedback, reward, success = self._handle_answer(answer)
            return feedback, reward, success, False
        
        else:
            return "Invalid action format.", 0.0, False, False
    
    async def _process_action_async(self, action_str: str) -> Tuple[str, float, bool, bool]:
        """Process the agent's action asynchronously."""
        if action_str.startswith("[finish]"):
            return "Episode ended.", 0.0, True, False
        
        elif action_str.startswith("[action]"):
            question = action_str[8:].strip()
            feedback, reward = await self._handle_action_async(question)
            return feedback, reward, False, False
        
        elif action_str.startswith("[answer]"):
            answer = action_str[8:].strip()
            feedback, reward, success = await self._handle_answer_async(answer)
            return feedback, reward, success, False
        
        else:
            return "Invalid action format.", 0.0, False, False
    
    def _handle_action(self, question: str) -> Tuple[str, float]:
        """Handle action commands."""
        # TODO: Implement your domain-specific action handling
        feedback = f"Received: '{question}'"
        reward = 0.0
        return feedback, reward
    
    async def _handle_action_async(self, question: str) -> Tuple[str, float]:
        """Handle action commands asynchronously."""
        # TODO: Implement your domain-specific async action handling
        feedback = f"Received: '{question}'"
        reward = 0.0
        return feedback, reward
    
    def _handle_answer(self, answer: str) -> Tuple[str, float, bool]:
        """Handle answer commands."""
        # TODO: Implement your domain-specific answer evaluation
        score = self._evaluate_answer(answer)
        
        base_reward = max(0, score - self.best_score)
        if score > self.best_score:
            self.best_score = score
        
        reward = base_reward * self.config.reward_scale
        reward -= self.config.step_penalty * self.step_count
        
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        terminated = score >= 0.8  # TODO: Adjust threshold
        feedback = f"Score: {score:.2f}"
        
        return feedback, reward, terminated
    
    async def _handle_answer_async(self, answer: str) -> Tuple[str, float, bool]:
        """Handle answer commands asynchronously."""
        # TODO: Implement your domain-specific async answer evaluation
        score = await self._evaluate_answer_async(answer)
        
        base_reward = max(0, score - self.best_score)
        if score > self.best_score:
            self.best_score = score
        
        reward = base_reward * self.config.reward_scale
        reward -= self.config.step_penalty * self.step_count
        
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        terminated = score >= 0.8  # TODO: Adjust threshold
        feedback = f"Score: {score:.2f}"
        
        return feedback, reward, terminated
    
    def _evaluate_answer(self, answer: str) -> float:
        """Evaluate the answer and return a score between 0 and 1."""
        # TODO: Implement your domain-specific evaluation logic
        ground_truth = self.current_data.get("ground_truth", "template answer")
        
        if answer.lower().strip() == ground_truth.lower().strip():
            return 1.0
        elif ground_truth.lower() in answer.lower():
            return 0.7
        else:
            return 0.3
    
    async def _evaluate_answer_async(self, answer: str) -> float:
        """Evaluate the answer asynchronously and return a score between 0 and 1."""
        # TODO: Implement your domain-specific async evaluation logic
        ground_truth = self.current_data.get("ground_truth", "template answer")
        
        if answer.lower().strip() == ground_truth.lower().strip():
            return 1.0
        elif ground_truth.lower() in answer.lower():
            return 0.7
        else:
            return 0.3
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data based on configuration."""
        if self.config.data_mode == "random":
            return self._get_random_data()
        elif self.config.data_mode == "single":
            return self._get_specific_data(self.config.data_source)
        elif self.config.data_mode == "list":
            return self._get_next_from_list(self.config.data_source)
        else:
            raise ValueError(f"Invalid data_mode: {self.config.data_mode}")
    
    def _get_random_data(self) -> Dict[str, Any]:
        """Get random data from the dataset."""
        # TODO: Load your actual data
        template_data = [
            {
                "id": "template_1",
                "description": "Template task description.",
                "goal": "Complete this task.",
                "ground_truth": "template answer",
            },
            {
                "id": "template_2", 
                "description": "Another template task.",
                "goal": "Solve this problem.",
                "ground_truth": "another answer",
            }
        ]
        return random.choice(template_data)
    
    def _get_specific_data(self, data_source: str) -> Dict[str, Any]:
        """Get specific data by ID."""
        # TODO: Implement specific data retrieval
        return self._get_random_data()  # Fallback
    
    def _get_next_from_list(self, data_source: list) -> Dict[str, Any]:
        """Get next data from a list."""
        # TODO: Implement list-based data cycling
        return self._get_random_data()  # Fallback
    
    def close(self):
        """Clean up the environment."""
        pass
