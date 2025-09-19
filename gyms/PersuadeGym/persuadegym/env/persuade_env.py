import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List
import random
import asyncio

from .prompts import evaluate_action, evaluate_action_async, stance_to_score
from .statement_data import load_statements, get_statement_by_id
from ..config import PersuadeGymConfig, get_default_config


class PersuadeEnv(gym.Env):
    """
    Custom Gymnasium environment for persuasion simulation.
    
    The agent attempts to persuade an AI environment that holds initial beliefs
    about various statements. The environment responds with arguments and tracks
    how much its position changes over time.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: PersuadeGymConfig = None):
        """
        Initialize the Persuade Environment.
        
        Args:
            config: PersuadeGymConfig instance with all configuration settings
        """
        super().__init__()
        
        # Use provided config or default
        self.config = config or get_default_config()
        self.config.validate()
        
        # Load statements based on data configuration
        self._load_statements()
        
        # Environment state
        self.current_statement = None
        self.current_statement_index = 0
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.conversation_history = []  # Track (persuader_arg, env_response) pairs
        self.current_stance = ""
        self.current_confidence = 0.0
        self.best_score = 0.0
        self.best_json_response = None
        self.total_reward = 0.0
        
        # Set random seed if provided
        if self.config.seed is not None:
            self.seed(self.config.seed)
        
        # Action space: discrete actions representing different action types
        self.action_space = spaces.Discrete(3)
        
        # Observation space: dictionary containing persuasion state and feedback
        self.observation_space = spaces.Dict({
            "statement_title": spaces.Text(max_length=1024),
            "statement_text": spaces.Text(max_length=1024),
            "goal": spaces.Text(max_length=256),
            "feedback": spaces.Text(max_length=1024),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=np.int32),
            "episode_complete": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            "current_score": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "best_score": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "current_stance": spaces.Text(max_length=256),
            "confidence_level": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)
        })
    
    def _load_statements(self):
        """Load statements based on configuration."""
        all_statements = load_statements()
        
        if self.config.data_mode == "random":
            self.statements = all_statements
        elif self.config.data_mode == "single":
            # Find the specific statement
            statement = get_statement_by_id(self.config.data_source)
            self.statements = [statement]
        elif self.config.data_mode == "list":
            # Load multiple specific statements
            self.statements = []
            for id in self.config.data_source:
                try:
                    statement = get_statement_by_id(id)
                    self.statements.append(statement)
                except ValueError:
                    if self.config.verbose:
                        print(f"Warning: Statement '{id}' not found, skipping")
            
            if not self.statements:
                raise ValueError("No valid statements found in data_source list")
        
        if self.config.verbose:
            print(f"Loaded {len(self.statements)} statements in {self.config.data_mode} mode")
    
    def _get_next_statement(self):
        """Get the next statement based on data mode."""
        if self.config.data_mode == "random":
            return random.choice(self.statements)
        elif self.config.data_mode == "single":
            return self.statements[0]
        elif self.config.data_mode == "list":
            # Cycle through the list
            statement = self.statements[self.current_statement_index]
            self.current_statement_index = (self.current_statement_index + 1) % len(self.statements)
            return statement
    
    def reset(self, seed=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment to start a new persuasion session."""
        super().reset(seed=seed)
        
        # Get next statement
        self.current_statement = self._get_next_statement()
        
        # Set default initial stance (no confidence needed)
        self.current_stance = "Strongly Agree"
        
        # Reset episode state
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.conversation_history = []
        self.best_score = stance_to_score(self.current_stance)
        self.best_json_response = None
        self.total_reward = 0.0
        
        # Create initial observation
        observation = {
            "statement_title": self.current_statement.get("claim", "")[:100],  # Use first 100 chars of claim as title
            "statement_text": self.current_statement["claim"],
            "goal": "Try to persuade me to change my position on this statement.",
            "feedback": f"Welcome to PersuadeGym! I currently {self.current_stance.lower()} with this statement: '{self.current_statement['claim']}'. Here's my reasoning: {self.current_statement['argument']} Try to change my mind!",
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "current_score": stance_to_score(self.current_stance),
            "best_score": self.best_score,
            "current_stance": self.current_stance
        }
        
        info = {
            "initial_stance": self.current_stance,
            "action_history": self.action_history.copy(),
            "conversation_history": self.conversation_history.copy()
        }
        
        if self.config.verbose:
            print(f"🗣️ New persuasion session: {self.current_statement['id']}")
            print(f"Initial stance: {self.current_stance}")
        
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
        
        # Evaluate the action using LLM
        feedback, new_stance, reward, json_response, judge_success = evaluate_action(
            action_str, self.current_statement, model_config, 
            self.conversation_history, self.current_stance
        )
        
        # Update conversation history (for all non-finish actions)
        if not action_str.startswith("[finish]"):
            # Extract the actual argument for history
            if action_str.startswith("[action]"):
                persuader_arg = action_str[8:].strip()
            elif action_str.startswith("[answer]"):
                persuader_arg = action_str[8:].strip()
            else:
                persuader_arg = action_str.strip()
            
            self.conversation_history.append((persuader_arg, feedback))
        
        # Track stance change
        old_stance = self.current_stance
        self.current_stance = new_stance
        
        # Update best score tracking
        current_score = stance_to_score(self.current_stance)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_json_response = json_response
        
        # Apply step penalty if configured
        final_reward = reward
        if self.config.step_penalty > 0:
            final_reward -= self.config.step_penalty
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            final_reward = max(0.0, min(1.0, final_reward))
        
        self.total_reward += final_reward
        
        # Check if episode should end
        terminated = False
        truncated = False
        
        if action_str.startswith("[finish]"):
            terminated = True
            self.episode_complete = True
        elif self.current_stance == "Strongly Disagree":
            # Terminate when AI reaches complete disagreement
            terminated = True
            self.episode_complete = True
            if self.config.verbose:
                print("🎉 SUCCESS! AI reached 'Strongly Disagree' - conversation complete!")
        elif self.step_count >= self.config.max_steps:
            truncated = True
            self.episode_complete = True
        
        # Create observation
        observation = {
            "statement_title": self.current_statement.get("claim", "")[:100],
            "statement_text": self.current_statement["claim"],
            "goal": "Try to persuade me to change my position on this statement.",
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "current_score": current_score,
            "best_score": self.best_score,
            "current_stance": self.current_stance
        }
        
        info = {
            "initial_stance": "Strongly Agree",
            "stance_change": f"{old_stance} -> {new_stance}",
            "levels_changed": reward,  # The raw reward shows how many levels changed
            "action_history": self.action_history.copy(),
            "conversation_history": self.conversation_history.copy(),
            "json_response": json_response,
            "judge_success": judge_success
        }
        
        if self.config.verbose:
            print(f"Step {self.step_count}: {action_str[:50]}...")
            print(f"Response: {feedback[:100]}...")
            print(f"Stance: {old_stance} -> {new_stance}")
            print(f"Reward: {final_reward:.3f}")
        
        return observation, final_reward, terminated, truncated, info
    
    async def step_async(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Async version of step method."""
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
        
        # Evaluate the action using async LLM
        feedback, new_stance, reward, json_response, judge_success = await evaluate_action_async(
            action_str, self.current_statement, model_config,
            self.conversation_history, self.current_stance
        )
        
        # Update conversation history (for all non-finish actions)
        if not action_str.startswith("[finish]"):
            # Extract the actual argument for history
            if action_str.startswith("[action]"):
                persuader_arg = action_str[8:].strip()
            elif action_str.startswith("[answer]"):
                persuader_arg = action_str[8:].strip()
            else:
                persuader_arg = action_str.strip()
            
            self.conversation_history.append((persuader_arg, feedback))
        
        # Track stance change
        old_stance = self.current_stance
        self.current_stance = new_stance
        
        # Update best score tracking
        current_score = stance_to_score(self.current_stance)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_json_response = json_response
        
        # Apply step penalty if configured
        final_reward = reward
        if self.config.step_penalty > 0:
            final_reward -= self.config.step_penalty
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            final_reward = max(0.0, min(1.0, final_reward))
        
        self.total_reward += final_reward
        
        # Check if episode should end
        terminated = False
        truncated = False
        
        if action_str.startswith("[finish]"):
            terminated = True
            self.episode_complete = True
        elif self.current_stance == "Strongly Disagree":
            # Terminate when AI reaches complete disagreement
            terminated = True
            self.episode_complete = True
        elif self.step_count >= self.config.max_steps:
            truncated = True
            self.episode_complete = True
        
        # Create observation
        observation = {
            "statement_title": self.current_statement.get("claim", "")[:100],
            "statement_text": self.current_statement["claim"],
            "goal": "Try to persuade me to change my position on this statement.",
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "current_score": current_score,
            "best_score": self.best_score,
            "current_stance": self.current_stance
        }
        
        info = {
            "initial_stance": "Strongly Agree",
            "stance_change": f"{old_stance} -> {new_stance}",
            "levels_changed": reward,  # The raw reward shows how many levels changed
            "action_history": self.action_history.copy(),
            "conversation_history": self.conversation_history.copy(),
            "json_response": json_response,
            "judge_success": judge_success
        }
        
        return observation, final_reward, terminated, truncated, info
    
    def render(self, mode="human"):
        """Render the environment state."""
        if mode == "human":
            print(f"\n=== PersuadeGym Step {self.step_count} ===")
            print(f"Statement: {self.current_statement['statement']}")
            print(f"Current Stance: {self.current_stance}")
            print(f"Confidence: {self.current_confidence:.2f}")
            print(f"Score: {stance_to_score(self.current_stance):.3f}")
            if self.conversation_history:
                print(f"Last exchange:")
                last_arg, last_response = self.conversation_history[-1]
                print(f"  You: {last_arg}")
                print(f"  Me: {last_response}")
            print("=" * 40)
        else:
            super().render(mode=mode)
    
    def close(self):
        """Close the environment."""
        pass
    
    def seed(self, seed=None):
        """Set the random seed."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed] 