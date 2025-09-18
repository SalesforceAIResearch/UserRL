import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List
import random
import traceback
import re

from ..data_loader import load_functions, get_function_by_id, evaluate_function, compare_answers
from ..config import FunctionGymConfig, get_default_config


class FunctionEnv(gym.Env):
    """
    Custom Gymnasium environment for mathematical function learning.
    
    The agent can perform calculations with four numbers and learn the underlying rule,
    receiving rewards based on answer correctness for test cases.
    """
    
    def __init__(self, config: FunctionGymConfig = None):
        """
        Initialize the Function Environment.
        
        Args:
            config: FunctionGymConfig instance with all configuration settings
        """
        super().__init__()
        
        # Use provided config or default
        self.config = config or get_default_config()
        self.config.validate()
        
        # Load functions based on data configuration
        self._load_functions()
        
        # Environment state
        self.current_function = None
        self.current_function_index = 0
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.answered_correctly = False
        self.submitted_answer = None
        self.total_reward = 0.0
        
        # Set random seed if provided
        if self.config.seed is not None:
            self.seed(self.config.seed)
        
        # Action space: discrete actions representing different action types
        # 0: [action], 1: [search], 2: [answer], 3: [finish]
        self.action_space = spaces.Discrete(4)
        
        # Observation space: dictionary containing session state and feedback
        self.observation_space = spaces.Dict({
            "feedback": spaces.Text(max_length=2048),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=np.int32),
            "episode_complete": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            "answered_correctly": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_)
        })
    
    def _load_functions(self):
        """Load functions based on configuration."""
        all_functions = load_functions()
        
        if self.config.data_mode == "random":
            self.functions = all_functions
        elif self.config.data_mode == "single":
            # Find the specific function
            function = get_function_by_id(self.config.data_source)
            self.functions = [function]
        elif self.config.data_mode == "list":
            # Load multiple specific functions
            self.functions = []
            for function_id in self.config.data_source:
                try:
                    function = get_function_by_id(function_id)
                    self.functions.append(function)
                except ValueError:
                    if self.config.verbose:
                        print(f"Warning: Function '{function_id}' not found, skipping")
            
            if not self.functions:
                raise ValueError("No valid functions found in data_source list")
        
        if self.config.verbose:
            print(f"Loaded {len(self.functions)} functions in {self.config.data_mode} mode")
    
    def _get_next_function(self):
        """Get the next function based on data mode."""
        if self.config.data_mode == "random":
            return random.choice(self.functions)
        else:
            # For single and list modes, cycle through functions
            function = self.functions[self.current_function_index % len(self.functions)]
            self.current_function_index += 1
            return function
    
    def reset(self, seed=None):
        """Reset the environment to start a new episode."""
        if seed is not None:
            self.seed(seed)
        
        # Reset episode state
        self.current_function = self._get_next_function()
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.answered_correctly = False
        self.submitted_answer = None
        self.total_reward = 0.0
        
        # Create initial observation
        observation = {
            "feedback": "New function problem loaded. You can use [action] with four numbers to test the function, [search] for test case, or [answer] with your final guess.",
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "answered_correctly": self.answered_correctly
        }
        
        # Create info dictionary
        info = {
            "function_id": self.current_function["id"],
            "rule": self.current_function["rule"],
            "test_case": self.current_function["test_case"],
            "expected_result": self.current_function["expected_result"],
            "action_history": self.action_history.copy()
        }
        
        if self.config.verbose:
            print(f"🔄 New episode started!")
            print(f"Function ID: {self.current_function['id']}")
            print(f"Hidden rule: {self.current_function['rule']}")
        
        return observation, info
    
    def step(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        action_str = str(action_input).strip()
        
        # Add action to history
        self.action_history.append(action_str)
        
        feedback = ""
        reward = 0.0
        
        try:
            # Process different action types
            if action_str.startswith("[action]"):
                feedback, reward = self._handle_action_action(action_str)
            elif action_str.startswith("[search]"):
                feedback, reward = self._handle_search_action(action_str)
            elif action_str.startswith("[answer]"):
                feedback, reward = self._handle_answer_action(action_str)
            elif action_str.startswith("[finish]"):
                feedback, reward = self._handle_finish_action(action_str)
            else:
                feedback = "Invalid action format. Please use [action], [search], [answer], or [finish]."
                reward = 0.0
        
        except Exception as e:
            feedback = f"Error processing action: {str(e)}"
            reward = 0.0
            if self.config.verbose:
                print(f"Error in step: {e}")
                traceback.print_exc()
        
        # Apply step penalty
        step_penalty = self.config.step_penalty * self.step_count
        reward -= step_penalty
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards:
            reward = max(0.0, min(1.0, reward))
        
        self.total_reward += reward
        
        # Check termination conditions
        terminated = False
        if action_str.startswith("[answer]") and self.answered_correctly:
            terminated = True
        elif action_str.startswith("[finish]"):
            terminated = True
        
        truncated = self.step_count >= self.config.max_steps
        
        if terminated or truncated:
            self.episode_complete = True
        
        # Create observation
        observation = {
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "answered_correctly": self.answered_correctly
        }
        
        # Create info dictionary
        info = {
            "raw_action": action_str,
            "function_id": self.current_function["id"],
            "rule": self.current_function["rule"],
            "test_case": self.current_function["test_case"],
            "expected_result": self.current_function["expected_result"],
            "submitted_answer": self.submitted_answer,
            "action_history": self.action_history.copy(),
            "total_reward": self.total_reward
        }
        
        if terminated and self.config.verbose:
            if self.answered_correctly:
                print(f"🎉 Correct answer! Episode completed successfully!")
            else:
                print(f"❌ Episode ended without correct answer.")
        elif truncated and self.config.verbose:
            print(f"⏰ Episode ended due to max steps.")
        
        return observation, reward, terminated, truncated, info
    
    def _handle_action_action(self, action_str: str) -> Tuple[str, float]:
        """Handle action with four numbers and return calculation result."""
        numbers_str = action_str[8:].strip()  # Remove "[action]" prefix
        
        if not numbers_str:
            return "Please provide four numbers after [action].", 0.0
        
        try:
            # Parse four numbers from the input
            numbers = self._parse_numbers(numbers_str)
            if len(numbers) != 4:
                return "Please provide exactly four numbers.", 0.0
            
            # Check if these are the test case numbers
            if numbers == self.current_function["test_case"]:
                return "This is the test case and I cannot provide you with the calculation feedback.", 0.0
            
            # Calculate result using the hidden rule
            result = evaluate_function(numbers, self.current_function["rule"])
            # round the result to 4 decimal places
            result = round(result, 4)
            
            feedback = f"For numbers {numbers}, the result is {result}."
            
            if self.config.verbose:
                print(f"🧮 Calculated: {numbers} -> {result}")
            
            return feedback, 0.0  # No reward for calculation actions
            
        except Exception as e:
            return f"Error parsing numbers or calculating result. This may happen like when division by zero occurs. Please try again with different number combinations.", 0.0
    
    def _handle_search_action(self, action_str: str) -> Tuple[str, float]:
        """Handle search action for test case."""
        search_content = action_str[8:].strip()  # Remove "[search]" prefix
        
        if search_content.lower() != "test":
            return "Search content should only be 'test' to get the test case.", 0.0
        
        test_case = self.current_function["test_case"]
        feedback = f"Test case: {test_case}"
        
        if self.config.verbose:
            print(f"🔍 Revealed test case: {test_case}")
        
        return feedback, 0.0  # No reward for search actions
    
    def _handle_answer_action(self, action_str: str) -> Tuple[str, float]:
        """Handle answer action and evaluate correctness."""
        answer_str = action_str[8:].strip()  # Remove "[answer]" prefix
        
        if not answer_str:
            return "Answer is empty. Please provide a numerical answer.", 0.0
        
        try:
            answer = float(answer_str)
            self.submitted_answer = answer
            expected = self.current_function["expected_result"]
            
            # Compare answers
            self.answered_correctly = compare_answers(answer, expected)
            
            if self.answered_correctly:
                feedback = "Your answer is correct!"
                reward = self.config.correct_answer_reward
            else:
                feedback = "Your answer is wrong. Please continue to try and reason about the hidden function and make your answer attempt later."
                reward = self.config.incorrect_answer_reward
            
            if self.config.verbose:
                print(f"💭 Submitted answer: {answer}")
                print(f"✅ Correct!" if self.answered_correctly else f"❌ Incorrect (expected: {expected})")
            
            return feedback, reward
            
        except ValueError:
            return "Invalid numerical answer. Please provide a valid number.", 0.0
    
    def _handle_finish_action(self, action_str: str) -> Tuple[str, float]:
        """Handle finish action."""
        if self.answered_correctly:
            feedback = "Episode finished. You answered correctly!"
            reward = 0.0  # No additional reward for finishing after correct answer
        else:
            feedback = f"Episode finished without correct answer. The expected result for the test case was: {self.current_function['expected_result']}"
            reward = 0.0
        
        if self.config.verbose:
            print("🏁 Episode finished by agent")
        
        return feedback, reward
    
    def _parse_numbers(self, numbers_str: str) -> List[float]:
        """Parse numbers from string input."""
        # Use regex to find all numbers (including floats)
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, numbers_str)
        return [float(match) for match in matches if match]
    
    def render(self, mode="human"):
        """Render the current state of the environment."""
        if not self.current_function:
            print("No function loaded. Call reset() first.")
            return
        
        print("\n" + "="*60)
        print(f"FUNCTION GYM SESSION")
        print("="*60)
        print(f"Function ID: {self.current_function['id']}")
        print(f"Steps taken: {self.step_count}/{self.config.max_steps}")
        print(f"Answered correctly: {self.answered_correctly}")
        
        if self.submitted_answer is not None:
            print(f"\n💭 Submitted Answer: {self.submitted_answer}")
        
        if self.action_history:
            print(f"\nLast action: {self.action_history[-1]}")
        
        print("="*60)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
    
    async def step_async(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment (async version)."""
        # For FunctionGym, async and sync versions are the same since no external API calls
        return self.step(action_input) 