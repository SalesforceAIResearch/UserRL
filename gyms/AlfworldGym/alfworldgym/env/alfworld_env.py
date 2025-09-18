import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, List
import random
import traceback
import asyncio
import sys

# Alfworld imports
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic

from ..config import AlfworldGymConfig, get_default_config
from ..data_loader import (
    parse_task_description, extract_action_from_text
)
from ..utils import validate_action_format


class AlfworldEnv(gym.Env):
    """
    Custom Gymnasium environment for alfworld household task completion.
    
    The agent can perform household actions and complete tasks,
    receiving rewards based on task completion success.
    """
    
    def __init__(self, config: AlfworldGymConfig = None):
        """
        Initialize the Alfworld Environment.
        
        Args:
            config: AlfworldGymConfig instance with all configuration settings
        """
        super().__init__()
        
        # Use provided config or default
        self.config = config or get_default_config()
        self.config.validate()
        
        # Initialize alfworld environment
        self._setup_alfworld_env()
        
        # Environment state
        self.current_task = None
        self.current_task_description = ""
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.task_completed = False
        self.current_score = 0.0
        self.total_reward = 0.0
        self.last_observation = ""
        self.last_alfworld_info = {}
        
        # Set random seed if provided
        if self.config.seed is not None:
            self.seed(self.config.seed)
        
        # Action space: discrete actions representing different action types
        # 0: [action], 1: [finish]
        self.action_space = spaces.Discrete(2)
        
        # Observation space: dictionary containing session state and feedback
        self.observation_space = spaces.Dict({
            "task": spaces.Text(max_length=1024),
            "feedback": spaces.Text(max_length=2048),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=np.int32),
            "episode_complete": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            "task_completed": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_)
        })
    
    def _setup_alfworld_env(self):
        """Setup the alfworld environment based on configuration."""
        try:
            sys.argv = ["dummy", self.config.env_config_path]
            # Load alfworld config
            alfworld_config = generic.load_config()
            
            # Override with our settings
            alfworld_config['env']['type'] = self.config.env_type
            
            # Get environment class
            env_type = alfworld_config['env']['type']
            env_class = get_environment(env_type)
            
            # Create environment
            self.alfworld_env = env_class(alfworld_config, train_eval=self.config.train_eval)
            self.alfworld_env = self.alfworld_env.init_env(batch_size=self.config.batch_size)
            
            if self.config.verbose:
                print(f"✅ Alfworld environment initialized: {env_type}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Failed to initialize alfworld environment: {e}")
                traceback.print_exc()
            raise RuntimeError(f"Failed to initialize alfworld environment: {e}")
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        if seed is not None:
            self.seed(seed)
        
        self.config.data_source = int(self.config.data_source)
        
        try:
            if self.config.data_source >= 0:
                try:
                    # Reset with specific index if data source is provided
                    obs, info = self.alfworld_env.reset(idx=self.config.data_source)
                except Exception as e:
                    obs, info = self.alfworld_env.reset()
            else:
                # Reset alfworld environment - this generates a new scenario
                obs, info = self.alfworld_env.reset()
            
            # Extract observation (alfworld returns list)
            if isinstance(obs, list) and len(obs) > 0:
                obs = obs[0]
            
            # Extract info (alfworld returns list)
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            
            # Parse task description
            task_info = parse_task_description(obs)
            self.current_task_description = task_info["task"]
            
            # Reset episode state
            self.step_count = 0
            self.episode_complete = False
            self.action_history = []
            self.task_completed = False
            self.current_score = 0.0
            self.total_reward = 0.0
            self.last_observation = obs
            self.last_alfworld_info = info
            
            # Create initial observation
            feedback = f"New household task started.\n\n{obs}"
            observation = {
                "task": self.current_task_description,
                "feedback": feedback,
                "step_count": self.step_count,
                "episode_complete": self.episode_complete,
                "task_completed": self.task_completed
            }
            
            # Create info dictionary
            gym_info = {
                "raw_action": "",
                "task_desc": self.current_task_description,
                "alfworld_info": info,
                "action_history": self.action_history,
                "step_count": self.step_count,
                "total_reward": self.total_reward,
                "admissible_commands": info.get("admissible_commands", [])[0] if info else []
            }
            
            if self.config.verbose:
                print(f"🔄 New alfworld episode started!")
                print(f"Task: {self.current_task_description}")
            
            return observation, gym_info
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Error during reset: {e}")
                traceback.print_exc()
            raise RuntimeError(f"Failed to reset alfworld environment: {e}")
    
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
            # Validate action format
            is_valid, error_msg = validate_action_format(action_str)
            if not is_valid:
                feedback = f"Invalid action format: {error_msg}"
                reward = 0.0
            else:
                # Process the action
                if action_str.startswith("[action]"):
                    feedback, reward = self._handle_action(action_str)
                elif action_str.startswith("[finish]"):
                    feedback, reward = self._handle_finish(action_str)
                else:
                    feedback = "Invalid action format. Please use [action] or [finish]."
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
        if self.task_completed:
            terminated = True
        elif action_str.startswith("[finish]"):
            terminated = True
        
        truncated = self.step_count >= self.config.max_steps
        
        if terminated or truncated:
            self.episode_completvalidate_action_formate = True
        
        # Create observation
        observation = {
            "task": self.current_task_description,
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "task_completed": self.task_completed
        }
        
        # Create info dictionary
        info = {
            "raw_action": action_str,
            "task_desc": self.current_task_description,
            "alfworld_info": self.last_alfworld_info,
            "action_history": self.action_history,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "admissible_commands": self.last_alfworld_info.get("admissible_commands", [])[0] if self.last_alfworld_info else []
        }
        
        if terminated and self.config.verbose:
            if self.task_completed:
                print(f"🎉 Task completed successfully! Episode completed!")
            else:
                print(f"❌ Episode ended without task completion.")
        elif truncated and self.config.verbose:
            print(f"⏰ Episode ended due to max steps.")
        
        return observation, reward, terminated, truncated, info
    
    def _handle_action(self, action_str: str) -> Tuple[str, float]:
        """Handle action and return feedback and reward."""
        try:
            # Extract action from input
            action = action_str[8:].strip()  # Remove [action] prefix

            # print(f"Executing action: {action}")
            obs, score, done, info = self.alfworld_env.step([action])
            obs = obs[0]; score = score[0]; done = done[0]

            # Update state
            self.last_observation = obs
            self.last_alfworld_info = info
            self.current_score = score
            
            admissible_commands = info.get("admissible_commands", [])[0]
            goto_count = 0
            command_string = "In your next action, the content could be following: "
            for cmd in admissible_commands:
                if cmd.startswith("go to"):
                    goto_count += 1
                    if goto_count == 1:
                        command_string += cmd.strip() + ", "
                    elif goto_count == 2:
                        command_string += cmd.split("go to")[1].strip() + ", "
                    elif goto_count == 3:
                        command_string += "etc. (any receptacles presented); "
                else:
                    command_string += cmd.strip() + "; "
            command_string = command_string.strip().rstrip(";") + ". According to your task goal, please wisely choose your next action through calling the tool."

            # Check if task is completed
            if score > 0 or done:
                self.task_completed = True
            
            # Format feedback
            feedback = obs.strip() + "\n" + command_string
            # Calculate reward
            reward = score
            
            if self.config.verbose:
                print(f"🎮 Action: {action}")
                print(f"📊 Score: {score}, Done: {done}")
            
            return feedback, reward
            
        except Exception as e:
            return f"Error executing action: {str(e)}", 0.0
    
    def _handle_finish(self, action_str: str) -> Tuple[str, float]:
        """Handle finish action and return feedback and reward."""
        if self.task_completed:
            feedback = f"Episode finished. Task was completed successfully! Final score: {self.current_score}"
            reward = 0.0  # No additional reward for finishing after success
        else:
            feedback = f"Episode finished without completing the task. Final score: {self.current_score}"
            reward = 0.0
        
        if self.config.verbose:
            print("🏁 Episode finished by agent")
        
        return feedback, reward
    
    def render(self, mode="human"):
        """Render the current state of the environment."""
        if not self.current_task_description:
            print("No task loaded. Call reset() first.")
            return
        
        print("\n" + "="*60)
        print(f"ALFWORLD GYM SESSION")
        print("="*60)
        print(f"Task: {self.current_task_description}")
        print(f"Steps taken: {self.step_count}/{self.config.max_steps}")
        print(f"Task completed: {self.task_completed}")
        print(f"Current score: {self.current_score}")
        
        if self.action_history:
            print(f"\n🎮 Action History:")
            for i, action in enumerate(self.action_history[-5:], 1):  # Show last 5 actions
                print(f"  {i}. {action}")
        
        if self.last_observation:
            print(f"\n📍 Last Observation:")
            print(f"  {self.last_observation[:200]}..." if len(self.last_observation) > 200 else f"  {self.last_observation}")
        
        print("="*60)
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'alfworld_env'):
            self.alfworld_env.close()
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
    
    async def step_async(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment (async version)."""
        # For alfworld, we can just call the sync version since alfworld doesn't have async support
        # This maintains the interface compatibility with SearchGym
        return self.step(action_input) 