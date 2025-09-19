import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import traceback
from typing import Dict, Any, Tuple, List
import random

# Tau-bench imports
from tau_bench.envs import get_env
from tau_bench.types import Action, RESPOND_ACTION_NAME

from ..config import TauGymConfig, get_default_config


def validate_action_format(action_str: str) -> bool:
    """Simple validation of action format."""
    action_str = action_str.strip()
    valid_prefixes = ["[search]", "[action]", "[answer]", "[finish]"]
    return any(action_str.startswith(prefix) for prefix in valid_prefixes)


def parse_tau_action(action_str: str) -> Action:
    """Parse TauGym action into tau-bench Action object."""
    action_str = action_str.strip()
    
    if action_str.startswith("[search]"):
        content = action_str[8:].strip()
        if content in ["tools", "help"]:
            return Action(name="search_help", kwargs={"query": content})
        else:
            return Action(name="search_tools", kwargs={"query": content})
    
    elif action_str.startswith("[action]"):
        # Direct user communication - use respond action
        content = action_str[8:].strip()
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": content})
    
    elif action_str.startswith("[answer]"):
        # Parse tool call from answer
        content = action_str[8:].strip()
        try:
            # Try to parse as JSON for tool call
            if content.startswith('{') and content.endswith('}'):
                tool_call = json.loads(content)
                name = tool_call.get("name", "")
                arguments = tool_call.get("arguments", {})
                return Action(name=name, kwargs=arguments)
            else:
                parts = content.split(':', 1)
                name = parts[0].strip()
                kwargs = json.loads(parts[1].strip())
                return Action(name=name, kwargs=kwargs)
        except Exception:
            # If parsing fails, treat as invalid action
            return Action(name="invalid_action", kwargs={"content": content})
    
    elif action_str.startswith("[finish]"):
        return Action(name="finish", kwargs={})
    
    else:
        return Action(name="invalid_action", kwargs={"content": action_str})


class TauEnv(gym.Env):
    """
    TauGym Environment - A proxy to tau-bench for RL agents.
    
    This environment translates between gym-style interactions and tau-bench,
    supporting [search], [action], [answer], and [finish] actions.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, config: TauGymConfig = None):
        """
        Initialize the TauGym Environment.
        
        Args:
            config: TauGymConfig instance with all configuration settings
        """
        super().__init__()
        
        # Use provided config or default
        self.config = config or get_default_config()
        self.config.validate()
        
        # Environment state
        self.tau_env = None
        self.current_task = None
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.total_reward = 0.0
        self.last_observation = ""
        self.last_tau_response = None
        
        # Set random seed if provided
        if self.config.seed is not None:
            self.seed(self.config.seed)
        
        # Action space: text actions with specific formats
        self.action_space = spaces.Text(max_length=2048)
        
        # Observation space: dictionary containing session state and feedback
        self.observation_space = spaces.Dict({
            "instruction": spaces.Text(max_length=1024),
            "feedback": spaces.Text(max_length=2048),
            "step_count": spaces.Box(low=0, high=self.config.max_steps, shape=(), dtype=np.int32),
            "episode_complete": spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
            "tools_info": spaces.Text(max_length=8192)  # Available tools info
        })
    
    def _setup_tau_env(self, task_index: int = None):
        """Setup the tau-bench environment."""
        try:
            self.tau_env = get_env(
                env_name=self.config.task_category,
                user_strategy=self.config.user_strategy,
                user_model=self.config.user_model,
                task_split=self.config.task_split,
                user_provider=self.config.user_provider,
                task_index=task_index
            )
            
            if self.config.verbose:
                print(f"✅ Tau-bench environment created: {self.config.task_category}")
                print(f"📊 Total tasks available: {len(self.tau_env.tasks)}")
                
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Failed to create tau-bench environment: {e}")
                traceback.print_exc()
            raise RuntimeError(f"Failed to create tau-bench environment: {e}")
    
    def _get_task_index_from_id(self, task_id: str) -> int:
        """Extract task index from task ID (e.g., 'retail-019' -> 19)."""
        if task_id is None:
            return 0
        
        try:
            # Extract number from task_id like "retail-019" or "airline-005"
            parts = str(task_id).split('-')
            if len(parts) >= 2:
                return int(parts[-1])
            else:
                return int(task_id)
        except (ValueError, IndexError):
            if self.config.verbose:
                print(f"⚠️ Invalid task_id format: {task_id}, using index 0")
            return 0
    
    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        if seed is not None:
            self.seed(seed)
        
        # Get task index from config
        task_index = self._get_task_index_from_id(self.config.data_source)
        
        # Setup tau-bench environment
        self._setup_tau_env(task_index)
        
        # Reset tau-bench environment
        reset_response = self.tau_env.reset(task_index=task_index)
        self.last_observation = reset_response.observation
        self.current_task = reset_response.info.task
        
        # Reset episode state
        self.step_count = 0
        self.episode_complete = False
        self.action_history = []
        self.total_reward = 0.0
        self.last_tau_response = None

        self.tools_info_called = False
        
        # Create tools info string
        tools_info = self._format_tools_info()
        
        # Create initial observation
        observation = {
            "instruction": self.current_task.instruction,
            "feedback": f"User says: {self.last_observation}",
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "tools_info": tools_info
        }
        
        # Create info dictionary
        info = {
            "task_id": self.config.data_source,
            "task_category": self.config.task_category,
            "task_index": task_index,
            "tau_task": self.current_task.model_dump()
        }
        
        if self.config.verbose:
            print(f"🔄 New episode started!")
            print(f"Task: {self.current_task.instruction}")
            print(f"User: {self.last_observation}")
        
        return observation, info
    
    def step(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.episode_complete:
            raise ValueError("Episode is complete. Call reset() to start a new episode.")
        
        self.step_count += 1
        action_str = str(action_input).strip()
        
        # Add action to history
        self.action_history.append(action_str)
        
        # Validate action format
        if not validate_action_format(action_str):
            feedback = "Invalid action format: " + action_str + ". Please follow the system instruction to call the interact tool."
            reward = 0.0
            terminated = False
            truncated = False
        else:
            try:
                feedback, reward, terminated, truncated = self._execute_action(action_str)
            except Exception as e:
                feedback = f"Error executing action: {str(e)}"
                reward = 0.0
                terminated = False
                truncated = False
                if self.config.verbose:
                    print(f"❌ Error in step: {e}")
                    traceback.print_exc()
        
        # Check truncation
        truncated = truncated or self.step_count >= self.config.max_steps
        
        if terminated or truncated:
            self.episode_complete = True
        
        self.total_reward += reward
        
        # Create observation
        observation = {
            "instruction": self.current_task.instruction,
            "feedback": feedback,
            "step_count": self.step_count,
            "episode_complete": self.episode_complete,
            "tools_info": self._format_tools_info()
        }
        
        # Create info dictionary
        info = {
            "raw_action": action_str,
            "task_id": self.config.data_source,
            "task_category": self.config.task_category,
            "action_history": self.action_history.copy(),
            "total_reward": self.total_reward,
            "tau_response": self.last_tau_response.model_dump() if self.last_tau_response else None
        }
        
        if self.config.verbose:
            print(f"🎯 Action: {action_str}")
            print(f"📥 Feedback: {feedback}")
            print(f"🏆 Reward: {reward}")
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action_str: str) -> Tuple[str, float, bool, bool]:
        """Execute the action and return feedback, reward, terminated, truncated."""
        
        if action_str.startswith("[search]"):
            return self._handle_search_action(action_str)
        elif action_str.startswith("[action]"):
            return self._handle_action_action(action_str)
        elif action_str.startswith("[answer]"):
            return self._handle_answer_action(action_str)
        elif action_str.startswith("[finish]"):
            return self._handle_finish_action(action_str)
        else:
            return "Invalid action type.", 0.0, False, False
    
    def _handle_search_action(self, action_str: str) -> Tuple[str, float, bool, bool]:
        """Handle [search] action - return tools or help info."""
        query = action_str[8:].strip().lower()
        
        if query == "tools":
            if not self.tools_info_called:
                self.tools_info_called = True
                feedback = self._format_tools_info()
            else:
                feedback = "Tools information is already provided in previous turns in our conversation. Please directly refer to those information and do not search for available tools again."
        elif query == "help":
            feedback = self._format_help_info()
        else:
            feedback = f"Search not supported for '{query}'. Please search for available tools by providing 'tools' in content or show help information by providing 'help' in content."
        
        return feedback, 0.0, False, False
    
    def _handle_action_action(self, action_str: str) -> Tuple[str, float, bool, bool]:
        """Handle [action] - direct communication with user."""
        tau_action = parse_tau_action(action_str)
        
        try:
            response = self.tau_env.step(tau_action)
            self.last_tau_response = response
            feedback = response.observation
            reward = response.reward
            terminated = response.done
            
            return feedback, reward, terminated, False
            
        except Exception as e:
            return f"Error communicating with user: {str(e)}", 0.0, False, False
    
    def _handle_answer_action(self, action_str: str) -> Tuple[str, float, bool, bool]:
        """Handle [answer] - tool call parsing and execution."""
        tau_action = parse_tau_action(action_str)
        
        if tau_action.name == "invalid_action":
            return "Invalid tool call format. Please provide valid JSON format in content when calling the internal tool.", 0.0, False, False
        
        try:
            response = self.tau_env.step(tau_action)
            self.last_tau_response = response
            feedback = response.observation
            reward = response.reward
            terminated = response.done
            
            return feedback, reward, terminated, False
            
        except Exception as e:
            return f"Error executing tool call: {str(e)}", 0.0, False, False
    
    def _handle_finish_action(self, action_str: str) -> Tuple[str, float, bool, bool]:
        """Handle [finish] - end episode."""
        # Check if we have any final reward from tau-bench
        final_reward = 0.0
        if self.last_tau_response and hasattr(self.last_tau_response, 'reward'):
            final_reward = self.last_tau_response.reward
        
        feedback = f"Episode finished. Final reward: {final_reward}"
        return feedback, final_reward, True, False
    
    def _format_tools_info(self) -> str:
        """Format available tools information as string."""
        if not self.tau_env or not hasattr(self.tau_env, 'tools_info'):
            return "No tools information available."
        
        info_lines = ["Available Tools:"]
        for tool in self.tau_env.tools_info:
            func_info = tool.get("function", {})
            name = func_info.get("name", "unknown")
            description = func_info.get("description", "No description")
            info_lines.append(f"- {name}: {description}")
            
            # Add parameters info
            params = func_info.get("parameters", {}).get("properties", {})
            if params:
                info_lines.append("  Parameters:")
                for param_name, param_info in params.items():
                    param_desc = param_info.get("description", "")
                    info_lines.append(f"    - {param_name}: {param_desc}")
        
        return "\n".join(info_lines)

    def _format_help_info(self) -> str:
        return """You should choose from one of the following:
        * search: search for available tools by providing "tools" in content or show this help information by providing "help" in content
        * action: directly communicate with user by providing a message in content
        * answer: call an internal tool by providing the tool name and its arguments in JSON format in content (e.g. {\"name\": tool_name, \"arguments\": {\"arg_1\": \"value_1\", \"arg_2\": \"value_2\"}})
        Please always call the `interact_with_env` tool to provide your choice and content for the next step. Usually you should first understand the user's request and figure out what internal tool you should invocate. Based on this final goal, you should then inversely think about how to get the information (argument contents) needed to reach this goal, either through invocating other internal tools or directly asking the user."""
    
    async def step_async(self, action_input: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment (async version)."""
        # For tau-bench, we just call the sync version since tau-bench doesn't have native async support
        # This maintains interface compatibility with other gyms
        return self.step(action_input)
    
    def render(self, mode="human"):
        """Render the current state of the environment."""
        if not self.current_task:
            print("No task loaded. Call reset() first.")
            return
        
        print("\n" + "="*60)
        print(f"TAU GYM SESSION")
        print("="*60)
        print(f"Category: {self.config.task_category}")
        print(f"Task ID: {self.config.data_source}")
        print(f"Instruction: {self.current_task.instruction}")
        print(f"Steps taken: {self.step_count}/{self.config.max_steps}")
        print(f"Total reward: {self.total_reward}")
        
        if self.last_observation:
            print(f"\n👤 Last User Message:")
            print(f"  {self.last_observation}")
        
        if self.action_history:
            print(f"\n🎯 Recent Actions:")
            for i, action in enumerate(self.action_history[-3:], 1):  # Show last 3 actions
                print(f"  {i}. {action}")
        
        print("="*60)
    
    def close(self):
        """Clean up resources."""
        pass
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
