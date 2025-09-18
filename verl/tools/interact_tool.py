# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Tuple
from copy import deepcopy
import torch
import asyncio
import time
import os

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from .env_manager import get_environment_manager

class InteractTool(BaseTool):
    """A tool for interacting with environments across multi-turn conversations.

    - `create`: create environment for a conversation (request_id)
    - `execute`: interact with the persistent environment  
    - `calc_reward`: calculate reward from environment state
    - `release`: clean up environment and conversation state
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._conversation_data = {}  # request_id -> conversation state
        self._env_manager = get_environment_manager()

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: str, env_name: Optional[str] = None, max_turns: int = 15, **kwargs) -> str:
        """Create environment and initialize conversation state.
        
        Args:
            instance_id: Request ID for the conversation (serves as conversation identifier)
            env_name: Type of environment to create
            max_turns: Maximum number of interaction turns
            **kwargs: Environment-specific configuration
            
        Returns:
            instance_id (request_id)
        """
        if instance_id in self._conversation_data:
            print(f"!!!!!!!! Conversation {instance_id} already exists !!!!!!!!")
            return instance_id
        
        # Create environment through environment manager
        if env_name:
            kwargs["max_turns"] = max_turns
            self._env_manager.create_environment(instance_id, env_name, **kwargs)
        
        # Initialize conversation state (separate from environment)
        self._conversation_data[instance_id] = {
            "history": [],
            "reward": 0.0,
            "ground_truth": kwargs.get("ground_truth"),
            "env_name": env_name,
        }
        
        print(f"Created conversation {instance_id} with {env_name} environment")
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], current_turns, **kwargs) -> Tuple[str, float, dict]:
        """Execute action in the persistent environment.
        
        Args:
            instance_id: Request ID (conversation identifier)
            parameters: Action parameters (choice, content)
            
        Returns:
            (response_text, step_reward, is_terminated)
        """
        
        if instance_id not in self._conversation_data:
            raise ValueError(f"Conversation {instance_id} not found. Call create() first.")
        
        # Get persistent environment
        env = self._env_manager.get_environment(instance_id)
        if env is None:
            raise ValueError(f"Environment for conversation {instance_id} not found")
        
        # Parse action parameters
        choice = str(parameters.get("choice", ""))
        content = str(parameters.get("content", ""))
        
        # Format action for environment
        if choice == "action" and not content.startswith("[action]"):
            formatted_action = "[action] " + content
        elif choice == "answer" and not content.startswith("[answer]"):
            formatted_action = "[answer] " + content
        elif choice == "search" and not content.startswith("[search]"):
            formatted_action = "[search] " + content
        elif choice == "finish":
            formatted_action = "[finish]"
        else:
            formatted_action = content
        
        try:
            # Add timeout to prevent hanging for too long
            observation, reward, terminated, truncated, info = await asyncio.wait_for(
                env.step_async(formatted_action),
                timeout=30.0  # 30 seconds timeout
            )
        except asyncio.TimeoutError:
            print(f"Environment step timed out for {instance_id} after 30s")
            # Fallback: Try in separate process to avoid NCCL interference
            try:
                print(f"Attempting fallback process isolation for {instance_id}")
                result = await asyncio.to_thread(
                    self._run_env_in_process, env, formatted_action
                )
                observation, reward, terminated, truncated, info = result
            except Exception as e:
                print(f"Process isolation fallback failed: {e}")
                observation = {"feedback": "Environment operation failed completely"}
                reward, terminated, truncated, info = 0.0, True, False, {}
        except Exception as e:
            print(f"Environment step failed for {instance_id}: {e}")
            # Return safe fallback values
            observation = {"feedback": f"Error: {str(e)}"}
            reward, terminated, truncated, info = 0.0, True, False, {}

        # Update conversation state
        conversation_state = self._conversation_data[instance_id]
        current_env_name = conversation_state["env_name"]
        conversation_state["reward"] = reward
        conversation_state["history"].append({
            "choice": choice,
            "content": content,
            "observation": observation,
            "reward": reward,
            "info": info
        })
        
        # Format response
        feedback = observation.get("feedback", "") if isinstance(observation, dict) else str(observation)
        response_text = f"{feedback}\nReward: {reward}"
        
        is_done = terminated or truncated
        print(f"Turn {current_turns}: Executed {choice} in conversation {instance_id} (Env: {current_env_name}), action: {formatted_action}, feedback: {feedback}, reward: {reward}, done: {is_done}")

        return response_text, reward, is_done, choice, content, {}

    def _run_env_in_process(self, env, formatted_action):
        """Run environment step in separate process to isolate from NCCL context."""
        try:
            # Use synchronous step since we're in a separate process
            if hasattr(env, 'step'):
                return env.step(formatted_action)
            else:
                # If only async available, run in new event loop
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(env.step_async(formatted_action))
        except Exception as e:
            return {"feedback": f"Process error: {e}"}, 0.0, True, False, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate final reward for the conversation.
        
        Args:
            instance_id: Request ID (conversation identifier)
            
        Returns:
            Final conversation reward
        """
        if instance_id not in self._conversation_data:
            print(f"!!!!!!!! Conversation {instance_id} not found for reward calculation !!!!!!!!")
            return 0.0
        
        conversation_state = self._conversation_data[instance_id]
        
        # Return the highest reward achieved during the conversation
        if conversation_state["history"]:
            max_reward = max(step["reward"] for step in conversation_state["history"])
            return max_reward
        
        return conversation_state["reward"]
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Clean up conversation and environment.
        
        Args:
            instance_id: Request ID (conversation identifier)
        """
        # Clean up conversation state
        if instance_id in self._conversation_data:
            del self._conversation_data[instance_id]
        
        # Clean up environment through manager
        self._env_manager.release_environment(instance_id)
        
