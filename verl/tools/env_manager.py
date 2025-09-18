"""
Environment Manager for Multi-turn Interactions
Handles environment lifecycle and persistence across conversation turns.
"""

import logging
from typing import Any, Dict, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Manages environment instances for multi-turn conversations."""
    
    def __init__(self):
        self._environments: Dict[str, Any] = {}  # request_id -> environment
        self._env_configs: Dict[str, Dict] = {}   # request_id -> config
    
    def create_environment(self, request_id: str, env_name: str, **kwargs) -> str:
        """Create and store environment for a conversation.
        
        Args:
            request_id: Unique identifier for the conversation
            env_name: Type of environment (TurtleGym, TelepathyGym, etc.)
            **kwargs: Environment-specific configuration
            
        Returns:
            request_id for the created environment
        """
        if request_id in self._environments:
            logger.warning(f"Environment for request_id {request_id} already exists")
            return request_id
            
        env = None
        
        if env_name == "TurtleGym":
            env = self._create_turtlegym_environment(**kwargs)
        elif env_name == "TelepathyGym":
            env = self._create_telepathygym_environment(**kwargs)
        elif env_name == "PersuadeGym":
            env = self._create_persuadegym_environment(**kwargs)
        elif env_name == "IntentionGym":
            env = self._create_intentiongym_environment(**kwargs)
        elif env_name == "TravelGym":
            env = self._create_travelgym_environment(**kwargs)
        elif env_name == "SearchGym":
            env = self._create_searchgym_environment(**kwargs)
        elif env_name == "TauGym":
            env = self._create_taugym_environment(**kwargs)
        elif env_name == "FunctionGym":
            env = self._create_functiongym_environment(**kwargs)
        else:
            raise ValueError(f"Unknown environment type: {env_name}")
        
        self._environments[request_id] = env
        self._env_configs[request_id] = {
            "env_name": env_name,
            "kwargs": kwargs
        }
        
        logger.info(f"Created {env_name} environment for request {request_id}")
        return request_id
    
    def get_environment(self, request_id: str) -> Optional[Any]:
        """Get environment for a conversation."""
        return self._environments.get(request_id)
    
    def release_environment(self, request_id: str) -> None:
        """Clean up environment for a conversation."""
        if request_id in self._environments:
            # Cleanup environment if it has cleanup methods
            env = self._environments[request_id]
            if hasattr(env, 'close'):
                env.close()
            
            del self._environments[request_id]
            del self._env_configs[request_id]
            logger.info(f"Released environment for request {request_id}")
    
    def _create_turtlegym_environment(self, **kwargs):
        """Create TurtleGym environment."""
        import turtlegym
        
        env_config = turtlegym.get_default_config()
        
        # Configure from kwargs
        title = kwargs.get("title")
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        
        env_config.max_steps = max_turns
        env_config.success_threshold = 1.0
        env_config.data_mode = "single"
        env_config.data_source = title
        env_config.model_name = model_name
        
        env = turtlegym.StoryEnv(config=env_config)
        env.reset()
        
        return env
    
    def _create_telepathygym_environment(self, **kwargs):
        """Create TelepathyGym environment."""
        import telepathygym
        
        env_config = telepathygym.get_default_config()
        
        # Configure from kwargs
        title = kwargs.get("title")
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = title
        env_config.model_name = model_name

        env = telepathygym.TelepathyEnv(config=env_config)
        env.reset()
        
        return env

    def _create_persuadegym_environment(self, **kwargs):
        """Create PersuadeGym environment."""
        import persuadegym
        
        env_config = persuadegym.get_default_config()
        
        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        id = kwargs.get("id")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name
        
        env = persuadegym.PersuadeEnv(config=env_config)
        env.reset()
        
        return env

    def _create_intentiongym_environment(self, **kwargs):
        """Create IntentionGym environment."""
        import intentiongym
        
        env_config = intentiongym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 15)
        model_name = kwargs.get("model_name", "gpt-4o-mini")
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name

        env = intentiongym.IntentionEnv(config=env_config)
        env.reset()

        return env

    def _create_travelgym_environment(self, **kwargs):
        """Create TravelGym environment."""
        import travelgym
        
        env_config = travelgym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        model_name = kwargs.get("model_name", "gpt-4o")
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.model_name = model_name
        env_config.one_choice_per_aspect = True

        env_config.search_correct_reward = 0.0
        env_config.preference_correct_reward = 0.8

        env = travelgym.TravelEnv(config=env_config)
        env.reset()

        return env

    def _create_searchgym_environment(self, **kwargs):
        """Create SearchGym environment."""
        import searchgym
        
        env_config = searchgym.get_default_config()
        model_name = kwargs.get("model_name", "gpt-4o")
        
        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        id = kwargs.get("id")

        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.eval_method = "llm"
        env_config.model_name = model_name

        env = searchgym.SearchEnv(config=env_config)
        env.reset()

        return env
    
    def _create_taugym_environment(self, **kwargs):
        """Create TauGym environment."""
        import taugym
        
        env_config = taugym.get_default_config()
        
        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        model_name = kwargs.get("model_name", "gpt-4o")
        id = kwargs.get("id")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        env_config.user_model = model_name

        if "retail" in id:
            env_config.task_category = "retail"
        else:
            env_config.task_category = "airline"
        if "train" in id:
            env_config.task_split = "train"
        else:
            env_config.task_split = "test"

        env = taugym.TauEnv(config=env_config)
        env.reset()
        
        return env
    
    def _create_functiongym_environment(self, **kwargs):
        """Create FunctionGym environment."""
        import functiongym
        
        env_config = functiongym.get_default_config()

        # Configure from kwargs
        max_turns = kwargs.get("max_turns", 20)
        id = kwargs.get("id")
        
        env_config.max_steps = max_turns
        env_config.data_mode = "single"
        env_config.data_source = id
        
        env = functiongym.FunctionEnv(config=env_config)
        env.reset()
        
        return env

# Global environment manager instance
_env_manager = EnvironmentManager()


def get_environment_manager() -> EnvironmentManager:
    """Get the global environment manager instance."""
    return _env_manager 