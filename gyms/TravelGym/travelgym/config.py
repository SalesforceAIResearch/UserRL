"""
Configuration management for TravelGym environments.
"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Any
import os


@dataclass
class TravelGymConfig:
    """Configuration class for TravelGym environments."""
    
    # Data configuration
    data_mode: str = "random"  # "random", "single", "list"
    data_source: Union[str, List[str]] = "random"  # path, scenario key, or list of keys
    data_path: Optional[str] = None  # Override default data path
    
    # Model configuration
    api_key: Optional[str] = ""
    model_name: str = "gpt-4o"
    base_url: str = ""  # Custom base URL for the API endpoint
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 15.0
    
    # Environment configuration
    max_steps: int = 20
    search_failure_interval: int = 5 # every N times of search will yeild an system error on purpose
    elicitation_interval: int = 3 # will proactively elicit a preference if the conversation is off topic for N times consecutively
    
    # Reward configuration
    reward_scale: float = 1.0
    step_penalty: float = 0.0
    search_correct_reward: float = 0.2
    preference_correct_reward: float = 0.2
    choice_best_reward: float = 1.0
    choice_correct_reward: float = 0.8
    wrong_choice_penalty: float = 0.0

    # Choice number customization
    wrong_choice_number: int = 10  # Number of wrong choices to allow before penalty
    noise_choice_number: int = 5  # Number of noise choices to allow in the environment
    
    one_choice_per_aspect: bool = True  # Only one final choice per aspect allowed

    normalize_rewards: bool = False
    
    # Environment behavior
    verbose: bool = False
    seed: Optional[int] = None
    silent_fallback: bool = True
    enable_async: bool = True
    
    # Tracking configuration  
    track_conversation_history: bool = True
    track_preference_states: bool = True
    track_function_calls: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Set API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # Set default data path if not provided
        if self.data_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(current_dir, "data", "usergym_data_sample.json")
    
    def validate(self):
        """Validate configuration parameters."""
        if self.data_mode not in ["random", "single", "list"]:
            raise ValueError(f"Invalid data_mode: {self.data_mode}")
        
        if self.data_mode == "single" and not isinstance(self.data_source, str):
            raise ValueError("data_source must be a string when data_mode is 'single'")
        
        if self.data_mode == "list" and not isinstance(self.data_source, list):
            raise ValueError("data_source must be a list when data_mode is 'list'")
        
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TravelGymConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


def get_default_config() -> TravelGymConfig:
    """Get the default TravelGym configuration."""
    return TravelGymConfig()


def get_demo_config() -> TravelGymConfig:
    """Get a demonstration configuration with verbose output."""
    return TravelGymConfig(
        verbose=True,
        max_steps=15,
        temperature=0.5,
        user_response_style="natural",
        track_conversation_history=True,
        track_preference_states=True,
    ) 