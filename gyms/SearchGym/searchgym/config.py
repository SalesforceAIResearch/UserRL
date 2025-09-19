import os
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class SearchGymConfig:
    """Complete configuration for SearchGym environment."""
    
    # Serper API configuration
    serper_api_key: str = ""
    api_key: str = ""
    
    # Environment configuration
    max_steps: int = 20
    verbose: bool = False
    seed: Optional[int] = 42

    # answer evaluation
    model_name: str = "gpt-4o"  # Model name for answer evaluation
    base_url: str = ""  # Custom base URL for the API endpoint
    # LLM configuration
    temperature: float = 0.0
    max_tokens: int = 512
    timeout: int = 10

    eval_method: str = "llm" # "llm" or "rule"
    
    # Scoring configuration
    correct_answer_reward: float = 1.0
    incorrect_answer_reward: float = 0.0
    step_penalty: float = 0.0
    normalize_rewards: bool = False
    
    # Search configuration
    max_search_results: int = 5
    max_search_steps: int = 5

    # Data configuration
    data_mode: str = "single"  # "random", "single", "list"
    data_source: Optional[Union[str, List[str]]] = None  # question IDs or None for random
    
    def __post_init__(self):
        # Auto-load Serper API key from secret.json or environment if not provided
        if not self.serper_api_key:
            self.serper_api_key = os.getenv("SERPER_API_KEY", "")

        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'SearchGymConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SearchGymConfig':
        """Load configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        if self.correct_answer_reward < 0:
            raise ValueError("correct_answer_reward must be non-negative")
        
        if self.max_search_results <= 0:
            raise ValueError("max_search_results must be positive")
        
        if self.data_mode not in ["random", "single", "list"]:
            raise ValueError("data_mode must be 'random', 'single', or 'list'")
        
        if self.data_mode in ["single", "list"] and self.data_source is None:
            raise ValueError(f"data_source must be provided when data_mode is '{self.data_mode}'")


def get_default_config() -> SearchGymConfig:
    """Get default configuration for SearchGym."""
    return SearchGymConfig()


def get_demo_config() -> SearchGymConfig:
    """Get demo configuration with verbose output enabled."""
    return SearchGymConfig(
        verbose=True,
        max_steps=10,
        correct_answer_reward=1.0,
        step_penalty=0.1
    ) 