import os
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class FunctionGymConfig:
    """Complete configuration for FunctionGym environment."""
    
    # Environment configuration
    max_steps: int = 20
    verbose: bool = False
    seed: Optional[int] = 42
    
    # Scoring configuration
    correct_answer_reward: float = 1.0
    incorrect_answer_reward: float = 0.0
    step_penalty: float = 0.0
    normalize_rewards: bool = False
    
    # Data configuration
    data_mode: str = "single"  # "random", "single", "list"
    data_source: Optional[Union[str, List[str]]] = None  # function IDs or None for random
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FunctionGymConfig':
        """Load configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        
        if self.correct_answer_reward < 0:
            raise ValueError("correct_answer_reward must be non-negative")
        
        if self.data_mode not in ["random", "single", "list"]:
            raise ValueError("data_mode must be 'random', 'single', or 'list'")
        
        if self.data_mode in ["single", "list"] and self.data_source is None:
            raise ValueError(f"data_source must be provided when data_mode is '{self.data_mode}'")


def get_default_config() -> FunctionGymConfig:
    """Get default configuration for FunctionGym."""
    return FunctionGymConfig()


def get_demo_config() -> FunctionGymConfig:
    """Get demo configuration with verbose output enabled."""
    return FunctionGymConfig(
        verbose=True,
        max_steps=10,
        correct_answer_reward=1.0,
        step_penalty=0.1
    ) 