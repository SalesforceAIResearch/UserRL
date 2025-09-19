# Contributing a New Gym to UserRL

This guide provides instructions for creating and contributing a new gym to the UserRL project.

## Table of Contents

1. [Overview](#overview)
2. [Required Files Structure](#required-files-structure)
3. [Implementation Guidelines](#implementation-guidelines)
4. [Action Format Standards](#action-format-standards)
5. [Testing Requirements](#testing-requirements)
6. [Template Gym](#template-gym)
7. [Submission Process](#submission-process)

## Overview

A UserRL gym is a Gymnasium-compatible environment that enables reinforcement learning research in specific domains. Each gym follows a standardized structure while implementing domain-specific logic.

## Required Files Structure

### Standard Structure
```
YourGym/
├── yourgym/                    # Main package
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Configuration management
│   ├── env/                   # Environment implementation
│   │   ├── __init__.py        # Environment module init
│   │   └── your_env.py        # Main environment class
│   └── data/                  # Domain-specific data
│       └── your_data.json     # Data files
├── README.md                  # Documentation
├── setup.py                   # Package setup
├── requirements.txt           # Dependencies
└── test_your_gym.py          # Basic tests
```

### Required Files

#### `yourgym/__init__.py`
```python
from .config import YourGymConfig, get_default_config, get_demo_config
from .env.your_env import YourEnv

__all__ = ["YourEnv", "YourGymConfig", "get_default_config", "get_demo_config"]

try:
    import gymnasium as gym
    gym.register(id='YourGym-v0', entry_point='yourgym.env:YourEnv', max_episode_steps=20)
except ImportError:
    pass
```

#### `yourgym/config.py`
```python
import os
from dataclasses import dataclass
from typing import Optional, Union, List

@dataclass
class YourGymConfig:
    """Configuration class for YourGym environment."""
    
    # Model configuration
    api_key: str = ""
    model_name: str = "gpt-4o"
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 10
    
    # Environment configuration
    max_steps: int = 20
    verbose: bool = False
    seed: Optional[int] = None
    
    # Reward configuration
    reward_scale: float = 1.0
    step_penalty: float = 0.0
    normalize_rewards: bool = True
    
    # Data configuration
    data_mode: str = "random"  # "random", "single", "list"
    data_source: Optional[Union[str, List[str]]] = None
    
    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    def validate(self):
        """Validate configuration parameters."""
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        if self.data_mode not in ["random", "single", "list"]:
            raise ValueError("data_mode must be 'random', 'single', or 'list'")
        return True

def get_default_config() -> YourGymConfig:
    return YourGymConfig()

def get_demo_config() -> YourGymConfig:
    return YourGymConfig(verbose=True, max_steps=15)
```

#### `yourgym/env/your_env.py`
```python
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import numpy as np
from ..config import YourGymConfig, get_default_config

class YourEnv(gym.Env):
    """Your domain-specific environment."""
    
    def __init__(self, config: Optional[YourGymConfig] = None):
        super().__init__()
        
        self.config = config or get_default_config()
        self.config.validate()
        
        # Define spaces
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
        return 0.5  # Placeholder
    
    async def _evaluate_answer_async(self, answer: str) -> float:
        """Evaluate the answer asynchronously and return a score between 0 and 1."""
        # TODO: Implement your domain-specific async evaluation logic
        return 0.5  # Placeholder
    
    def _load_data(self) -> Dict[str, Any]:
        """Load data based on configuration."""
        # TODO: Implement your data loading logic
        return {"id": "template", "description": "Template", "goal": "Template goal"}
    
    def close(self):
        """Clean up the environment."""
        pass
```

#### `setup.py`
```python
from setuptools import setup, find_packages

setup(
    name="yourgym",
    version="1.0.0",
    description="A Gymnasium environment for [your domain description]",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gymnasium",
        "numpy",
        "openai",
        "pyyaml",
    ],
    include_package_data=True,
)
```

## Implementation Guidelines

### 1. Action Format Standards

All gyms must support these standard action prefixes:

- **`[action]`**: For questions, requests, or general actions
- **`[answer]`**: For providing solutions, explanations, or final responses
- **`[finish]`**: For terminating episodes
- **Domain-specific prefixes**: Add as needed (e.g., `[search]`, `[recommend]`)

### 2. Async Support

**All new gyms must implement async support:**

- Implement both `step()` and `step_async()` methods
- Implement both sync and async versions of action handlers
- Use async for LLM calls and other I/O operations

### 3. Reward System Design

- **Delta-based rewards**: Reward improvement over previous attempts
- **Step penalties**: Encourage efficiency
- **Success thresholds**: Clear criteria for episode completion
- **Normalization**: Optional reward normalization to [0,1]

### 4. Configuration Best Practices

- Use dataclasses for configuration
- Provide sensible defaults
- Support environment variable overrides
- Include validation methods
- Provide pre-built configurations

## Testing Requirements

### Required Tests

1. **Basic Functionality**: Environment creation, reset, step
2. **Action Processing**: All action types work correctly
3. **Async Support**: Async methods work correctly
4. **Configuration**: Different configs produce expected behavior
5. **Edge Cases**: Invalid inputs, API failures

### Test Template

```python
import yourgym
from yourgym import YourEnv, get_default_config
import asyncio

def test_basic_functionality():
    """Test basic environment operations."""
    config = get_default_config()
    env = YourEnv(config)
    
    obs, info = env.reset()
    assert "description" in obs
    
    obs, reward, terminated, truncated, info = env.step("[action] test")
    assert isinstance(reward, float)
    
    env.close()

def test_async_functionality():
    """Test async functionality."""
    async def async_test():
        config = get_default_config()
        env = YourEnv(config)
        obs, info = await env.reset()
        
        obs, reward, terminated, truncated, info = await env.step_async("[action] test")
        assert isinstance(reward, float)
        
        env.close()
    
    asyncio.run(async_test())

if __name__ == "__main__":
    test_basic_functionality()
    test_async_functionality()
    print("All tests passed!")
```

## Template Gym

A complete template gym is available in `TemplateGym/`. This template includes:

- All required files with placeholder implementations
- Both sync and async implementations
- Basic test suite
- Simple documentation

Use this template as a starting point for your gym implementation.

## Submission Process

### Pre-submission Checklist

- [ ] All required files present and properly structured
- [ ] Both sync and async step methods implemented
- [ ] README.md follows the established format
- [ ] setup.py is minimal and correct
- [ ] Basic tests pass
- [ ] Documentation is comprehensive and accurate

### Submission Steps

1. **Fork the repository**
2. **Create your gym directory** following the naming convention: `YourDomainGym`
3. **Implement all required components**
4. **Add basic tests**
5. **Submit a pull request** with:
   - Clear description of your gym's purpose
   - Link to your gym's README
   - Test results

### Review Process

Your submission will be reviewed for:

- **Code Quality**: Clean, well-documented, follows patterns
- **Functionality**: Works as described, handles edge cases
- **Async Support**: Proper async implementation
- **Documentation**: Clear, comprehensive, follows format
- **Testing**: Basic test coverage

## Support

If you have questions about contributing a gym:

1. **Check existing gyms** for similar patterns
2. **Use the template gym** as a reference
3. **Open an issue** for specific questions

Remember: The goal is to create consistent, high-quality gyms that enable meaningful research while maintaining established patterns and standards.
