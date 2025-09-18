# FunctionGym

A Gymnasium-compatible environment for mathematical function learning through reinforcement learning. This gym provides a standardized interface for agents to discover hidden mathematical rules by testing different number combinations and receiving calculation feedback, enabling research in symbolic reasoning and mathematical pattern recognition.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Function Learning**: Agents discover hidden mathematical rules through systematic experimentation
- **Multiple Action Types**: Supports calculation testing, test case retrieval, answer submission, and episode termination
- **Configurable**: Flexible configuration system for different learning scenarios and function types
- **Clean API**: Simple, intuitive interface following Gymnasium standards
- **Logging**: Verbose mode for debugging and detailed monitoring

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Quick Start

Please first set up your OPENAI_API_KEY as environment variable.

```python
import functiongym
from functiongym import FunctionEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = FunctionEnv(config)

# Reset to get initial task
observation, info = env.reset()
print(f"Initial feedback: {observation['feedback']}")

# Test some numbers to discover the function
obs, reward, terminated, truncated, info = env.step("[action] 1 2 3 4")
print(f"Calculation result: {obs['feedback']}")

# Search for test case
obs, reward, terminated, truncated, info = env.step("[search] test")
print(f"Test case: {obs['feedback']}")

# Submit answer for test case
obs, reward, terminated, truncated, info = env.step("[answer] 7.5")
print(f"Answer result: {obs['feedback']}")

env.close()
```

## Action Format

The environment accepts actions in four specific formats:

### 1. Function Testing: `[action] a b c d`

Test the hidden function with four numbers to discover the mathematical rule:

- **Example**: `[action] 1 2 3 4` - Tests the function with numbers 1, 2, 3, 4
- **Example**: `[action] 5 10 2 3` - Tests with different number combinations
- **Purpose**: Discover the underlying mathematical pattern through experimentation

### 2. Test Case Retrieval: `[search] test`

Retrieve the test case numbers for which you need to predict the result:

- **Example**: `[search] test` - Returns the test case numbers
- **Purpose**: Get the specific numbers for which you must provide the final answer

### 3. Answer Submission: `[answer] value`

Submit your predicted answer for the test case:

- **Example**: `[answer] 7.5` - Submit 7.5 as your prediction
- **Example**: `[answer] 12` - Submit 12 as your prediction
- **Purpose**: Provide your final answer based on the discovered function

### 4. Episode Termination: `[finish]`

End the current episode:

- **Example**: `[finish]` - Terminate the episode
- **Purpose**: End the episode when you're done testing or want to give up

## Configuration

### Basic Configuration

```python
from functiongym import FunctionGymConfig

config = FunctionGymConfig(
    max_steps=20,
    verbose=True,
    correct_answer_reward=1.0,
    step_penalty=0.0,
    data_mode="single",
    data_source="func-1"
)

env = FunctionEnv(config)
```

### Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `max_steps` | Maximum steps per episode | `20` | Any positive integer |
| `verbose` | Enable verbose logging | `False` | `True`/`False` |
| `seed` | Random seed for reproducibility | `42` | Any integer or `None` |
| `correct_answer_reward` | Reward for correct final answer | `1.0` | Any float |
| `incorrect_answer_reward` | Reward for incorrect final answer | `0.0` | Any float |
| `step_penalty` | Penalty per step taken | `0.0` | Any float |
| `normalize_rewards` | Normalize rewards to [0,1] | `False` | `True`/`False` |
| `data_mode` | Function selection mode | `"single"` | `"random"`, `"single"`, `"list"` |
| `data_source` | Specific function ID(s) to use | `None` | Function ID string or list |

## Gymnasium Registration

The environment is automatically registered with Gymnasium upon import:

```python
import gymnasium as gym
import functiongym

# Use the registered environment
env = gym.make('FunctionGym-v0')
```

## Data Flow

The environment follows a standard reinforcement learning cycle:

1. **Reset**: Initializes a new mathematical function and returns the initial observation
2. **Step**: Processes the agent's action, performs calculations or evaluations, and returns the response
3. **Reward Calculation**: Assigns rewards based on final answer correctness (1.0 for correct, 0.0 for incorrect)
4. **Termination**: Episode ends when correct answer is submitted or `[finish]` is called
5. **Truncation**: Episode ends when `max_steps` is reached without correct answer

## Environment Behavior

### Function Testing Process

1. **Hidden Function**: Each episode contains a hidden mathematical function (e.g., `a * (b + c) / d`)
2. **Calculation Testing**: When you provide four numbers via `[action]`, the environment calculates the result using the hidden rule
3. **Test Case Protection**: If you try to test the exact test case numbers, the environment refuses to give feedback
4. **Answer Evaluation**: Your final answer is compared against the expected result for the test case
5. **Reward System**: You receive rewards only for correct final answers, not for intermediate calculations

### Sample Functions

The environment includes various mathematical functions such as:

- **Arithmetic Operations**: `a * (b + c) / d`, `a + b * c - d`
- **Parentheses Grouping**: `(a + b) * (c - d)`, `(a - b) + (c * d)`
- **Exponentiation**: `a ** b + c * d`, `(a + b) ** c - d`
- **Complex Expressions**: `a * b + c / d`, `(a + b) * c - d / 2`

## Example Actions

### Complete Action Examples

```python
# Function testing with different number combinations
env.step("[action] 1 2 3 4")
env.step("[action] 5 10 2 3")
env.step("[action] 2 4 6 8")
env.step("[action] 1 1 1 1")

# Retrieve test case numbers
env.step("[search] test")

# Submit predicted answers
env.step("[answer] 7.5")
env.step("[answer] 12")
env.step("[answer] 3.14")

# End episode
env.step("[finish]")
```

### Learning Strategy Example

```python
# Typical learning sequence
env.step("[action] 1 1 1 1")  # Test with simple numbers
env.step("[action] 2 2 2 2")  # Test with doubled numbers
env.step("[action] 1 2 3 4")  # Test with sequential numbers
env.step("[search] test")     # Get test case
env.step("[answer] 8.5")      # Submit prediction
```
