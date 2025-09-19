# IntentionGym

A Gymnasium-compatible environment for intention guessing simulation through reinforcement learning. This gym provides a standardized interface for agents to clarify vague user tasks by asking targeted questions to uncover missing details through multi-round conversations, enabling research in conversational AI and task clarification.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Intention Guessing**: Agents learn to clarify vague tasks through strategic questioning
- **Multi-round Conversations**: Supports interactive dialogue to uncover missing details
- **Importance-based Scoring**: Higher rewards for uncovering high-priority details
- **Configurable**: Flexible configuration system for different conversation scenarios
- **Logging**: Verbose mode for debugging and detailed monitoring

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Quick Start

Please first set up your OPENAI_API_KEY as environment variable.

```python
import intentiongym
from intentiongym import IntentionEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = IntentionEnv(config)

# Reset to get initial task
observation, info = env.reset()
print(f"Task: {observation['task_description']}")
print(f"Goal: {observation['goal']}")
print(f"Missing details: {observation['total_missing_details']}")

# Ask a clarifying question
obs, reward, terminated, truncated, info = env.step("[action] How long do you need this to take?")
print(f"User response: {obs['feedback']}")
print(f"Coverage: {obs['coverage_ratio']:.1%}")
print(f"Reward: {reward:.3f}")

# Finish episode
obs, reward, terminated, truncated, info = env.step("[finish]")
print(f"Final reward: {reward}")

env.close()
```

## Action Format

The environment accepts actions in two specific formats:

### 1. Clarifying Questions: `[action] <question>`

Ask targeted questions to uncover missing details:

- **Example**: `[action] How long do you need this to take?` - Ask about time requirements
- **Example**: `[action] What's your budget for this project?` - Ask about budget constraints
- **Example**: `[action] When do you need this completed?` - Ask about deadlines
- **Purpose**: Uncover specific missing details through strategic questioning

### 2. Episode Termination: `[finish]`

End the current conversation:

- **Example**: `[finish]` - Terminate the episode
- **Purpose**: End the conversation when satisfied with detail coverage

## Configuration

### Basic Configuration

```python
from intentiongym import IntentionGymConfig

config = IntentionGymConfig(
    max_steps=20,
    verbose=True,
    data_mode="random",
    step_penalty=0.0,
    multi_detail_penalty=0.2,
    normalize_rewards=True
)

env = IntentionEnv(config)
```

### Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `max_steps` | Maximum questions per episode | `20` | Any positive integer |
| `verbose` | Enable verbose logging | `False` | `True`/`False` |
| `data_mode` | Task selection mode | `"random"` | `"random"`, `"single"`, `"list"` |
| `data_source` | Specific task ID(s) to use | `None` | Task ID string or list |
| `step_penalty` | Penalty per question asked | `0.0` | Any float |
| `multi_detail_penalty` | Penalty for unfocused questions | `0.2` | Any float |
| `normalize_rewards` | Normalize rewards to [0,1] | `True` | `True`/`False` |

## Gymnasium Registration

The environment is automatically registered with Gymnasium upon import:

```python
import gymnasium as gym
import intentiongym

# Use the registered environment
env = gym.make('IntentionGym-v0')
```

## Data Flow

The environment follows a standard reinforcement learning cycle:

1. **Reset**: Initializes a new vague task and returns the initial observation
2. **Step**: Processes the agent's question, simulates user response, and returns feedback
3. **Reward Calculation**: Assigns rewards based on detail importance (1.0 for high, 0.7 for medium, 0.4 for low)
4. **Termination**: Episode ends when all details are uncovered or `[finish]` is called
5. **Truncation**: Episode ends when `max_steps` is reached without complete coverage

## Environment Behavior

### Intention Guessing Process

1. **Vague Task**: Each episode contains a user task with missing details
2. **Strategic Questioning**: Agents ask targeted questions to uncover specific missing details
3. **Importance-based Rewards**: Higher rewards for uncovering high-priority details
4. **Multi-detail Penalty**: Penalty for asking questions that cover multiple details at once
5. **Success Condition**: Episode succeeds when all missing details are uncovered

### Reward System

- **High importance (3)**: **1.0** reward - Critical details that significantly impact the task
- **Medium importance (2)**: **0.7** reward - Important but not critical details
- **Low importance (1)**: **0.4** reward - Nice-to-have details for completeness
- **Multi-detail penalty**: **-0.2** per extra detail covered in one question
- **Step penalty**: **-0.01** per question asked (configurable)

## Example Actions

### Complete Action Examples

```python
# Time-related questions
env.step("[action] How long do you need this to take?")
env.step("[action] When do you need this completed?")
env.step("[action] What's your preferred timeline?")

# Budget and resource questions
env.step("[action] What's your budget for this project?")
env.step("[action] Do you have any budget constraints?")
env.step("[action] What resources are available?")

# Experience and skill questions
env.step("[action] What's your experience level with this?")
env.step("[action] Do you have any relevant skills?")
env.step("[action] Have you done this before?")

# Specific requirement questions
env.step("[action] What specific features do you need?")
env.step("[action] Are there any must-have requirements?")
env.step("[action] What's your preferred approach?")

# End conversation
env.step("[finish]")
```

### Effective Questioning Strategy

```python
# Target high-importance details first
env.step("[action] What's your budget for this project?")  # High importance
env.step("[action] When do you need this completed?")      # High importance
env.step("[action] What's your experience level?")         # Medium importance
env.step("[action] Any specific preferences?")             # Low importance
env.step("[finish]")                                       # End when satisfied
```

