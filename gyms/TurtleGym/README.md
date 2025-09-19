# TurtleGym

A Gymnasium-compatible environment for Turtle Soup lateral thinking puzzle games through reinforcement learning. This gym provides a standardized interface for agents to solve mysterious story scenarios by asking questions and providing explanations, enabling research in lateral thinking and puzzle-solving strategies.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Turtle Soup Puzzles**: Classic lateral thinking puzzle game with mysterious scenarios
- **Question-Answer Format**: Agents ask questions to gather information and provide explanations
- **LLM Evaluation**: Uses OpenAI GPT models for intelligent puzzle evaluation
- **Story Collection**: 100+ diverse mysterious scenarios with evaluation criteria
- **Configurable**: Flexible configuration system for different puzzle scenarios
- **Logging**: Verbose mode for debugging and detailed monitoring

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Quick Start

Please first set up your OPENAI_API_KEY as environment variable.

```python
import turtlegym
from turtlegym import StoryEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = StoryEnv(config)

# Reset to get initial story scenario
observation, info = env.reset()
print(f"Story: {observation['description']}")

# Ask questions about the story
obs, reward, terminated, truncated, info = env.step("[action] Was someone pretending to be someone else?")
print(f"Response: {obs['feedback']}")

obs, reward, terminated, truncated, info = env.step("[action] Did someone die in this story?")
print(f"Response: {obs['feedback']}")

# Provide explanation
obs, reward, terminated, truncated, info = env.step("[answer] The protagonist was actually hiding during a home invasion...")
print(f"Score: {obs['current_score']}")
print(f"Final reward: {reward}")

env.close()
```

## Action Format

The environment accepts actions in three specific formats:

### 1. Questions: `[action] <question>`

Ask questions to gather information about the story:

- **Example**: `[action] Was someone pretending to be someone else?` - Ask about identity deception
- **Example**: `[action] Did someone die in this story?` - Ask about death or violence
- **Example**: `[action] Was there a supernatural element?` - Ask about supernatural aspects
- **Purpose**: Gather information to understand the mysterious scenario

### 2. Explanations: `[answer] <explanation>`

Provide your explanation of what really happened:

- **Example**: `[answer] The protagonist was actually hiding during a home invasion...` - Explain the true story
- **Example**: `[answer] The child was pretending to be invisible to avoid punishment...` - Explain the deception
- **Example**: `[answer] The person was sleepwalking and didn't remember their actions...` - Explain the mystery
- **Purpose**: Submit your solution to the puzzle

### 3. Episode Termination: `[finish]`

End the current episode:

- **Example**: `[finish]` - Terminate the episode
- **Purpose**: End the episode when puzzle solving is complete

## Configuration

### Basic Configuration

```python
from turtlegym import TurtleGymConfig

config = TurtleGymConfig(
    max_steps=20,
    success_threshold=0.9,
    verbose=True,
    data_mode="random",
    reward_scale=1.0,
    step_penalty=0.0,
    normalize_rewards=True
)

env = StoryEnv(config)
```

### Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `max_steps` | Maximum steps per episode | `20` | Any positive integer |
| `success_threshold` | Score needed for success | `0.9` | 0.0 to 1.0 |
| `verbose` | Enable verbose logging | `False` | `True`/`False` |
| `data_mode` | Story selection mode | `"random"` | `"random"`, `"single"`, `"list"` |
| `data_source` | Specific story title(s) to use | `None` | Story title string or list |
| `reward_scale` | Scale factor for rewards | `1.0` | Any positive float |
| `step_penalty` | Penalty per step taken | `0.0` | Any float |
| `normalize_rewards` | Normalize rewards to [0,1] | `True` | `True`/`False` |

## Gymnasium Registration

The environment is automatically registered with Gymnasium upon import:

```python
import gymnasium as gym
import turtlegym

# Use the registered environment
env = gym.make('TurtleSoup-v0')
```

## Data Flow

The environment follows a standard reinforcement learning cycle:

1. **Reset**: Initializes a new story scenario and returns the initial observation
2. **Step**: Processes the agent's action (question or explanation), evaluates with LLM, and returns feedback
3. **Reward Calculation**: Assigns rewards based on explanation quality and puzzle-solving progress
4. **Termination**: Episode ends when puzzle is solved or `[finish]` is called
5. **Truncation**: Episode ends when `max_steps` is reached without solving

## Environment Behavior

### Turtle Soup Puzzle Process

1. **Story Initialization**: Each episode contains a mysterious story scenario (the "surface")
2. **Information Gathering**: Agents ask questions to understand the situation
3. **Lateral Thinking**: Agents must think creatively to uncover hidden truths
4. **Explanation Phase**: Agents provide explanations of what really happened (the "bottom")
5. **Success Condition**: Episode succeeds when explanation meets success threshold

### Story Collection

The environment includes 100+ diverse mysterious scenarios:

- **Family Secrets**: Identity mix-ups, hidden relationships, family dynamics
- **Supernatural Elements**: Ghosts, time travel, magical occurrences
- **Psychological Twists**: Mental states, perception, memory issues
- **Everyday Mysteries**: Common situations with unexpected explanations
- **Crime and Deception**: Theft, fraud, hidden motives

### Reward System

- **Explanation Quality**: Rewards based on how well the explanation matches the ground truth
- **Score Progression**: Higher rewards for better explanations (delta-based scoring)
- **Success Threshold**: Default 0.9 score needed for episode success
- **Step Penalty**: Configurable penalty per step taken
- **Evaluation Criteria**: Each story has 2-4 specific evaluation criteria with weights

### LLM Evaluation Features

- **Intelligent Scoring**: GPT models evaluate explanation quality against ground truth
- **Multi-Criteria Assessment**: Stories evaluated on multiple dimensions
- **Weighted Scoring**: Different criteria have different importance weights
- **Progressive Improvement**: Rewards increase as explanations improve

## Example Actions

### Complete Action Examples

```python
# Ask about identity and deception
env.step("[action] Was someone pretending to be someone else?")
env.step("[action] Was there a case of mistaken identity?")
env.step("[action] Did someone change their appearance?")

# Ask about death and violence
env.step("[action] Did someone die in this story?")
env.step("[action] Was there any violence involved?")
env.step("[action] Did someone get hurt?")

# Ask about supernatural elements
env.step("[action] Was there a supernatural element?")
env.step("[action] Did something magical happen?")
env.step("[action] Was there a ghost or spirit involved?")

# Ask about psychological aspects
env.step("[action] Was someone sleepwalking?")
env.step("[action] Did someone have amnesia?")
env.step("[action] Was there a mental health issue?")

# Provide explanations
env.step("[answer] The protagonist was actually hiding during a home invasion...")
env.step("[answer] The child was pretending to be invisible to avoid punishment...")
env.step("[answer] The person was sleepwalking and didn't remember their actions...")

# End episode
env.step("[finish]")
```

### Effective Puzzle-Solving Strategy

```python
# Start with broad questions
env.step("[action] Was someone pretending to be someone else?")
env.step("[action] Did something supernatural happen?")

# Narrow down based on responses
env.step("[action] Was there a case of mistaken identity?")
env.step("[action] Did someone change their appearance?")

# Formulate explanation
env.step("[answer] The protagonist was actually hiding during a home invasion...")

# Or end if puzzle is solved
env.step("[finish]")
```

## API Requirements

### Required Environment Variables

- **`OPENAI_API_KEY`**: Required for LLM-based puzzle evaluation

### Getting API Keys

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

### Setup Example

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

