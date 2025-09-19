# TauGym

A Gymnasium-compatible environment for tau-bench style tool-agent-user interactions through reinforcement learning. This gym provides a standardized interface for agents to interact with simulated users using domain-specific tools, mimicking real-world customer support scenarios and enabling research in conversational AI and tool usage.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Tau-Bench Integration**: Uses tau-bench backend for realistic customer support scenarios
- **Tool-Agent-User Interactions**: Supports complex multi-party conversations with tool usage
- **Domain-Specific Tools**: Provides access to retail and airline customer support tools
- **Configurable**: Flexible configuration system for different interaction scenarios
- **Logging**: Verbose mode for debugging and detailed monitoring

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Prerequisites

Before using TauGym, you need to install tau-bench from source:

```bash
git clone https://github.com/sierra-research/tau-bench.git
cd tau-bench
pip install -e .
```

## Quick Start

Please first set up your OPENAI_API_KEY as environment variable.

```python
import taugym
from taugym import TauEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = TauEnv(config)

# Reset to get initial task
observation, info = env.reset()
print(f"Task: {observation['instruction']}")
print(f"User: {observation['feedback']}")

# Search for available tools
obs, reward, terminated, truncated, info = env.step("[search] tools")
print(f"Tools info: {obs['feedback']}")

# Communicate with user
obs, reward, terminated, truncated, info = env.step("[action] Hello! How can I help you today?")
print(f"User response: {obs['feedback']}")

# Use a tool
obs, reward, terminated, truncated, info = env.step("[answer] {\"name\": \"get_user_details\", \"arguments\": {\"user_id\": \"user_123\"}}")
print(f"Tool result: {obs['feedback']}")
print(f"Reward: {reward:.3f}")

# Finish episode
obs, reward, terminated, truncated, info = env.step("[finish]")
print(f"Final reward: {reward}")

env.close()
```

## Action Format

The environment accepts actions in four specific formats:

### 1. Tool Search: `[search] <query>`

Get information about available tools or help:

- **Example**: `[search] tools` - Show all available tools
- **Example**: `[search] help` - Show help information
- **Purpose**: Discover available tools and their usage

### 2. User Communication: `[action] <message>`

Direct communication with the simulated user:

- **Example**: `[action] Hello! How can I help you today?` - Greet the user
- **Example**: `[action] I can help you with your order` - Respond to user needs
- **Purpose**: Engage in conversation with the simulated user

### 3. Tool Usage: `[answer] <tool_call>`

Execute tool calls to perform actions:

- **Example**: `[answer] {"name": "get_user_details", "arguments": {"user_id": "user_123"}}` - JSON format
- **Example**: `[answer] get_order_details {"order_id": "ORD001"}` - Simple format
- **Purpose**: Use domain-specific tools to help the user

### 4. Episode Termination: `[finish]`

End the current conversation:

- **Example**: `[finish]` - Terminate the episode
- **Purpose**: End the conversation when task is completed

## Configuration

### Basic Configuration

```python
from taugym import TauGymConfig

config = TauGymConfig(
    task_category="retail",
    data_source="retail-019",
    user_strategy="llm",
    user_model="gpt-4o",
    user_provider="openai",
    max_steps=30,
    verbose=True,
    normalize_rewards=False
)

env = TauEnv(config)
```

### Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `task_category` | Domain for tasks | `"retail"` | `"retail"`, `"airline"` |
| `data_source` | Specific task ID | `None` | Task ID string (e.g., "retail-019") |
| `data_mode` | Task selection mode | `"single"` | `"single"` (only mode supported) |
| `user_strategy` | User simulation strategy | `"llm"` | `"llm"` |
| `user_model` | User simulation model | `"gpt-4o"` | Any supported model |
| `user_provider` | User simulation provider | `"openai"` | `"openai"`, `"anthropic"`, etc. |
| `max_steps` | Maximum actions per episode | `30` | Any positive integer |
| `verbose` | Enable verbose logging | `False` | `True`/`False` |
| `normalize_rewards` | Normalize rewards to [0,1] | `False` | `True`/`False` |

## Gymnasium Registration

The environment is automatically registered with Gymnasium upon import:

```python
import gymnasium as gym
import taugym

# Use the registered environment
env = gym.make('TauGym-v0')
```

## Data Flow

The environment follows a standard reinforcement learning cycle:

1. **Reset**: Initializes a new tau-bench task and returns the initial observation
2. **Step**: Processes the agent's action, forwards to tau-bench, and returns feedback
3. **Reward Calculation**: Uses tau-bench's native reward system for task completion
4. **Termination**: Episode ends when task is completed or `[finish]` is called
5. **Truncation**: Episode ends when `max_steps` is reached without completion

## Environment Behavior

### Tool-Agent-User Interaction Process

1. **Task Initialization**: Each episode contains a customer support scenario from tau-bench
2. **User Simulation**: Tau-bench provides LLM-based user simulation
3. **Tool Access**: Agents can use domain-specific tools (retail/airline)
4. **Multi-turn Conversations**: Supports complex back-and-forth interactions
5. **Success Condition**: Episode succeeds when the user's needs are met

### Available Domains

**Retail Domain** (`task_category="retail"`):
- Customer support scenarios
- Order management, returns, exchanges
- User account operations
- Product inquiries
- Task IDs: "retail-000" to "retail-114" (115 tasks)

**Airline Domain** (`task_category="airline"`):
- Flight booking and management
- Reservation changes
- Customer service scenarios
- Task IDs: "airline-000" to "airline-049" (50 tasks)

### Reward System

- **Task Completion**: Rewards based on tau-bench's native evaluation
- **Tool Usage**: Rewards for appropriate tool usage
- **User Satisfaction**: Rewards for successful user interactions
- **Step Penalty**: Configurable penalty per action taken

## Example Actions

### Complete Action Examples

```python
# Search for tools and help
env.step("[search] tools")
env.step("[search] help")
env.step("[search] available functions")

# Communicate with user
env.step("[action] Hello! How can I help you today?")
env.step("[action] I can help you with your order")
env.step("[action] Let me check that for you")

# Use retail tools
env.step("[answer] {\"name\": \"get_user_details\", \"arguments\": {\"user_id\": \"john_123\"}}")
env.step("[answer] {\"name\": \"get_order_details\", \"arguments\": {\"order_id\": \"ORD001\"}}")
env.step("[answer] {\"name\": \"process_return\", \"arguments\": {\"order_id\": \"ORD001\", \"items\": [\"item1\"]}}")

# Use airline tools
env.step("[answer] {\"name\": \"search_flights\", \"arguments\": {\"origin\": \"NYC\", \"destination\": \"LAX\"}}")
env.step("[answer] {\"name\": \"book_flight\", \"arguments\": {\"flight_id\": \"FL123\", \"passenger_id\": \"P456\"}}")

# End conversation
env.step("[finish]")
```

### Effective Interaction Strategy

```python
# Start with greeting and tool discovery
env.step("[search] tools")
env.step("[action] Hello! I'm here to help you with your request")

# Gather information using tools
env.step("[answer] {\"name\": \"get_user_details\", \"arguments\": {\"user_id\": \"user_123\"}}")

# Respond based on tool results
env.step("[action] I can see your order details. Let me help you with that")

# Complete the task
env.step("[finish]")
```

## API Requirements

### Required Environment Variables

- **`OPENAI_API_KEY`**: Required for user simulation using LLM

### Getting API Keys

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

### Setup Example

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Tau-Bench Integration

TauGym acts as a proxy between your RL agent and tau-bench:

```
[RL Agent] <-> [TauGym] <-> [Tau-Bench] <-> [User Simulation]
     ^             ^              ^               ^
 Gym Actions   Translation    Tool Calls      LLM User
```

**Key Features:**
- **Action Translation**: Converts gym actions to tau-bench format
- **Tool Parsing**: Parses `[answer]` actions into tau-bench tool calls
- **Search Handling**: Provides tool information for `[search]` actions
- **Reward Passthrough**: Directly uses tau-bench rewards
- **Async Support**: Provides `step_async()` for compatibility

