# SearchGym

A Gymnasium-compatible environment for search-based question answering through reinforcement learning. This gym provides a standardized interface for agents to perform web searches using the Serper API and answer questions based on search results, enabling research in information retrieval and question answering systems.

## Features

- **Gymnasium Compatible**: Fully compliant with the Gymnasium environment interface
- **Web Search Integration**: Uses Serper API for real-time web search capabilities
- **Question Answering**: Agents learn to search and answer questions based on results
- **Structured Search Results**: Receives organized search data for better decision making
- **Configurable**: Flexible configuration system for different search scenarios
- **Logging**: Verbose mode for debugging and detailed monitoring

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Quick Start

Please first set up your OPENAI_API_KEY and SERPER_API_KEY as environment variables.

```python
import searchgym
from searchgym import SearchEnv, get_default_config

# Create environment with default configuration
config = get_default_config()
config.verbose = True  # Enable detailed logging
env = SearchEnv(config)

# Reset to get initial question
observation, info = env.reset()
print(f"Question: {observation['question']}")

# Perform web search
obs, reward, terminated, truncated, info = env.step("[search] What is the capital of France?")
print(f"Search results: {obs['feedback']}")

# Submit answer
obs, reward, terminated, truncated, info = env.step("[answer] Paris")
print(f"Answer evaluation: {obs['feedback']}")
print(f"Reward: {reward:.3f}")

# Finish episode
obs, reward, terminated, truncated, info = env.step("[finish]")
print(f"Final reward: {reward}")

env.close()
```

## Action Format

The environment accepts actions in three specific formats:

### 1. Web Search: `[search] <query>`

Perform web searches using the Serper API:

- **Example**: `[search] What is the capital of France?` - Search for information about France's capital
- **Example**: `[search] latest news about artificial intelligence` - Search for recent AI news
- **Example**: `[search] population of Tokyo Japan` - Search for demographic information
- **Purpose**: Gather information from the web to help answer questions

### 2. Answer Submission: `[answer] <answer>`

Submit answers based on search results:

- **Example**: `[answer] Paris` - Submit Paris as the answer
- **Example**: `[answer] The capital of France is Paris` - Submit a complete answer
- **Purpose**: Provide the final answer based on gathered information

### 3. Episode Termination: `[finish]`

End the current episode:

- **Example**: `[finish]` - Terminate the episode
- **Purpose**: End the episode when done searching or want to give up

## Configuration

### Basic Configuration

```python
from searchgym import SearchGymConfig

config = SearchGymConfig(
    max_steps=20,
    verbose=True,
    data_mode="single",
    correct_answer_reward=1.0,
    step_penalty=0.0,
    max_search_results=5,
    max_search_steps=5,
    normalize_rewards=False
)

env = SearchEnv(config)
```

### Configuration Options

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `max_steps` | Maximum actions per episode | `20` | Any positive integer |
| `verbose` | Enable verbose logging | `False` | `True`/`False` |
| `data_mode` | Question selection mode | `"single"` | `"random"`, `"single"`, `"list"` |
| `data_source` | Specific question ID(s) to use | `None` | Question ID string or list |
| `correct_answer_reward` | Reward for correct answer | `1.0` | Any float |
| `incorrect_answer_reward` | Reward for incorrect answer | `0.0` | Any float |
| `step_penalty` | Penalty per action taken | `0.0` | Any float |
| `normalize_rewards` | Normalize rewards to [0,1] | `False` | `True`/`False` |
| `max_search_results` | Maximum search results per query | `5` | Any positive integer |
| `max_search_steps` | Maximum search actions per episode | `5` | Any positive integer |

## Gymnasium Registration

The environment is automatically registered with Gymnasium upon import:

```python
import gymnasium as gym
import searchgym

# Use the registered environment
env = gym.make('SearchGym-v0')
```

## Data Flow

The environment follows a standard reinforcement learning cycle:

1. **Reset**: Initializes a new question and returns the initial observation
2. **Step**: Processes the agent's action (search or answer), performs web search or evaluation, and returns feedback
3. **Reward Calculation**: Assigns rewards based on answer correctness (1.0 for correct, 0.0 for incorrect)
4. **Termination**: Episode ends when correct answer is submitted or `[finish]` is called
5. **Truncation**: Episode ends when `max_steps` is reached without correct answer

## Environment Behavior

### Search-Based Question Answering Process

1. **Question Presentation**: Each episode contains a question that requires web search to answer
2. **Search Actions**: Agents perform web searches using the Serper API to gather information
3. **Search Results**: Structured search results are returned with titles, snippets, and URLs
4. **Answer Evaluation**: Submitted answers are evaluated against ground truth using LLM-based evaluation
5. **Success Condition**: Episode succeeds when the correct answer is submitted

### Serper API Integration

- **Real-time Search**: Uses Serper API for live web search capabilities
- **Structured Results**: Returns organized search results with metadata
- **Search Limits**: Configurable limits on search results and search steps per episode
- **API Key Required**: Requires `SERPER_API_KEY` environment variable

### Reward System

- **Correct Answer**: **1.0** reward when the submitted answer is correct
- **Incorrect Answer**: **0.0** reward when the submitted answer is wrong
- **Search Actions**: **0.0** reward for search actions (information gathering)
- **Step Penalty**: Configurable penalty per action taken

## Example Actions

### Complete Action Examples

```python
# Search for information
env.step("[search] What is the capital of France?")
env.step("[search] France capital city")
env.step("[search] Paris France capital")

# Search for different types of information
env.step("[search] latest news about climate change")
env.step("[search] population of Tokyo Japan")
env.step("[search] who invented the telephone")

# Submit answers based on search results
env.step("[answer] Paris")
env.step("[answer] The capital of France is Paris")
env.step("[answer] Alexander Graham Bell")

# End episode
env.step("[finish]")
```

### Effective Search Strategy

```python
# Start with broad search queries
env.step("[search] What is the capital of France?")

# Refine search if needed
env.step("[search] France capital city Paris")

# Submit answer based on search results
env.step("[answer] Paris")

# Or end if no clear answer found
env.step("[finish]")
```

## API Requirements

### Required Environment Variables

- **`OPENAI_API_KEY`**: Required for answer evaluation using LLM
- **`SERPER_API_KEY`**: Required for web search functionality via Serper API

### Getting API Keys

1. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Serper API Key**: Get from [Serper.dev](https://serper.dev/) - provides Google search API access

### Setup Example

```bash
export OPENAI_API_KEY="your-openai-api-key"
export SERPER_API_KEY="your-serper-api-key"
```

