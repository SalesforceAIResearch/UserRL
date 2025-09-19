# UserRL Evaluation Pipeline

A comprehensive evaluation framework for training and evaluating language models on diverse reinforcement learning environments. This pipeline supports both trained models (requiring model merging and hosting) and API-based models (direct evaluation).

## Overview

The evaluation pipeline consists of three main stages:

1. **Model Preparation** (for trained models only)
   - **Merge**: Convert trained model checkpoints to HuggingFace format
   - **Host**: Deploy model using vLLM for API access

2. **Evaluation**: Run comprehensive evaluation across multiple environments
3. **Analysis**: Process results and generate performance reports

## Pipeline Architecture

### For Trained Models
```
Trained Model → Merge → Host → Evaluate → Analyze
```

### For API Models  
```
API Model → Evaluate → Analyze
```

## Directory Structure

```
UserRL/eval/
├── eval.py                 # Core evaluation script
├── merge.py                # Model merging script (FSDP/Megatron → HuggingFace)
├── analyze.py              # Results analysis script
├── eval.sh                 # Evaluation shell script
├── merge.sh                # Model merging shell script
├── host.sh                 # Model hosting shell script
└── schema/
    └── interact_tool.yaml  # Tool calling schema definition
```

## Supported Environments

The pipeline evaluates models across 8 diverse environments:

| Environment | Description | Key Features |
|-------------|-------------|--------------|
| **TravelGym** | Travel planning preference elicitation | Multi-aspect planning, function calls |
| **TurtleGym** | Turtle Soup lateral thinking puzzles | Creative reasoning, puzzle solving |
| **FunctionGym** | Mathematical function learning | Function discovery, parameter learning |
| **TauGym** | Tool-agent-user interactions | Multi-agent coordination, tool usage |
| **PersuadeGym** | AI persuasion simulation | Argumentation, social influence |
| **IntentionGym** | AI intention guessing | Preference inference, conversation |
| **TelepathyGym** | Mind reading games | Strategic questioning, logical reasoning |
| **SearchGym** | Search-based question answering | Web search, information synthesis |

## Usage Instructions

### 1. Model Preparation (Trained Models Only)

#### Step 1: Merge Model Checkpoints
```bash
cd /fsx/home/cqian/projects/UserRL/eval
./merge.sh
```

**Manual merge command:**
```bash
python merge.py merge \
    --backend fsdp \
    --local_dir ${MODEL_PATH}/actor \
    --target_dir ${MODEL_PATH}_hf
```

This script converts trained model checkpoints to HuggingFace format:
- **Input**: `/path/to/your/model/global_step_XXX/actor`
- **Output**: `/path/to/your/model/global_step_XXX_hf`
- **Supported backends**: FSDP, Megatron

#### Step 2: Host Model
```bash
cd /fsx/home/cqian/projects/UserRL/eval
./host.sh
```

**Manual host command:**
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MERGED_MODEL_PATH="path/to/your/model/global_step_XXX_hf"
export CUSTOMIZED_SERVED_MODEL_NAME="your_model_name"

vllm serve ${MERGED_MODEL_PATH} \
   --max-model-len 32768 \
   --port 8500 \
   --gpu-memory-utilization 0.8 \
   --tensor-parallel-size 8 \
   --enable-auto-tool-choice \
   --tool-call-parser hermes \
   --reasoning-parser qwen3 \
   --served-model-name ${CUSTOMIZED_SERVED_MODEL_NAME}
```

Deploy the merged model using vLLM with:
- Tensor parallelism across multiple GPUs
- Tool calling support with Hermes parser
- Reasoning support for Qwen3 models
- Auto tool choice enabled

### 2. Evaluation

#### For API Models
```bash
cd /fsx/home/cqian/projects/UserRL/eval
./eval.sh
```

**Manual evaluation command:**
```bash
python eval.py \
    --model_name gpt-4o \
    --max_turns 16 \
    --pass_k 1 \
    --temperature 0 \
    --envs travel22 travel33 travel44 travel233 travel333 travel334 travel444 travel2222 bamboogle function intention persuasion tau telepathy turtle \
    --save_name outputs/results_gpt-4o
```

#### For Self-hosted Models
```bash
python eval.py \
    --model_name ${CUSTOMIZED_SERVED_MODEL_NAME} \
    --port 8500 \
    --max_turns 16 \
    --pass_k 1 \
    --temperature 0 \
    --envs travel22 travel33 travel44 travel233 travel333 travel334 travel444 travel2222 bamboogle function intention persuasion tau telepathy turtle \
    --save_name outputs/results_${CUSTOMIZED_SERVED_MODEL_NAME}
```

### 3. Analysis

#### Process Results
```bash
cd /fsx/home/cqian/projects/UserRL/eval
python analyze.py
```

This generates:
- Weighted average reward ranking across all environments
- Task-specific performance breakdown
- Comparative analysis between models

## Configuration

### Environment Variables
```bash
# Required for evaluation
export MAX_WORKER_NUM=10                    # Concurrent evaluation workers
export USER_MODEL_NAME="gpt-4o"            # Model name for user simulation
export OPENAI_API_KEY="your_openai_key"    # OpenAI API key
export OPENAI_BASE_URL="https://api.openai.com/v1"  # API base URL
export TOOL_CHOICE="auto"                  # Tool choice strategy

# Optional API keys
export GENAI_API_KEY="your_gemini_key"     # Google Gemini API key
export SERPER_API_KEY="your_serper_key"    # Serper API key (for SearchGym)
```

### Evaluation Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model_name` | Model identifier | Required | Model name or API endpoint |
| `--port` | Local model port | 8000 | Any available port |
| `--max_turns` | Maximum steps per episode | 16 | Positive integer |
| `--pass_k` | Number of evaluation passes | 1 | Positive integer |
| `--temperature` | Sampling temperature | 0.0 | 0.0-2.0 |
| `--envs` | Environments to evaluate | All | Space-separated list |
| `--save_name` | Output file prefix | "results" | String |

### Supported Models

#### API Models
- **OpenAI**: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Google**: `gemini-2.0-flash`, `gemini-1.5-pro`
- **Custom**: Any OpenAI-compatible API endpoint

#### Local Models
- **Qwen**: Qwen2.5, Qwen3 series
- **Llama**: Llama-3.1, Llama-3.3 series
- **Custom**: Any HuggingFace-compatible model via vLLM

## Evaluation Process

### 1. Data Loading
- Loads evaluation datasets from `UserRL/data`
- Supports both multiturn and one-choice variants
- Formats data for environment initialization

### 2. Environment Setup
- Initializes each environment with specific configuration
- Sets up model parameters and evaluation settings
- Configures reward structures and success criteria

### 3. Rollout Execution
- Runs asynchronous rollouts across multiple workers
- Implements tool calling with proper error handling
- Supports different model APIs (OpenAI, Google)

### 4. Result Processing
- Calculates performance metrics per environment
- Aggregates results across multiple passes
- Generates comprehensive performance reports

## Tool Calling Schema

The evaluation uses a standardized tool calling schema defined in `schema/interact_tool.yaml`:

```yaml
tool_schema:
  type: "function"
  function:
    name: "interact_with_env"
    description: "A tool for interact with a target environment..."
    parameters:
      type: "object"
      properties:
        choice:
          type: "string"
          enum: ["action", "answer", "search"]
          description: "Your choice of what to do next..."
        content:
          type: "string"
          description: "The content of your choice..."
      required: ["choice", "content"]
```

### Action Types
- **`action`**: General actions (questions, moves, etc.)
- **`answer`**: Final answers or submissions
- **`search`**: Search queries (for SearchGym)

## Performance Metrics

### TravelGym Metrics
- **Micro Average**: Average reward across all episodes
- **Micro Maximum**: Maximum reward achieved
- **Best Choice Rate**: Percentage of optimal recommendations (reward = 1.0)
- **Correct Choice Rate**: Percentage of acceptable recommendations (reward ≥ 0.8)

### Other Environments
- **Micro Average**: Average episode reward
- **Micro Maximum**: Maximum episode reward
- **Success Rate**: Percentage of successful episodes

### Analysis Features
- **Weighted Ranking**: Performance ranking across all environments
- **Task-specific Breakdown**: Individual environment performance
- **Pass-k Analysis**: Multi-pass evaluation results

## Output Files

### Evaluation Results
- `{save_name}_results.json`: Aggregated performance metrics
- `{save_name}_reward_cache.json`: Detailed rollout data

### Analysis Outputs
- Performance ranking tables
- Weighted average scores
- Task-specific performance breakdown

## Environment-Specific Configurations

### TravelGym
```python
config.max_steps = 16
config.one_choice_per_aspect = True
config.search_correct_reward = 0.2
config.preference_correct_reward = 0.6
```

### TurtleGym
```python
config.max_steps = 16
config.success_threshold = 1.0
config.data_mode = "single"
```

### FunctionGym
```python
config.max_steps = 16
config.data_mode = "single"
```

### TauGym
```python
config.max_steps = 16
config.data_mode = "single"
# Automatic task category detection (retail/airline)
# Automatic task split detection (train/test)
```

## Model Merging Details

### FSDP Backend
- Merges distributed FSDP checkpoints
- Supports tensor parallelism
- Handles DTensor placements
- Automatic world size detection

### Megatron Backend
- Merges Megatron-LM checkpoints
- Supports tensor and pipeline parallelism
- Handles QKV splitting and gate/up projections
- Automatic TP/PP size detection

### Supported Operations
- **Merge**: Convert checkpoints to HuggingFace format
- **Test**: Validate merged checkpoints against reference models

## Troubleshooting

### Common Issues

1. **Model Hosting Failures**
   - Check GPU memory availability
   - Verify model path and format
   - Ensure vLLM compatibility

2. **API Rate Limits**
   - Adjust `MAX_WORKER_NUM`
   - Implement retry logic
   - Use multiple API keys

3. **Environment Errors**
   - Verify environment installation
   - Check data file paths
   - Validate configuration parameters

4. **Tool Calling Errors**
   - Verify model supports function calling
   - Check tool schema compatibility
   - Ensure proper action formatting

### Performance Optimization

1. **Concurrent Evaluation**
   - Increase `MAX_WORKER_NUM` for faster evaluation
   - Use multiple GPUs for model hosting
   - Implement result caching

2. **Memory Management**
   - Monitor GPU memory usage
   - Use gradient checkpointing
   - Implement batch processing

## Contributing

### Adding New Environments
1. Implement environment following UserRL standards
2. Add environment configuration to `eval.py`
3. Update evaluation scripts
4. Add analysis scripts

### Adding New Models
1. Implement model API client
2. Add model configuration
3. Update evaluation scripts
4. Test with multiple environments

## Citation

If you use this evaluation pipeline in your research, please cite:

```bibtex
@misc{userrl-eval,
  title={UserRL Evaluation Pipeline: Comprehensive Evaluation Framework for Language Models in Reinforcement Learning Environments},
  author={UserRL Team},
  year={2024},
  url={https://github.com/your-org/UserRL}
}
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

---

**For questions and support, please open an issue or contact the development team.**
