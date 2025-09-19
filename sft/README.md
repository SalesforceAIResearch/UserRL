# UserRL Supervised Fine-Tuning (SFT) Pipeline

This directory contains the supervised fine-tuning pipeline for training language models on UserRL gym environments using LLaMA-Factory.

## Overview

The SFT pipeline trains language models to interact with UserRL environments by fine-tuning on conversation data collected from successful interactions across all nine gym environments.

## Pipeline Components

### 1. Training Data (`merged_gym_sft.json`)
- **Size**: 25MB, ~121K lines
- **Format**: ShareGPT conversation format
- **Content**: Multi-turn conversations from successful interactions across all UserRL gyms
- **Structure**: 
  ```json
  {
    "conversations": [
      {"from": "human", "value": "Environment prompt..."},
      {"from": "gpt", "value": "Model response with tool calls..."},
      {"from": "observation", "value": "Environment feedback..."}
    ],
    "system": "System prompt for the environment",
    "tools": "Tool schema definitions"
  }
  ```

### 2. Training Configuration (`qwen3_customized.yaml`)
- **Model**: Qwen3 instruction-tuned models
- **Training**: Full fine-tuning with DeepSpeed ZeRO-3
- **Sequence Length**: 16,384 tokens
- **Batch Size**: 2 per device × 4 gradient accumulation = 8 effective batch size
- **Learning Rate**: 1e-5 with cosine scheduling
- **Epochs**: 3 training epochs

### 3. Environment Integration
- **Tool Calling**: Supports `interact_with_env` function calls
- **Multi-Environment**: Training data spans all nine UserRL gyms
- **Conversation Format**: Maintains proper tool call and observation patterns

## Installation & Setup

### Step 1: Install LLaMA-Factory
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[all]
```

### Step 2: Prepare Training Data
```bash
# Copy the training data to LLaMA-Factory data directory
cp merged_gym_sft.json /path/to/LLaMA-Factory/data/
```

### Step 3: Configure Dataset Metadata
Add the following entry to `LLaMA-Factory/data/dataset_info.json`:

```json
{
  "merged_gym_sft": {
    "file_name": "merged_gym_sft.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system",
      "tools": "tools"
    }
  }
}
```

### Step 4: Configure Training Parameters
```bash
# Copy and edit the training configuration
cp qwen3_customized.yaml /path/to/LLaMA-Factory/examples/train_full/
```

**Important**: Update the `model_name_or_path` and `output_dir` in `qwen3_customized.yaml` to point to your base model and desired saving directory respectively:
```yaml
model_name_or_path: /path/to/your/qwen3-model
output_dir: saves/saved_folder_name
```

## Training Execution

### Multi-GPU Training (Recommended)
```bash
cd /path/to/LLaMA-Factory
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
llamafactory-cli train examples/train_full/qwen3_customized.yaml
```

## Training Configuration Details

### Model Configuration
- **Base Model**: Qwen3 instruction-tuned models
- **Trust Remote Code**: Enabled for custom model architectures
- **Fine-tuning Type**: Full fine-tuning (all parameters)

### Training Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `per_device_train_batch_size` | 2 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `learning_rate` | 1e-5 | Initial learning rate |
| `num_train_epochs` | 3.0 | Number of training epochs |
| `lr_scheduler_type` | cosine | Learning rate scheduler |
| `warmup_ratio` | 0.1 | Warmup ratio |
| `cutoff_len` | 16384 | Maximum sequence length |
