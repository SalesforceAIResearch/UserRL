#!/bin/bash

# Required field
export MAX_WORKER_NUM=10 # Maximum number of workers to use
export USER_MODEL_NAME="gpt-4o" # Model name for the user
export OPENAI_API_KEY="XXX"
export OPENAI_BASE_URL="https://api.openai.com/v1" # Base URL for the OpenAI API
export TOOL_CHOICE="auto" # Tool choice for the model, please keep it as "auto"
export PROJECT_ROOT="path/to/this/evaluation/directory"

# API key for the Gemini API
export GENAI_API_KEY="XXX"
# If using SearchGym to test on bamboogle
export SERPER_API_KEY="XXX"

# If you are using the official model
python eval.py \
    --model_name gpt-4o \
    --max_turns 16 \
    --pass_k 1 \
    --temperature 0 \
    --envs travel22 travel33 travel44 travel233 travel333 travel334 travel444 travel2222 bamboogle function intention persuasion tau telepathy turtle \
    --save_name outputs/results_gpt-4o

# If you are using your self-hosted model
python eval.py \
    --model_name ${CUSTOMIZED_SERVED_MODEL_NAME} \
    --port 8500 \
    --max_turns 16 \
    --pass_k 1 \
    --temperature 0 \
    --envs travel22 travel33 travel44 travel233 travel333 travel334 travel444 travel2222 bamboogle function intention persuasion tau telepathy turtle \
    --save_name outputs/results_${CUSTOMIZED_SERVED_MODEL_NAME}
