#!/bin/bash

# run on 8xH100
# make sure your current working directory is the root of the project

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OPENAI_API_KEY="XXX"
# export OPENAI_BASE_URL="http://localhost:8000/v1"
# export MULTITURN_MODEL_NAME="Qwen/Qwen3-32B"

export OPENAI_BASE_URL="https://api.openai.com/v1"
export MULTITURN_MODEL_NAME="gpt-4o"

set -x

ulimit -n 65535

PROJECT_DIR="/path/to/your/UserRL"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='grpo_multiturn' \
    algorithm.adv_estimator=grpo_multiturn \
    algorithm.gamma=0.8 \
    data.train_batch_size=128 \
    data.max_prompt_length=1152 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=/path/to/your/model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.max_turns=16 \
    actor_rollout_ref.rollout.multi_turn.model_name="$MULTITURN_MODEL_NAME" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config/interact_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.turn_level_method="Equalized" \
    actor_rollout_ref.rollout.multi_turn.trajectory_score_method="Sum" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='UserRL' \
    trainer.experiment_name='turn_Equalized_trajectory_Sum' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    data.train_files=$PROJECT_DIR/data/alltrain_multiturn/train.parquet \
    data.val_files=$PROJECT_DIR/data/alltest_multiturn/test.parquet \
    trainer.total_epochs=15 $@