export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MERGED_MODEL_PATH="path/to/your/model/global_step_XXX_hf"
export CUSTOMIZED_SERVED_MODEL_NAME="XXX"

vllm serve ${MERGED_MODEL_PATH} \
   --max-model-len 32768 \
   --port 8500 \
   --gpu-memory-utilization 0.8 \
   --tensor-parallel-size 8 \
   --enable-auto-tool-choice \
   --tool-call-parser hermes \
   --reasoning-parser qwen3 \
   --served-model-name ${CUSTOMIZED_SERVED_MODEL_NAME}
