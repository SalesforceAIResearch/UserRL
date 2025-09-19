

MODEL_PATH="path/to/your/model/global_step_XXX"

python merger.py merge \
    --backend fsdp \
    --local_dir ${MODEL_PATH}/actor \
    --target_dir ${MODEL_PATH}_hf
