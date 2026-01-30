export MAX_MODEL_LEN=$((30*1024))
export TRAJECTORY_ROLLOUT_N=1
export STEP_ROLLOUT_N=1
# We run experiments on 2 A100-80G GPUs
# Run Llama-3.1-8B
export AIM_GPUS="0,1"
export DP=2
export MODEL_DIR="<path_to_your_model>/Llama-3.1-8B"
bash ./scripts/blocksworld/run_exp.sh
bash ./scripts/frozen_lake/run_exp.sh
bash ./scripts/sodoku/run_exp.sh
bash ./scripts/alfworld/run_exp.sh
bash ./scripts/webshop/run_exp.sh

# Run Llama-3.3-70B
export AIM_GPUS="0,1"
export DP=1
export TP=2
export MODEL_DIR="<path_to_your_model>/Llama-3.3-70B"
bash ./scripts/blocksworld/run_exp.sh
bash ./scripts/frozen_lake/run_exp.sh
bash ./scripts/sodoku/run_exp.sh
bash ./scripts/alfworld/run_exp.sh
bash ./scripts/webshop/run_exp.sh
