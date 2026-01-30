export MODEL_SERIES="deepseek"
export CLIENT_MODEL_NAME="deepseek-v3.2"
export DP=16
export BATCH_SIZE=2
export TRAJECTORY_ROLLOUT_N=1
export STEP_ROLLOUT_N=1

bash ./scripts/blocksworld/run_client.sh
bash ./scripts/frozen_lake/run_client.sh
bash ./scripts/sodoku/run_client.sh
bash ./scripts/alfworld/run_client.sh
bash ./scripts/webshop/run_client.sh