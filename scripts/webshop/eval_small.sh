export TASK="webshop"
export FILE_NAME_PREFIX="eval_small"
# Special Environment configuration parameters
export WEBSHOP_EVAL_MODE="small" # Evaluation mode in webshop environment, options: "small" and "full"
export DATA_PATH="data/webshop/test_data.json"

bash scripts/base.sh

