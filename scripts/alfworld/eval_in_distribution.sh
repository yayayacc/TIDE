# General settings
export TASK="alfworld"
export FILE_NAME_PREFIX="eval_in_distribution"
export DATA_PATH="./data/alfworld/eval_in_distribution.json"


# Environment specific settings
export ALFWORLD_EVAL_MODE="eval_in_distribution" # Evaluation mode in alfworld environment, options: "eval_in_distribution" and "eval_out_of_distribution"
bash scripts/base.sh