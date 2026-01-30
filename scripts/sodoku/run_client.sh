export FILE_NAME_PREFIX="eval_sodoku_sample100_gs3_1_15"
export DATA_PATH="data/sodoku/100_gs3_1_15.json"
export AGENT_TYPE="client"
export MAX_STEPS=30

export CHAT_FORMAT="user_assistant_format"  

export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/sodoku/eval_sodoku.sh

export CHAT_FORMAT="user_assistant_format_part"  

export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/sodoku/eval_sodoku.sh