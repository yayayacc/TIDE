export FILE_NAME_PREFIX="eval_blocksworld_sample100_1_10"
export DATA_PATH="data/blocksworld/100_1_10.json"
export AGENT_TYPE="client"
export MAX_STEPS=30

export CHAT_FORMAT="user_assistant_format"  

export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/blocksworld/eval_blocksworld.sh

export CHAT_FORMAT="user_assistant_format_part"
export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/blocksworld/eval_blocksworld.sh