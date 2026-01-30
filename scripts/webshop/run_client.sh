export DATA_PATH="./data/webshop/test_data.json"
export DP=1
export AGENT_TYPE="client"
export BATCH_SIZE=32
export MAX_STEPS=20

export CHAT_FORMAT="user_assistant_format"  

export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/webshop/eval_small.sh

export CHAT_FORMAT="user_assistant_format_part"
export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/webshop/eval_small.sh