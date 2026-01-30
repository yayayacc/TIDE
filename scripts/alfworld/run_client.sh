
export AGENT_TYPE="client"
export MAX_STEPS=60
export CHAT_FORMAT="user_assistant_format"  

export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/alfworld/eval_in_distribution.sh

export CHAT_FORMAT="user_assistant_format_part"
export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/alfworld/eval_in_distribution.sh