export BATCH_SIZE=70
export MAX_STEPS=60
export OFFER_FEEDBACK=True


export CHAT_FORMAT="user_assistant_format"  
export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/alfworld/eval_in_distribution.sh


export CHAT_FORMAT="user_assistant_format_part"
export HISTORY_WINDOW_SIZE=0
export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/alfworld/eval_in_distribution.sh
