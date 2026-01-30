export DATA_PATH="./data/webshop/test_data.json"
# different from other scripts, specify GPU settings here
export AIM_GPUS="0,1"
export DP=1
export TP=2
export BATCH_SIZE=100
export MAX_STEPS=20
export OFFER_FEEDBACK=True


export CHAT_FORMAT="user_assistant_format"  

export ENABLE_THINKING=False
export HISTORY_HAS_COT=True

export STATE="env"
bash scripts/webshop/eval_small.sh


export CHAT_FORMAT="user_assistant_format_part"
export HISTORY_WINDOW_SIZE=0
export ENABLE_THINKING=False
export HISTORY_HAS_COT=True

export STATE="env"
bash scripts/webshop/eval_small.sh