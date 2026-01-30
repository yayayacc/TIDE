export FILE_NAME_PREFIX="eval_fl_sample100_level6_15"
export DATA_PATH="data/frozen_lake/100_6_15.json"
export BATCH_SIZE=50
export MAX_STEPS=30
export OFFER_FEEDBACK=True


export CHAT_FORMAT="user_assistant_format"
export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/frozen_lake/eval_frozen_lake.sh


##############################################
export CHAT_FORMAT="user_assistant_format_part"
export HISTORY_WINDOW_SIZE=0
export ENABLE_THINKING=False
export HISTORY_HAS_COT=True
export STATE="env"
bash scripts/frozen_lake/eval_frozen_lake.sh

