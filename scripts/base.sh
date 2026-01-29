# General settings
TASK=${TASK:-"alfworld"}
OUT_DIR=${OUT_DIR:-"./res"}
FILE_NAME_PREFIX=${FILE_NAME_PREFIX:-"eval_in_distribution"}
DATA_PATH=${DATA_PATH:-"./data/alfworld/eval_in_distribution.json"}

# agent_proxy settings
AGENT_TYPE=${AGENT_TYPE:-"vllm"} # Agent type, options: "vllm" and "client"
STATE=${STATE:-"env"} # "no" means no environment feedback, "env" means use environment feedback, "empty" means use empty environment feedback, "internal" means use model-generated internal environment feedback
CHAT_FORMAT=${CHAT_FORMAT:-"user_assistant_format"} # Chat format, options: "default_format", "user_assistant_format", "user_history_format"
PROMPT_EXAMPLE=${PROMPT_EXAMPLE:-"fewshot"} # Prompt example type, options: "zeroshot", "fewshot"
TRAJECTORY_ROLLOUT_N=${TRAJECTORY_ROLLOUT_N:-4}
STEP_ROLLOUT_N=${STEP_ROLLOUT_N:-1}
MAX_STEPS=${MAX_STEPS:-50}
STOP_ON_ERROR=${STOP_ON_ERROR:-False}
ENABLE_THINKING=${ENABLE_THINKING:-True}
HISTORY_HAS_COT=${HISTORY_HAS_COT:-False}
HISTORY_WINDOW_SIZE=${HISTORY_WINDOW_SIZE:-0} # Only effective when chat_format is user_assistant_format, represents the history state window size
OFFER_FEEDBACK=${OFFER_FEEDBACK:-True} # Whether to provide environment feedback (effective in env state, additionally provides the environment error reason from the previous step)
DP=${DP:-4}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_STEP_LEN=${MAX_STEP_LEN:-$((1024*4))}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((1024*40))}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-1}
STOP=${STOP:-"</step>"}
SAFETY_MARGIN=${SAFETY_MARGIN:-100}
# client_agent settings
CLIENT_MODEL_NAME=${CLIENT_MODEL_NAME:-"qwen-plus"}
MODEL_SERIES=${MODEL_SERIES:-"qwen"} # qwen, minimax, hunyuan, deepseek
# vLLM_agent settings
MODEL_DIR=${MODEL_DIR:-"/your/model/dir"}
AIM_GPUS=${AIM_GPUS:-"4,5,6,7"}
TP=${TP:-1}
PP=${PP:-1}
MAX_LOGPROBS=${MAX_LOGPROBS:-30}
MAX_GPU_MEM_UTIL=${MAX_GPU_MEM_UTIL:-0.95}
USE_VLLM_TQDM=${USE_VLLM_TQDM:-False}
INCLUDE_STOP_STR_IN_OUTPUT=${INCLUDE_STOP_STR_IN_OUTPUT:-True}
ENFORCE_EAGER=${ENFORCE_EAGER:-False} # Whether to force enable vLLM eager mode
# Environment specific settings
ALFWORLD_EVAL_MODE=${ALFWORLD_EVAL_MODE:-"eval_in_distribution"} # Evaluation mode in alfworld environment, options: "eval_in_distribution" and "eval_out_of_distribution"
WEBSHOP_EVAL_MODE=${WEBSHOP_EVAL_MODE:-"full"} # Evaluation mode in webshop environment, options: "small" and "full"
RANDOM_STEP_ENABLE=${RANDOM_STEP_ENABLE:-False} # Whether to enable random step injection
RANDOM_STEP_NUM=${RANDOM_STEP_NUM:-5} # Number of random steps to inject
STOP_BY_SELF_ENABLE=${STOP_BY_SELF_ENABLE:-False} # Whether to enable model self-stop functionality
# --- Dynamically construct log file name based on parameters ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FILENAME_PARTS=(${TIMESTAMP})
if [ "${AGENT_TYPE}" = "vllm" ]; then
    MODEL_NAME=$(basename ${MODEL_DIR})
elif [ "${AGENT_TYPE}" = "client" ]; then
    MODEL_NAME=${CLIENT_MODEL_NAME}
fi

FILENAME_PARTS+=(${MODEL_NAME})

if [ -n "${STATE}" ]; then
    FILENAME_PARTS+=("${STATE}_state")
fi
if [ "${ENABLE_THINKING}" = "True" ]; then
    FILENAME_PARTS+=("thinking")
fi
if [ "${STOP_ON_ERROR}" = "False" ]; then
    FILENAME_PARTS+=("revise")
fi
if [ "${TRAJECTORY_ROLLOUT_N}" -gt 1 ]; then
    FILENAME_PARTS+=("traj_rollout${TRAJECTORY_ROLLOUT_N}")
fi
if [ "${STEP_ROLLOUT_N}" -gt 1 ]; then
    FILENAME_PARTS+=("step_rollout${STEP_ROLLOUT_N}")
fi
FILENAME_PARTS+=("${CHAT_FORMAT}")
# 4. Combine final log file path
FILENAME=$(IFS=-; echo "${FILENAME_PARTS[*]}")

# 4. Combine final log file path
LOG_DIR="./logs/${TASK}"
mkdir -p ${LOG_DIR} # Ensure log directory exists
LOG_FILE="${LOG_DIR}/${FILENAME}.log"

echo "Log will be saved to: ${LOG_FILE}"


# --- Execute Python script and redirect all output to log file ---
python -u ./main.py \
task=${TASK} \
data_path=${DATA_PATH} \
out_dir=${OUT_DIR} \
file_name_prefix=${FILE_NAME_PREFIX} \
agent_proxy.type=${AGENT_TYPE} \
agent_proxy.state=${STATE} \
agent_proxy.offer_feedback=${OFFER_FEEDBACK} \
agent_proxy.chat_format=${CHAT_FORMAT} \
agent_proxy.prompt_example=${PROMPT_EXAMPLE} \
agent_proxy.trajectory_rollout_n=${TRAJECTORY_ROLLOUT_N} \
agent_proxy.step_rollout_n=${STEP_ROLLOUT_N} \
agent_proxy.max_steps=${MAX_STEPS} \
agent_proxy.stop_on_error=${STOP_ON_ERROR} \
agent_proxy.enable_thinking=${ENABLE_THINKING} \
agent_proxy.history_has_cot=${HISTORY_HAS_COT} \
agent_proxy.history_window_size=${HISTORY_WINDOW_SIZE} \
agent_proxy.dp_size=${DP} \
agent_proxy.batch_size=${BATCH_SIZE} \
agent_proxy.max_step_len=${MAX_STEP_LEN} \
agent_proxy.max_model_len=${MAX_MODEL_LEN} \
agent_proxy.temperature=${TEMPERATURE} \
agent_proxy.top_p=${TOP_P} \
agent_proxy.stop="'${STOP}'" \
agent_proxy.safety_margin=${SAFETY_MARGIN} \
vllm_agent.model_path=${MODEL_DIR} \
vllm_agent.aim_gpus="'${AIM_GPUS}'" \
vllm_agent.tp_size=${TP} \
vllm_agent.pp_size=${PP} \
vllm_agent.max_logprobs=${MAX_LOGPROBS} \
vllm_agent.max_gpu_mem_util=${MAX_GPU_MEM_UTIL} \
vllm_agent.use_vllm_tqdm=${USE_VLLM_TQDM} \
vllm_agent.include_stop_str_in_output=${INCLUDE_STOP_STR_IN_OUTPUT} \
vllm_agent.enforce_eager=${ENFORCE_EAGER} \
client_agent.model_name=${CLIENT_MODEL_NAME} \
client_agent.model_series=${MODEL_SERIES} \
additional_exp.random_step.enable=${RANDOM_STEP_ENABLE} \
additional_exp.random_step.num=${RANDOM_STEP_NUM} \
additional_exp.stop_by_self.enable=${STOP_BY_SELF_ENABLE} \
env.alfworld.eval_mode=${ALFWORLD_EVAL_MODE} \
env.webshop.eval_mode=${WEBSHOP_EVAL_MODE} > ${LOG_FILE} 2>&1


echo "Inference completed. Log saved to: ${LOG_FILE}"