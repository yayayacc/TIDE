
# TIDE: Trajectory-based Diagnostic Evaluation of Test-Time Improvement in LLM Agents

## Setup TIDE
First, create and activate a new conda environment
```bash
conda create -n tide python=3.12 -y
conda activate tide
```
If you want to run models other than Mistral-3-14B-Instruct, use the following command to install vllm==0.11.0
```bash
bash ./scripts/setup.sh
```
If you need to run the Mistral-3-14B-Instruct model, use the following command to install vllm==0.12.0 and the latest version of transformers from GitHub
```bash
bash ./scripts/setup_new.sh
pip install https://github.com/huggingface/transformers.git
```

### Download ALFWorld Dataset
```bash
export ALFWORLD_DATA="./data/alfworld"
alfworld-download
```

### Run Evaluations
Please refer to the script files in the `scripts` directory to run evaluations in different environments.

Notes:
1. For the Mistral-3-14B-Instruct model with vllm == 0.12.0, you need to set `export ENFORCE_EAGER=True` and `export MAX_LOGPROBS=0`
2. When running in the WebShop environment, DP can only be set to 1

### Analysis of Results
Analyze the trajectories of the experiments and generate the output json files
```bash
bash scripts/process_exp_res.sh
```
Analyze memory recall of alfworld experiments
```bash
bash scripts/process_alfworld_mem_recall.sh
```

## Output Data Format
```python

result: Dict[str, Any] = {
    "query": str,
    "all_accuracies": List[int],
    "avg_accuracy": float,
    "pass_at_k": float,
    "traj_rollouts": [
        {
            "rollout_idx": int,
            "rollout_results": rollout_trajectory
        },
        # ... more rollouts
    ]
}

rollout_trajectory: Dict[str, Any] = {
    "query": str ,
    "success": bool ,
    "loop": [
        {
            "step": int,
            "action": str,
        },
        ...
    ],
    "steps": [
        {
            "last_step_feedback": {
                "is_valid": bool,
                "message": str
            },
            "model_input": str,
            "response": str,
            "observation": str,
            "input_state": str,
            "true_state": str,
            "analysis": str,
            "action": str,
            "tokens": List[str],
            "token_ids": List[int],
            "action_space_entropy": float,

            "token_entropy_stats": {
                "analysis_stats": {
                    "mean": float,
                    "std": float,
                    "max": float,
                    "min": float,
                    "raw": List[float]
                },
                "action_stats": dict[str, float]
                },
            "env_feedback":{
                "action_is_valid": bool,
                "accuracy": int,
                "level": str,
                "done": bool
            }
        },
        ...
    ]
}
```