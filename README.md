
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

## Run Evaluations
Please refer to the script files in the `scripts` directory to run evaluations in different environments.

Notes:
1. For the Mistral-3-14B-Instruct model with vllm == 0.12.0, you need to set `export ENFORCE_EAGER=True` and `export MAX_LOGPROBS=0`
2. When running in the WebShop environment, DP can only be set to 1
### Run Remote models
You need to create an yaml file with your API keys at `config/api_key.yaml`.
Example:
```yaml
api_config:
  deepseek:
    base_url: "<api_url>"
    api_key: "<api_key>"
  gemini:
    base_url: "<api_url>"
    api_key: "<api_key>"
```
You can set different key for different remote model providers.
### Analysis of Results
Analyze the trajectories of the experiments and generate the output json files
```bash
bash scripts/process_exp_res.sh
```
Analyze memory recall of alfworld experiments
```bash
bash scripts/process_alfworld_mem_recall.sh
```
If you meet the error of `module 'numpy' has no attribute 'trapz'`,you can try to downgrade numpy to version 2.2.6

## Citation
If you find TIDE useful in your research, please consider citing the following paper:

```
wait for paper publication
```