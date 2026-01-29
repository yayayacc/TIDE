import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import pandas as pd
from vllm import CompletionOutput
import numpy as np
from scipy.special import logsumexp
from openai.types.chat.chat_completion import Choice
import threading
from contextlib import contextmanager
import sys


def save_result_local(result: Dict[str, Any], out_dir: str , file_name: str, config, lock= None) -> Path:
    os.makedirs(out_dir, exist_ok=True)
    file_name += f"_traj_rollout_{config.agent_proxy.trajectory_rollout_n}_step_rollout_{config.agent_proxy.step_rollout_n}.jsonl"
    path = Path(out_dir) / file_name
    line_to_write = json.dumps(result, ensure_ascii=False) + "\n"
     # Use file lock to ensure safety during multi-process writing
    if lock:
        with lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line_to_write)
    else:
        # If no lock (single process single thread case), write directly
        with path.open("a", encoding="utf-8") as f:
            f.write(line_to_write)
    # print("save result to ", path)
    return path


def load_data_parquet(data_path: str) -> List[Dict[str, Any]]:
    df = pd.read_parquet(data_path)
    data = []
    for data_idx, row in df.iterrows():
        question = row['prompt'][0]['content']
        data.append({"query": question, "label": row['reward_model']['ground_truth']})
    return data

def get_logp_distribution(
    output: CompletionOutput,max_logprobs: int = 30
):
    """
    Get the log probability distribution for each generated token.
    Args:
        output (CompletionOutput): The output from the model generation.
        max_logprobs (int): The maximum number of log probabilities to retrieve for each token.
    Returns:
        np.ndarray: An array of shape (num_generated_tokens, max_logprobs) containing
                    the log probabilities for each generated token.
    """
    generated_token_ids = output.token_ids
    all_logprobs = []
    for i, token_id in enumerate(generated_token_ids):
        logprob_distribution = [
            logprob.logprob for token_id, logprob in output.logprobs[i].items()
        ]
        all_logprobs.append(logprob_distribution)

    all_logprobs = [logprobs[:max_logprobs] for logprobs in all_logprobs]
        
    all_logprobs = np.array(all_logprobs)
    return all_logprobs

def get_logp_distribution_from_api(choice: Choice, max_logprobs: int =30):
    """
    Get the log probability distribution for each generated token from an OpenAI Chat Completion Choice object.
    Args:
        choice (Choice): A single choice object from OpenAI Chat Completion.
        max_logprobs (int): Maximum number of log probabilities to retrieve per token.
    Returns:
        np.ndarray: Array of shape (num_generated_tokens, max_logprobs) containing log probabilities for each generated token.
    """
    all_logprobs = []
    for content in choice.logprobs.content:
        logprob_distribution = [t.logprob for t in content.top_logprobs]
        all_logprobs.append(logprob_distribution)
    all_logprobs = [logprobs[:max_logprobs] for logprobs in all_logprobs]
    all_logprobs = np.array(all_logprobs)
    return all_logprobs

def calculate_entropy(log_probs: np.ndarray | list) -> tuple[list, list]:
    """
    Calculate entropy from log probabilities (log_probs) in a numerically stable way.
    The base of log_probs determines the unit of entropy (e.g., natural log -> nats, base 2 -> bits).
    
    Args:
        log_probs (np.ndarray | list): Log probability array of shape (num_samples, vocab_size).
    Returns:
        list: List of entropy values for each token.
    """
    if isinstance(log_probs, list):
        log_probs = np.array(log_probs)
    # 1. Calculate probability p
    # Use logsumexp to ensure numerical stability when calculating probs from log_probs
    # p = exp(log_probs)
    probs = np.exp(log_probs - logsumexp(log_probs, axis=-1, keepdims=True))
    # Recalculate log_probs through probs
    log_probs = np.log(probs + 1e-20)  # Add a small constant to avoid log(0)
    # 2. Calculate entropy (Entropy)
    # Entropy = E[-log(p)] = -sum(p * log(p))
    # Use input log_probs for calculation to avoid log(0)
    entropy = -np.sum(probs * log_probs, axis=-1)

    # # 3. Calculate Varentropy
    # # Varentropy = Var(-log(p)) = E[(-log(p))^2] - (E[-log(p)])^2
    # # E[(-log(p))^2] = sum(p * (-log(p))^2)
    # second_moment = np.sum(probs * (log_probs ** 2), axis=-1)
    # varentropy = second_moment - entropy ** 2

    return entropy.tolist()

def cal_action_space_entropy(rollout_steps: List[dict]):
    action_counts = {}
    for step in rollout_steps:
        if step["action"] in action_counts:
            action_counts[step["action"]] += 1
        else:
            action_counts[step["action"]] = 1
    total_actions = len(rollout_steps)
    action_probs = np.array(
        [np.log(count / total_actions) for count in action_counts.values()]
    )
    entropy = calculate_entropy(
        action_probs.reshape(1, -1)
    )
    return entropy

_stdout_lock = threading.Lock()
@contextmanager
def suppress_output():
    """A context manager that temporarily suppresses printing to stdout and stderr (thread-safe)."""
    with _stdout_lock:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            try:
                yield
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr