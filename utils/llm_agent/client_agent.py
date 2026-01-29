from utils.llm_agent.base_agent import BaseAgent
from openai import OpenAI
from openai.types.chat.chat_completion import Choice, ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from typing import Dict, Any, List
from transformers import AutoTokenizer
import re
from bisect import bisect_right
from utils.func import get_logp_distribution_from_api, calculate_entropy
import numpy as np
import os
from dataclasses import dataclass
from utils.TaskRunner.TaskRunner import TrajectoryInfo
import asyncio
from openai import AsyncOpenAI
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.llm_agent.client import VLLMClient
from omegaconf import OmegaConf
class AutoClient:
    def __init__(self, config) -> None:
        self.model_series = config.client_agent.model_series
        self.model_name = config.client_agent.model_name
        api_config = getattr(config.api_config, self.model_series.lower())
        self.client_type = api_config.client_type.lower()
        model_config = getattr(api_config, self.model_name.lower(), None)
        self.apikey = api_config.api_key
        self.base_url = api_config.base_url
        if self.client_type == "openai":
            self.client = OpenAI(
                api_key=self.apikey,
                base_url=self.base_url
            )
        elif self.client_type == "vllm":
            self.client = VLLMClient(
                base_url=self.base_url,
                verbose=False,
            )
            self.client.health_check()
        else:
            raise NotImplementedError(f"Client type {api_config.client_type} not supported.")

    
    def chat(self,model: str, messages: List[Dict[str, str]], **kwargs) -> ChatCompletion:
        if self.client_type == "openai":
            output: ChatCompletion = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
        elif self.client_type == "vllm":
            output = self.client.chat_completion(
                messages=messages,
                model=model,
                **kwargs,
            )
        else:
            raise NotImplementedError(f"Client type {self.client_type} not supported.")
        return output
class ClientAgent(BaseAgent):
    def __init__(self, config) -> None:
        self.config = config
        self.model_series = config.client_agent.model_series
        self.model_name = config.client_agent.model_name
        api_config = getattr(config.api_config, self.model_series.lower())
        model_config = getattr(api_config, self.model_name.lower(), None)
        self.apikey = api_config.api_key
        self.base_url = api_config.base_url
        kwargs = getattr(model_config, "kwargs", {})
        # Convert OmegaConf object to regular dictionary
        if kwargs:
            # Use OmegaConf.to_container to convert to regular Python object
            kwargs = OmegaConf.to_container(kwargs, resolve=True)
            if not isinstance(kwargs, dict):
                kwargs = {}
        else:
            kwargs = {}
        self.api_model_name = model_config.get("model_name", self.model_name)
        self.sampling_param = {
            "n": config.agent_proxy.step_rollout_n,
            "temperature": config.agent_proxy.temperature,
            "top_p": config.agent_proxy.top_p,
            "max_tokens": config.agent_proxy.max_step_len,
            "stop": [config.agent_proxy.stop] if config.agent_proxy.stop else None,
            **kwargs,
        }
        self.llm = AutoClient(config)
        if api_config.client_type.lower() == "vllm":
            self.model_name = self.llm.client.list_models()[0]
        if model_config.has_enable_thinking:
            self.enable_thinking: bool = config.agent_proxy.enable_thinking
        else:
            self.enable_thinking: bool = False
            if config.agent_proxy.enable_thinking:
                raise ValueError(f"Model {self.model_name} does not support thinking.")
        self.step_rollout_n: int = config.agent_proxy.step_rollout_n
        self.think_tag: str = "analysis"

        
    def get_next_step(
        self, trajectory: TrajectoryInfo
    ) -> Dict[str, Any]:
        messages = trajectory.ctx_manager.format_messages()

        output: ChatCompletion = self.llm.chat.completions.create(
            model=self.api_model_name,
            messages=messages,
            **self.sampling_param,
        )
        # step rollout_n
        rollout_steps = []
        if self.step_rollout_n > 1:
            for choice in output.choices:
                step_info = self._parse_response(choice)
                rollout_steps.append(step_info)
            action_space_entropy = self._cal_action_space_entropy(rollout_steps)
            # TODO: if use tts, can choose best one
            step_info = rollout_steps[0]
        else:
            step_info = self._parse_response(output.choices[0])
            rollout_steps.append(step_info)
            action_space_entropy = 0

        return {
            "model_input": messages,
            **step_info,
            "action_space_entropy": action_space_entropy,
            "rollout_steps": rollout_steps,
        }

    def _parse_response(self, choice: Choice) -> Dict[str, Any]:
        if self.llm.client_type == "openai":
            return self._parse_response_openai(choice)
        elif self.llm.client_type == "vllm":
            return self._parse_response_vllm(choice)
        else:
            raise NotImplementedError(f"Client type {self.llm.client_type} not supported.")
        
    def _parse_response_openai(self, choice: Choice) -> Dict[str, Any]:
        think_pattern = r"<analysis>(.*?)</analysis>"
        action_pattern = r"<action>(.*?)</action>"
        full_text = choice.message.content

        # 1) First match think/analysis, then search for action after it, to avoid matching action inside think
        analysis_match = re.search(think_pattern, full_text, re.DOTALL)

        analysis_text = "DO_NOTHING"
        action_text = "DO_NOTHING"


        if analysis_match:
            analysis_text = analysis_match.group(1).strip()

            # Only search for action after think/analysis ends
            post_text = full_text[analysis_match.end():]
            action_match_after = re.search(action_pattern, post_text, re.DOTALL)
            if action_match_after:
                action_text = action_match_after.group(1).strip()
        else:
            # No think, then globally search for the first action
            action_match_any = re.search(action_pattern, full_text, re.DOTALL)
            if action_match_any:
                action_text = action_match_any.group(1).strip()


        return {
            "analysis": analysis_text.strip().lower() or "do_nothing",
            "action": action_text.strip().lower() or "do_nothing",
            "response": full_text,
            "token_entropy_stats": {},
        }
    
    def _parse_response_vllm(self, choice) -> Dict[str, Any]:
        think_pattern = r"<analysis>(.*?)</analysis>"
        action_pattern = r"<action>(.*?)</action>"
        full_text = choice["message"]["content"]

        # 1) First match think/analysis, then search for action after it, to avoid matching action inside think
        analysis_match = re.search(think_pattern, full_text, re.DOTALL)

        analysis_text = "DO_NOTHING"
        action_text = "DO_NOTHING"


        if analysis_match:
            analysis_text = analysis_match.group(1).strip()

            # Only search for action after think/analysis ends
            post_text = full_text[analysis_match.end():]
            action_match_after = re.search(action_pattern, post_text, re.DOTALL)
            if action_match_after:
                action_text = action_match_after.group(1).strip()
        else:
            # No think, then globally search for the first action
            action_match_any = re.search(action_pattern, full_text, re.DOTALL)
            if action_match_any:
                action_text = action_match_any.group(1).strip()


        return {
            "analysis": analysis_text.strip().lower() or "do_nothing",
            "action": action_text.strip().lower() or "do_nothing",
            "response": full_text,
            "token_entropy_stats": {},
        }
    def _cal_action_space_entropy(self, rollout_steps: List[Dict[str, Any]]) -> float:
        """
        Calculate the entropy of the action space based on the rollout_steps.
        Args:
            rollout_steps (List[Dict[str, Any]]): The list of rollout steps containing action information.
        Returns:
            The entropy of the action space.
        """
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
        return entropy[0]
    
    def get_next_step_parallel(self, trajectories: List[TrajectoryInfo]) -> List[Dict[str, Any]]:
        """
        Use thread pool to concurrently process multiple trajectories, ensuring result order matches input
        Args:
            trajectories: List of trajectory information
        Returns:
            List of processing results (order matches input)
        """
        # Pre-allocate result list to ensure order
        batch_results = [None] * len(trajectories)
        max_workers = 32
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and record indices
            future_to_idx = {
                executor.submit(
                    self._get_next_step_with_retry, 
                    traj.ctx_manager.format_messages()
                ): i 
                for i, traj in enumerate(trajectories)
            }
            
            # Collect results, place them in corresponding positions by index
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]  # Get original index
                try:
                    batch_results[idx] = future.result()  # Place in correct position
                except Exception as e:
                    print(f"Trajectory {idx} failed: {str(e)}")
                    batch_results[idx] = {
                        "model_input": trajectories[idx].ctx_manager.format_messages(),
                        "action_space_entropy": 0,
                        "analysis": "do_nothing",
                        "action": "do_nothing",
                        "response": f"Error: {str(e)}",
                        "token_entropy_stats": {},
                        "rollout_steps": [],
                    }
        
        return batch_results

    def _get_next_step_with_retry(
        self, 
        messages: List[Dict[str, Any]], 
        max_retries: int = 6,
        retry_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Request method with retry mechanism
        Args:
            messages: List of messages
            trajectory_idx: Trajectory index
            max_retries: Maximum number of retries
            retry_delay: Retry delay (seconds)
        Returns:
            Processing result dictionary
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Make synchronous request
                output = self.llm.chat(
                    model=self.api_model_name,
                    messages=messages,
                    **self.sampling_param,
                )
                
                # Parse response
                rollout_steps = []
                if self.llm.client_type == "openai":
                    for choice in output.choices:
                        step_info = self._parse_response(choice)
                        rollout_steps.append(step_info)
                elif self.llm.client_type == "vllm":
                    for choice in output["choices"]:
                        step_info = self._parse_response(choice)
                        rollout_steps.append(step_info)
                
                if not rollout_steps:
                    raise ValueError("No valid rollout steps generated")
                
                step_info = rollout_steps[0]
                action_space_entropy = self._cal_action_space_entropy(rollout_steps)
                
                return {
                    "model_input": messages,
                    **step_info,
                    "action_space_entropy": action_space_entropy,
                    "rollout_steps": rollout_steps,
                }
                
            except Exception as e:
                last_error = e
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                print(error_msg)
                
                # If not the last attempt, wait and retry
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    # Last attempt failed, return error result
                    print(f"Trajectory failed after {max_retries} attempts")
                    return {
                        "model_input": messages,
                        "action_space_entropy": 0,
                        "analysis": "do_nothing",
                        "action": "do_nothing",
                        "response": f"Error: {str(last_error)}",
                        "token_entropy_stats": {},
                        "rollout_steps": [],
                    }
    
if __name__ == "__main__":
    from utils.llm_agent.ctx_manager import ContextManager
    from ragen.env.blocksworld.env import BlocksworldEnv
    from utils.TaskRunner.BlocksWorldRunner import BlocksWorldRunner
    import json
    data = json.load(open("data/blocksworld/test_data.json", "r"))
    runner = BlocksWorldRunner()
    ctx_manager = ContextManager(
        
    )
    t_test = TrajectoryInfo(
        ctx_manager=ctx_manager,
        traj_rollout_idx=0,
        
    )