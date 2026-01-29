from typing import Any, Dict, List, Optional, Tuple

import re
import json
from ragen.env.blocksworld.env import BlocksworldEnv, BlocksworldEnvConfig
from vllm import SamplingParams, LLM, CompletionOutput
from utils.TaskRunner.TaskRunner import TaskRunnerBase as TaskRunner
from transformers import AutoTokenizer
from utils.func import (
    save_result_local,
    suppress_output
)
from numpy import ndarray
import numpy as np
import threading
from contextlib import contextmanager
import sys
import os
from concurrent.futures import ThreadPoolExecutor
from utils.llm_agent.vllm_agent import VLLMAgent
from utils.TaskRunner.TaskRunner import TrajectoryInfo
from utils.llm_agent.ctx_manager import ContextManager, StepMemory
from utils.llm_agent.client_agent import ClientAgent

class BlocksWorldRunner(TaskRunner):
    def __init__(self, config: Any, agent: VLLMAgent|ClientAgent, data: Any=None) -> None:
        super().__init__(config, agent, data)

    def init_everything(self):
        num_envs = self.batch_size * self.traj_rollout_n
        print(f"Creating {num_envs} Blocksworld environments...")
        with ThreadPoolExecutor(max_workers=num_envs) as executor:
            futures = []
            for _ in range(num_envs):
                futures.append(executor.submit(self._create_env))
            self.env_pool = [future.result() for future in futures]
                
        self.used_envs = [False] * len(self.env_pool)
        
    def _create_env(self):
        with suppress_output():
            env = BlocksworldEnv()
            return env
                

    def _process_step(self, traj: TrajectoryInfo, step_info: Dict[str, Any]):
        """Process a single step of a single trajectory, including executing action and updating state."""
        traj.steps[-1].update(step_info)
        traj.ctx_manager.history[-1].analysis = step_info["analysis"]
        traj.ctx_manager.history[-1].action = step_info["action"]
        if self.config.additional_exp.stop_by_self.enable:
            if traj.success:
                traj.done = True
                if step_info["action"].lower() == "stop":
                    traj.stop_right = True
                else:
                    traj.stop_right = False
                traj.ctx_manager.history[-1].is_valid = True
                traj.ctx_manager.history[-1].feedback = "Task already succeeded, agent stopped correctly."
                traj.steps[-1]["env_feedback"] = {
                    "action_is_valid": True,
                    "reward": 0,
                    "done": True,
                    "info": {"success": True, "reason": "Task already succeeded, agent stopped correctly."},
                }
                return
        if step_info["action"].lower() == "stop":
            traj.done = True
            traj.ctx_manager.history[-1].is_valid = True
            traj.ctx_manager.history[-1].feedback = "Agent chose to stop."
            traj.steps[-1]["env_feedback"] = {
                "action_is_valid": True,
                "reward": 0,
                "done": True,
                "info": {"success": traj.success, "reason": "Agent chose to stop."},
            }
            return
        # execute action in environment
        _, reward, done, info = traj.env.step(step_info["action"])
        # update ctx_manager history based on env feedback
        if not info.get("action_is_valid", True):
            traj.ctx_manager.history[-1].is_valid = False
            traj.ctx_manager.history[-1].feedback = (
                f"Action {step_info['action']} failed to execute."
            )
        else:
            traj.ctx_manager.history[-1].is_valid = True
            traj.ctx_manager.history[-1].feedback = ""
        
        # update traj.steps
        traj.steps[-1]["env_feedback"] = {
            "action_is_valid": info.get("action_is_valid", True),
            "reward": reward,
            "done": done,
            "info": info,
        }

        if info.get("success", False):
            traj.success = True

        if self.config.additional_exp.stop_by_self.enable:
            return
            
        if (
            done
            or traj.success
            or (self.stop_on_error and not traj.ctx_manager.history[-1].is_valid)
        ):
            traj.done = True

    def _run_stepwise_episode(self, data) -> Dict[str, Any]:
        """Stepwise reasoning mode: use environment feedback for multi-step reasoning"""
        # Run multiple rollouts, save separately
        all_rollouts = []
        all_accuracies = []

        for traj_idx in range(self.traj_rollout_n):
            
            env_idx, env = self._get_env_from_pool()
            env.reset(data, mode="test")
            traj = TrajectoryInfo(
                idx_in_batch=0,
                traj_rollout_idx=traj_idx,
                env=env,
                env_idx=env_idx,
                ctx_manager=ContextManager(
                    system_prompt=self.system_prompt,
                    instruction_prompt=env.instruction_text,
                    tokenizer=self.agent.tokenizer,
                    config=self.config,
                ),
                steps=[],
            )
            self.agent.reset(self.system_prompt, env.instruction_text)
            rollout_result = self._run_single_stepwise_rollout(traj)

            # Save complete rollout information
            rollout_info = {
                "rollout_idx": traj_idx,
                "rollout_results": rollout_result,
            }
            all_rollouts.append(rollout_info)

            # Determine if successful
            accuracy = 1 if rollout_result["success"] else 0
            all_accuracies.append(accuracy)
            self._release_env_to_pool(env_idx)

        # Calculate statistics
        avg_accuracy = (
            sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        )
        pass_at_n = any(all_accuracies) if all_accuracies else False

        return {
            "query": traj.ctx_manager.instruction_prompt,
            "seed": data["seed"],
            "all_accuracies": all_accuracies,
            "avg_accuracy": avg_accuracy,
            "pass_at_n": pass_at_n,
            "traj_rollouts": all_rollouts,
            "meta": {
                "traj_rollout_n": self.traj_rollout_n,
                "step_rollout_n": self.step_rollout_n,
            },
        }

    def _initialize_trajectory(self, data_idx: int, data: dict, traj_rollout_idx: int) -> TrajectoryInfo:
        """Parallelly initialize environment and context for a single trajectory."""
        env_idx, env = self._get_env_from_pool()
        with self.init_lock:
            env.reset(data)
        return TrajectoryInfo(
            idx_in_batch=data_idx,
            traj_rollout_idx=traj_rollout_idx,
            env=env,
            env_idx=env_idx,
            ctx_manager=ContextManager(
                system_prompt=self.system_prompt,
                instruction_prompt=env.instruction_text,
                tokenizer=(
                    self.agent.tokenizer
                    if hasattr(self.agent, "tokenizer")
                    else None
                ),
                config=self.config,
            ),
            steps=[],
        )