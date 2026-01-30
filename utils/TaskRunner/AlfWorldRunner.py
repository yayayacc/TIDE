from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any, Dict, List, Optional, Tuple
import os
import re
import json
from ragen.env.alfworld.env import AlfredTXTEnv

from utils.TaskRunner.TaskRunner import TaskRunnerBase as TaskRunner
from transformers import AutoTokenizer
from utils.func import save_result_local, suppress_output
from numpy import ndarray
import numpy as np
import sys
from utils.llm_agent.vllm_agent import VLLMAgent
from utils.llm_agent.client_agent import ClientAgent
from utils.llm_agent.ctx_manager import ContextManager, StepMemory
from utils.TaskRunner.TaskRunner import TrajectoryInfo
import random

class AlfWorldRunner(TaskRunner):
    def __init__(
        self, config: Any, agent: VLLMAgent | ClientAgent, dataset: Any = None
    ) -> None:
        super().__init__(config, agent, dataset)
        # random step injection experiment
        self.random_step_enable = self.config.additional_exp.random_step.enable
        self.random_step_num = self.config.additional_exp.random_step.num
        

    def init_everything(self):
        os.environ["ALFWORLD_DATA"] = self.config.env.alfworld.data_path
        num_envs = self.batch_size * self.traj_rollout_n
        
        print(f"Creating {num_envs} AlfWorld environments...")
        with ThreadPoolExecutor(max_workers=num_envs) as executor:
            futures = []
            for _ in range(num_envs):
                futures.append(executor.submit(self._create_env))
            self.env_pool: list[AlfredTXTEnv] = [future.result() for future in futures]
        print(f"AlfWorld has {len(self.env_pool[0].game_files)} game files.")
        self.used_envs = [False] * len(self.env_pool)


    def _create_env(self) -> AlfredTXTEnv:
        with suppress_output():
            env = AlfredTXTEnv(mode=self.config.env.alfworld.eval_mode)
            return env

    def _initialize_trajectory(self, data_idx: int, data: dict, traj_rollout_idx: int) -> TrajectoryInfo:
        """Parallelly initialize environment and context for a single trajectory."""
        env_idx, env = self._get_env_from_pool()
        with self.init_lock:
            env.reset(data["seed"], mode="test")
        instruction_text = env.instruction_text if self.config.agent_proxy.chat_format=="user_assistant_format_part" else " "
        return TrajectoryInfo(
            idx_in_batch=data_idx,
            traj_rollout_idx=traj_rollout_idx,
            env=env,
            env_idx=env_idx,
            ctx_manager=ContextManager(
                system_prompt=self.system_prompt,
                instruction_prompt=instruction_text,
                tokenizer=self.agent.tokenizer if hasattr(self.agent, 'tokenizer') else None,
                config=self.config,
            ),
            steps=[],
        )
        