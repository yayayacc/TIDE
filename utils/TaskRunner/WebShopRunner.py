from typing import Any, Dict, List, Optional, Tuple

import re
import json
from ragen.env.webshop.env import WebShopEnv
from ragen.env.webshop.config import WebShopEnvConfig
from vllm import SamplingParams, LLM, CompletionOutput
from utils.TaskRunner.TaskRunner import TaskRunnerBase as TaskRunner
from transformers import AutoTokenizer
from utils.func import (
    save_result_local,
    suppress_output
)
import os
from concurrent.futures import ThreadPoolExecutor
from utils.llm_agent.vllm_agent import VLLMAgent
from utils.llm_agent.ctx_manager import ContextManager, StepMemory
from utils.llm_agent.client_agent import ClientAgent
from utils.TaskRunner.TaskRunner import TrajectoryInfo


class WebShopRunner(TaskRunner):
    def __init__(
        self, config: Any, agent: VLLMAgent | ClientAgent, dataset: Any = None
    ) -> None:
        super().__init__(config, agent, dataset)

    def init_everything(self):
        num_envs = self.batch_size * self.traj_rollout_n

        print(f"Creating {num_envs} WebShop environments...")
        with ThreadPoolExecutor(max_workers=num_envs) as executor:
            futures = []
            for _ in range(num_envs):
                futures.append(executor.submit(self._create_env))
            self.env_pool: list[WebShopEnv] = [future.result() for future in futures]

        self.used_envs = [False] * len(self.env_pool)

    def _create_env(self) -> WebShopEnv:
        with suppress_output():
            env = WebShopEnv(mode="test")
            return env