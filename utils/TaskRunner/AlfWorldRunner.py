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

    def check_target_involvement(self, traj: TrajectoryInfo, action: str) -> bool:
        """Check if action involves target object or locations"""
        from utils.analysis_files.alfworld_mem_recall import extract_objects_and_locations
        
        # Extract task-related objects and locations from instruction
        task_type, _, _, task_objects, task_locations = extract_objects_and_locations(
            traj.ctx_manager.instruction_prompt
        )
        
        # Extract objects and locations from action
        _, action_locations, action_objects, _, _ = extract_objects_and_locations(action)
        
        # Check if involves target
        target_obj_involved = any(
            obj[0].lower() in [t.lower() for t in task_objects] 
            for obj in action_objects
        )
        target_loc_involved = any(
            loc[0].lower() in [t.lower() for t in task_locations] 
            for loc in action_locations
        )
        
        return target_obj_involved or target_loc_involved
        
    def inject_random_steps(self, traj: TrajectoryInfo):
        """Inject random steps into trajectory"""
        import random
        
        for i in range(self.random_step_num):
            # Get currently available actions
            available_actions = traj.env.get_available_actions()
            if not available_actions:
                break
            self._prepare_step_history(traj)
            # Randomly select an action (can add more strategies)
            random_action = random.choice(available_actions)
            
            
            # Record this as an injected random step
            random_step_record = {
                "step_idx": len(traj.steps),
                "is_random_injected": True,
                "action": random_action,
                "analysis": " ",
            }
            
            self._process_step(traj, random_step_record)
    
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
        
    def _run_batch_stepwise_rollout(self, trajectories: List[TrajectoryInfo]) -> List[Dict[str, Any]]:
        injected_traj_indices = set()
        for step_idx in range(self.max_steps):
            active_trajectories = [
                traj for traj in trajectories if not traj.done
            ]
            if not active_trajectories:
                break

            # 3. Prepare history for each active trajectory
            for traj in active_trajectories:
                self._prepare_step_history(traj)
                
            valid_trajectories = self._check_and_filter_valid_trajectories(active_trajectories)

            if not valid_trajectories:
                continue
            # 4. batch get next step from agent
            # batch_outputs length should be len(valid_trajectories)
            batch_outputs = self.agent.get_next_step_parallel(valid_trajectories)

            # 5. Process each trajectory's step output
            for i, traj in enumerate(valid_trajectories):
                # Get corresponding step_info from batch_outputs
                step_info = batch_outputs[i]
                self._process_step(traj, step_info)
                if self.random_step_enable and not traj.done:
                    cur_step_num = len(traj.steps)
                    remain_steps = self.max_steps - cur_step_num
                    traj_id = traj.idx_in_batch
                    if (traj_id not in injected_traj_indices and
                        remain_steps >= self.random_step_num + 5):
                        action = step_info["action"]
                        if self.check_target_involvement(traj, action):
                            self.inject_random_steps(traj)
                            injected_traj_indices.add(traj_id)

        # 6. Release env and assemble final results
        for traj in trajectories:
            self._release_env_to_pool(getattr(traj, 'env_idx', -1))
        batch_results = [[] for _ in range(self.batch_size)]
        for traj in trajectories:
            loop_info = traj.ctx_manager.check_loop()
            rollout_result = {
                "steps": traj.steps,
                "success": traj.success,
                "stop_right": getattr(traj, "stop_right", None),
                "loop": loop_info,
                "meta": {"total_steps": len(traj.steps)},
            }
            rollout_info = {
                "rollout_idx": traj.traj_rollout_idx,
                "rollout_results": rollout_result,
            }
            batch_results[traj.idx_in_batch].append(rollout_info)
        return batch_results
