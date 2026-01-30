import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ragen.env.sodoku.config import SodokuEnvConfig
from ragen.env.sodoku.env import SodokuEnv
from utils.TaskRunner.TaskRunner import TaskRunnerBase as TaskRunner
from concurrent.futures import ThreadPoolExecutor
import threading

from utils.llm_agent.base_agent import BaseAgent
from utils.llm_agent.ctx_manager import ContextManager, StepMemory
from utils.TaskRunner.TaskRunner import TrajectoryInfo



class SodokuRunner(TaskRunner):
    def __init__(
        self, config, agent: BaseAgent, dataset: List[Dict[str, Any]] = None
    ) -> None:
        self.task = config.task
        self.config = config
        self.agent = agent
        self.state = config.agent_proxy.state
        self.max_steps = config.agent_proxy.max_steps
        self.traj_rollout_n = config.agent_proxy.trajectory_rollout_n
        self.step_rollout_n = config.agent_proxy.step_rollout_n
        self.stop_on_error = config.agent_proxy.stop_on_error
        self.batch_size = config.agent_proxy.batch_size
        self.offer_feedback = config.agent_proxy.offer_feedback
        self.enable_thinking = config.agent_proxy.enable_thinking
        self.env_lock = threading.Lock()
        self.init_lock = threading.Lock()
        self.think_tag = "analysis"
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = json.load(open(config.data_path, "r"))


    def get_state_description(self, state: str) -> str:
        """Get state description"""
        lines = state.strip().split("\n")
        n = len(lines)

        # Count empty positions
        empty_positions = []
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                if char == "_":
                    empty_positions.append((i, j))

        # description = f"This is a {n}x{n} Sodoku puzzle. "

        if len(empty_positions) == 1:
            description = f"There is one empty cell. The empty cell is at ({empty_positions[0][0]}, {empty_positions[0][1]}). "
        else:
            description = f"There are {len(empty_positions)} empty cells to fill. "
            description += f"The empty cells are at "
            for pos in empty_positions:
                description += f"({pos[0]}, {pos[1]}) and "
            description = description[:-5]
            description += "."

        return description

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
        _, reward, done, info = traj.env.safe_step(step_info["action"])
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
    
    def _initialize_trajectory(
        self, data_idx: int, data: dict, traj_rollout_idx: int
    ) -> TrajectoryInfo:
        """Parallelly initialize environment and context for a single trajectory."""
        with self.init_lock:
            env = SodokuEnv(
                config=SodokuEnvConfig(
                    grid_size=data["grid_size"],
                    seed=data["seed"],
                    remove_num=data["level"],
                )
            )
            init_obs = env.reset(data["seed"])
        task_prompts = getattr(self.config.prompt, self.task)
        chat_format = self.config.agent_proxy.chat_format
        fewshot_template = getattr(task_prompts.fewshot_example, chat_format)
        fewshot = fewshot_template.format(
            think_tag=self.think_tag,
            history_window_size=self.config.agent_proxy.history_window_size,
        )
        sodoku_size = data["grid_size"]
        sodoku_grid_size = sodoku_size * sodoku_size
        sodoku_grid_size_minus_1 = sodoku_grid_size - 1
        system_prompt = (
            task_prompts.system_prompt.format(
                think_tag=self.think_tag,
                max_steps=self.max_steps,
                sodoku_size=sodoku_size,
                sodoku_grid_size=sodoku_grid_size,
                sodoku_grid_size_minus_1=sodoku_grid_size_minus_1,
            )
            + fewshot
        )
        return TrajectoryInfo(
            idx_in_batch=data_idx,
            traj_rollout_idx=traj_rollout_idx,
            env=env,
            env_idx=0,
            ctx_manager=ContextManager(
                system_prompt=system_prompt,
                instruction_prompt=self.get_state_description(init_obs)+"\n"+init_obs,
                tokenizer=(
                    self.agent.tokenizer
                    if hasattr(self.agent, "tokenizer")
                    else None
                ),
                config=self.config,
            ),
            steps=[],
        )
    # Rewrite _run_batch_stepwise_rollout, because need not to release env back to pool
    def _run_batch_stepwise_rollout(self, trajectories: List[TrajectoryInfo]) -> List[Dict[str, Any]]:
        for step_idx in range(self.max_steps):
            if step_idx % 5 == 0:
                print(f"DP{self.dp_idx}: Starting step {step_idx} for all active trajectories...")
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

