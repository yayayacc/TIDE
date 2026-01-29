import os
import re
from typing import Any, Dict, Optional,List
from vllm import LLM
from transformers import AutoTokenizer
from dataclasses import dataclass
from utils.llm_agent.ctx_manager import ContextManager
from utils.llm_agent.ctx_manager import StepMemory
from utils.llm_agent.base_agent import BaseAgent
from ragen.env.base import BaseEnv
from omegaconf import DictConfig
import threading
from utils.func import save_result_local
from concurrent.futures import ThreadPoolExecutor
@dataclass
class TrajectoryInfo:
    idx_in_batch: int
    traj_rollout_idx: int
    env: BaseEnv
    env_idx: int
    ctx_manager: ContextManager
    steps: List[Dict[str, Any]]
    done: bool = False
    success: bool = False
    stop_right: bool = False
    

class TaskRunnerBase:
    def __init__(self, config, agent: BaseAgent, dataset: List[Dict[str, Any]] = None) -> None:
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
        self.init_everything()
        self._init_system_prompt()
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = json.load(open(config.data_path, "r"))
        
    def init_everything(self):
        """Initialize environment pool and other resources"""
        num_envs = self.batch_size * self.traj_rollout_n
        self.env_pool = [self._create_env() for _ in range(num_envs)]
        self.used_envs = [False] * len(self.env_pool)
        raise NotImplementedError("Subclass must implement init_everything method.")
        
                
    
    def _create_env(self) -> BaseEnv:
        """Create and return a new environment instance. Subclasses should override this method to return task-specific environment."""
        raise NotImplementedError("Subclass must implement _create_env method.")
    
    def _init_system_prompt(self):
        """Initialize system prompt. Subclasses can override this method as needed."""
        if not hasattr(self.config.prompt, self.task):
            raise ValueError(f"Task '{self.task}' definition not found in prompt configuration.")
        task_prompts = getattr(self.config.prompt, self.task)
        chat_format = self.config.agent_proxy.chat_format
        if not hasattr(task_prompts.fewshot_example, chat_format):
            raise ValueError(f"Chat format '{chat_format}' definition not found in prompt configuration.")
        if self.config.agent_proxy.prompt_example == "fewshot":
            exmaple_template = getattr(task_prompts.fewshot_example, chat_format)
            example = exmaple_template.format(
                think_tag=self.think_tag,
                history_window_size=self.config.agent_proxy.history_window_size,
            )
        elif self.config.agent_proxy.prompt_example == "zeroshot":
            example_template = getattr(task_prompts.zeroshot_example, chat_format)
            example = example_template.format(
                think_tag=self.think_tag,
                history_window_size=self.config.agent_proxy.history_window_size,
            )
        self.system_prompt = (
            task_prompts.system_prompt.format(
                think_tag=self.think_tag,
                max_steps=self.max_steps,
            )
            + example
        )

    def _get_env_from_pool(self):
        """Get an unused environment from the pool."""
        with self.env_lock:
            for i, used in enumerate(self.used_envs):
                if not used:
                    self.used_envs[i] = True
                    return i, self.env_pool[i]

    def _release_env_to_pool(self, env_idx):
        """Release environment back to pool."""
        if env_idx != -1:
            with self.env_lock:
                self.used_envs[env_idx] = False
    
    def _prepare_step_history(self, traj: TrajectoryInfo):
        """Prepare history and state information for current step of trajectory."""
        # add new StepMemory for current step
        step_idx = len(traj.steps)
        traj.ctx_manager.history.append(
            StepMemory(
                previous_memory=traj.ctx_manager.history[-1] if len(traj.ctx_manager.history) > 0 else StepMemory()
            )
        )
        # add new step record
        traj.steps.append(
            {
                "step_idx": step_idx,
                "last_step_feedback": {
                    "is_valid": traj.ctx_manager.history[-1].previous_memory.is_valid,
                    "message": traj.ctx_manager.history[-1].previous_memory.feedback,
                },
            }
        )
        # get current observation from environment
        observation = traj.env.render()
        traj.ctx_manager.history[-1].observation = observation
        traj.steps[-1]["observation"] = observation

        # Process state info depending on arg
        state_info = self._get_state(traj.ctx_manager.history[-1])
        traj.steps[-1].update(state_info)
        traj.ctx_manager.history[-1].true_state = state_info["true_state"]
        traj.ctx_manager.history[-1].input_state = state_info["input_state"]

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

    def run(self, dp_idx=0, lock=None) -> None:
        batch = []
        for data_idx, data in enumerate(self.dataset):
            if data_idx % 10 == 0:
                print(
                    f"Worker {dp_idx} processing sample {data_idx}/{len(self.dataset)}..."
                )
            batch.append(data)
            if len(batch) == self.batch_size:
                result_dicts = self.run_episode_batch(
                    batch=batch,
                )
                for result_dict in result_dicts:
                    # Save results
                    save_result_local(
                        result_dict,
                        out_dir=self.config.out_dir,
                        file_name=self.config.file_name_prefix,
                        config=self.config,
                        lock=lock,
                    )
                batch = []
        # Process remaining data
        if len(batch) > 0:
            result_dicts = self.run_episode_batch(
                batch=batch,
            )
            for result_dict in result_dicts:
                # Save results
                save_result_local(
                    result_dict,
                    out_dir=self.config.out_dir,
                    file_name=f"{self.config.file_name_prefix}",
                    config=self.config,
                    lock=lock,
                )


    def run_episode_batch(
        self,
        batch: List[dict],
    ) -> List[Dict[str, Any]]:

        return self._run_stepwise_episode_batch(batch)

    def _run_stepwise_episode(self, data) -> Dict[str, Any]:
        """Stepwise reasoning mode: use environment feedback for multi-step reasoning"""
        # Run multiple rollouts, save separately
        all_rollouts = []
        all_accuracies = []

        for traj_idx in range(self.traj_rollout_n):
            
            env_idx, env = self._get_env_from_pool()
            env.reset(data["seed"], mode="test")
            traj = TrajectoryInfo(
                idx_in_batch=0,
                traj_rollout_idx=traj_idx,
                env=env,
                env_idx=env_idx,
                ctx_manager=ContextManager(
                    system_prompt=self.system_prompt,
                    instruction_prompt=env.instruction_text,
                    tokenizer=self.agent.tokenizer if hasattr(self.agent, 'tokenizer') else None,
                    config=self.config,
                ),
                steps=[],
            )
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
    
    def _run_single_stepwise_rollout(self, traj: TrajectoryInfo) -> Dict[str, Any]:
        """Run a single trajectory for problem solving"""

        for step_idx in range(self.max_steps):
            self._prepare_step_history(traj)

            step_info = self.agent.get_next_step(traj)
            self._process_step(traj, step_info)
            if traj.done:
                break
                
        loop_info = traj.ctx_manager.check_loop()
        return {
            "steps": traj.steps,
            "success": traj.success,
            "loop": loop_info,
            "meta": {"total_steps": len(traj.steps)},
        }

    def _get_state(self, step_mem: StepMemory) -> str:
        true_state = step_mem.observation
        if self.state == "no" or self.state == "empty":
            input_state = " "
        elif self.state == "env":
            if self.offer_feedback:
                if step_mem.previous_memory.is_valid:
                    input_state = step_mem.observation
                else:
                    input_state = f"{step_mem.previous_memory.feedback}, {step_mem.observation}"
        elif self.state == "internal":
            input_state = self.agent.get_internal_state()
        elif self.state == "random":
            # TODO: implement random state
            raise NotImplementedError
        return {
            "true_state": true_state,
            "input_state": input_state,
        }

    def _initialize_trajectory(self, data_idx: int, data: dict, traj_rollout_idx: int) -> TrajectoryInfo:
        """Parallelly initialize environment and context for a single trajectory."""
        env_idx, env = self._get_env_from_pool()
        with self.init_lock:
            env.reset(data["seed"], mode="test")
        return TrajectoryInfo(
            idx_in_batch=data_idx,
            traj_rollout_idx=traj_rollout_idx,
            env=env,
            env_idx=env_idx,
            ctx_manager=ContextManager(
                system_prompt=self.system_prompt,
                instruction_prompt=env.instruction_text,
                tokenizer=self.agent.tokenizer if hasattr(self.agent, 'tokenizer') else None,
                config=self.config,
            ),
            steps=[],
        )
    
    def _check_and_filter_valid_trajectories(self, active_trajectories: List[TrajectoryInfo]) -> List[TrajectoryInfo]:
        """Check prompt length, terminate trajectory and remove current step if exceeds limit"""
        valid_trajectories = []
        # Set safety margin to prevent vllm errors due to tokenizer differences and reserve space for generation
        safety_margin = self.config.agent_proxy.safety_margin
        # Ensure limit does not exceed model max length minus margin
        limit = self.config.agent_proxy.max_model_len - safety_margin

        if hasattr(self.agent, "tokenizer") and self.agent.tokenizer:
            prompts = [traj.ctx_manager.format_prompt() for traj in active_trajectories]
            # Batch tokenize for efficiency
            batch_input_ids = self.agent.tokenizer(prompts, add_special_tokens=True)["input_ids"]
            
            for i, traj in enumerate(active_trajectories):
                token_len = len(batch_input_ids[i])
                if token_len >= limit:
                    traj.done = True
                    traj.success = False
                    # Remove the last step as it was only prepared but not processed
                    if len(traj.steps) > 0:
                        traj.steps.pop()
                    if len(traj.ctx_manager.history) > 0:
                        traj.ctx_manager.history.pop()
                    
                    # Simply print or log here, as traj has already been marked as done
                    print(f"Sample_idx in batch: {traj.idx_in_batch}, Traj_rollout_idx: {traj.traj_rollout_idx}, stopped: Prompt length {token_len} exceeds limit {limit} (max {self.config.agent_proxy.max_model_len} - margin {safety_margin})")
                else:
                    valid_trajectories.append(traj)
        else:
            valid_trajectories = active_trajectories
        return valid_trajectories

    def _run_stepwise_episode_batch(self, batch: List[dict]):
        # 1. initialize trajectories
        trajectories: List[TrajectoryInfo] = []
        print(f"Initializing {len(batch) * self.traj_rollout_n} trajectories...")
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            for i, data in enumerate(batch):
                for traj_idx in range(self.traj_rollout_n):
                    futures.append(
                        executor.submit(
                            self._initialize_trajectory,
                            data_idx=i,
                            data=data,
                            traj_rollout_idx=traj_idx,
                        )
                    )
            for future in futures:
                trajectories.append(future.result())
        print("All trajectories initialized.")
        # 2. Loop through steps until all trajectories are done
        batch_results = self._run_batch_stepwise_rollout(trajectories)
        
        final_results = []
        for i, data in enumerate(batch):
            trajs = [traj for traj in trajectories if traj.idx_in_batch == i]
            traj = trajs[0]
            all_rollouts = batch_results[i]
            all_accuracies = [
                1 if r["rollout_results"]["success"] else 0
                for r in all_rollouts
            ]
            avg_accuracy = (
                sum(all_accuracies) / len(all_accuracies)
                if all_accuracies
                else 0.0
            )
            pass_at_n = any(all_accuracies)

            final_results.append(
                {
                    "query": traj.ctx_manager.instruction_prompt,
                    "seed": data.get("seed", None),
                    "all_accuracies": all_accuracies,
                    "avg_accuracy": avg_accuracy,
                    "pass_at_n": pass_at_n,
                    "traj_rollouts": all_rollouts,
                    "meta": {
                        "traj_rollout_n": self.traj_rollout_n,
                        "step_rollout_n": self.step_rollout_n,
                    },
                }
            )

        return final_results

    def _run_batch_stepwise_rollout(self, trajectories: List[TrajectoryInfo]) -> List[Dict[str, Any]]:
        for step_idx in range(self.max_steps):
            active_trajectories = [
                traj for traj in trajectories if not traj.done
            ]
            if not active_trajectories:
                break

            # 3. Prepare history for each active trajectory
            for traj in active_trajectories:
                self._prepare_step_history(traj)

            # Check prompt length
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

        # 6. Release env, assemble final results
        for traj in trajectories:
            self._release_env_to_pool(getattr(traj, 'env_idx', -1))
        batch_results = [[] for _ in range(self.batch_size)]
        for traj in trajectories:
            loop_info = traj.ctx_manager.check_loop()
            rollout_result = {
                "steps": traj.steps,
                "success": traj.success,
                "stop_right": traj.stop_right,
                "loop": loop_info,
                "meta": {"total_steps": len(traj.steps)},
            }
            rollout_info = {
                "rollout_idx": traj.traj_rollout_idx,
                "rollout_results": rollout_result,
            }
            batch_results[traj.idx_in_batch].append(rollout_info)
        return batch_results

if __name__ == "__main__":
    pass