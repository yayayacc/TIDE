from omegaconf import DictConfig, OmegaConf
import hydra
from utils.llm_agent.client_agent import ClientAgent
from utils.llm_agent.vllm_agent import VLLMAgent
from typing import Any, Dict, List
import time
import pandas as pd
import json
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# from vllm import LLM


def list_all_gpu_indices() -> str:
    """Return a list of all visible GPU indices on the machine."""
    # Fallback to PyTorch if available
    try:
        import torch

        n = torch.cuda.device_count()
        if n and n > 0:
            return ",".join([str(i) for i in range(n)])
    except Exception:
        pass
    # Final fallback
    return "0"


def gen_output_dir(config: DictConfig) -> None:
    if config.agent_proxy.type == "vllm":
        model_name = config.vllm_agent.model_path.split("/")[-1]
    elif config.agent_proxy.type == "client":
        model_name = config.client_agent.model_name
    date_str = time.strftime("%Y%m%d")
    chat_format = config.agent_proxy.chat_format
    file_name = f"{model_name}_{chat_format}"
    if config.agent_proxy.enable_thinking:
        file_name += "_think"
    if config.additional_exp.stop_by_self.enable:
        file_name += "_stopBySelf"
    file_name += f"_{date_str}"
    candidate = config.out_dir + f"/{config.task}/{file_name}"
    id = 0
    unique = candidate
    while os.path.exists(unique):
        id += 1
        unique = f"{candidate}_{id:03d}"
    config.out_dir = unique
    os.makedirs(config.out_dir, exist_ok=True)
    # Save configuration file
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))


def get_task_runner(
    config: DictConfig, agent: VLLMAgent | ClientAgent, dataset=None
) -> Any:
    if config.task == "webshop":
        from utils.TaskRunner.WebShopRunner import WebShopRunner

        return WebShopRunner(config, agent, dataset)
    elif config.task == "math":
        from utils.TaskRunner.MathRunner import MathRunner

        return MathRunner(config, agent, dataset)
    elif config.task == "frozen_lake":
        from utils.TaskRunner.FrozenLakeRunner import FrozenLakeRunner

        return FrozenLakeRunner(config, agent, dataset)
    elif config.task == "alfworld":
        from utils.TaskRunner.AlfWorldRunner import AlfWorldRunner

        return AlfWorldRunner(config, agent, dataset)
    elif config.task == "blocksworld":
        from utils.TaskRunner.BlocksWorldRunner import BlocksWorldRunner

        return BlocksWorldRunner(config, agent, dataset)
    elif config.task == "sudoku":
        from utils.TaskRunner.SudokuRunner import SudokuRunner

        return SudokuRunner(config, agent, dataset)
    else:
        raise ValueError(f"Unknown task name: {config.task}")


def run_parallel_tasks(
    config: DictConfig,
    dp_idx,
    aim_gpus: str,
    dataset: List[Dict[str, Any]],
    file_lock,
    barrier,
) -> None:
    """Run tasks in parallel across multiple GPUs"""
    if config.agent_proxy.type == "client":
        agent = ClientAgent(config)
    else:
        agent = VLLMAgent(config, aim_gpus)
    runner = get_task_runner(config=config, agent=agent, dataset=dataset)
    runner.run(dp_idx, lock=file_lock)

    # Wait for all processes to complete their tasks
    print(f"Worker {dp_idx} waiting at barrier...")
    barrier.wait()

    print(f"Worker {dp_idx} releasing resources...")
    agent.close()
    print(f"Worker {dp_idx} finished processing.")


def check_config(config: DictConfig) -> None:
    """Check parameter validity"""
    # if config.task in ["alfworld", "webshop"]:
    #     if config.agent_proxy.state != "env":
    #         raise ValueError(
    #             f"For AlfWorld and WebShop tasks, the state must be 'env', but got '{config.agent_proxy.state}'."
    #         )

    if (
        config.agent_proxy.enable_thinking
        and config.agent_proxy.chat_format == "default_format"
    ):
        raise ValueError(
            "The 'default_format' chat format does not support thinking"
        )

    if config.agent_proxy.offer_feedback and config.agent_proxy.state != "env":
        print(
            "Warning: Offering feedback is only supported in 'env' state. Disabling feedback."
        )
        config.agent_proxy.offer_feedback = False


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)

    # check config
    check_config(config)
    print("Effective configuration:\n", OmegaConf.to_yaml(config))
    config.vllm_agent.aim_gpus = (
        list_all_gpu_indices()
        if config.vllm_agent.aim_gpus == "-1"
        else config.vllm_agent.aim_gpus
    )
    gen_output_dir(config)
    dataset = json.load(open(config.data_path, "r", encoding="utf-8"))
    # random example
    import random

    # dataset = random.sample(dataset, 400)

    dp_size = config.agent_proxy.dp_size
    if dp_size == 1:
        start_time = time.time()

        if config.agent_proxy.type == "client":
            agent = ClientAgent(config)
        else:
            agent = VLLMAgent(config, config.vllm_agent.aim_gpus)
        runner = get_task_runner(config, agent, dataset)
        runner.run()
        end_time = time.time()
        agent.close()
        print(
            f"Total inference time: {(end_time - start_time) / 60:.2f} minutes"
        )
    else:
        from multiprocessing import Process, Lock, Barrier

        # 4. Check if GPU count matches, determine if data parallelism is needed
        if config.agent_proxy.type != "client":
            gpu_ids = [
                gpu.strip()
                for gpu in config.vllm_agent.aim_gpus.split(",")
                if gpu.strip()
            ]
            num_available_gpus = len(gpu_ids)
            gpus_per_replica = config.vllm_agent.tp_size * config.vllm_agent.pp_size
            required_gpus = gpus_per_replica * dp_size
            if num_available_gpus != required_gpus:
                raise ValueError(
                    f"GPU count mismatch: requires {required_gpus} GPUs (tp={config.vllm_agent.tp_size} * pp={config.vllm_agent.pp_size} * dp={dp_size}), "
                    f"but {num_available_gpus} GPUs provided (aim_gpus='{config.vllm_agent.aim_gpus}')."
                )
            print(
                f"Data parallelism enabled with dp={dp_size}, launching multiple instances..."
            )
            replica_gpu_lists = []
            for i in range(dp_size):
                start_idx = i * gpus_per_replica
                end_idx = start_idx + gpus_per_replica
                replica_gpus = gpu_ids[start_idx:end_idx]
                replica_gpu_lists.append(",".join(replica_gpus))
        else:
            # Client mode does not require GPU allocation
            replica_gpu_lists = [""] * dp_size

        start_time = time.time()
        print(
            f"Starting processing of {len(dataset)} items with {dp_size} workers..."
        )
        procs = []
        file_lock = Lock()  # Lock for file writing
        barrier = Barrier(dp_size)
        # data parallelism
        for dp_idx, dp_aim_gpus in enumerate(replica_gpu_lists):
            # Allocate data subset for each data parallel instance
            data_subset = [
                {**item}
                for idx, item in enumerate(dataset)
                if idx % dp_size == dp_idx
            ]
            print(
                f"Worker {dp_idx} assigned {len(data_subset)} items on GPUs: {dp_aim_gpus}"
            )
            proc = Process(
                target=run_parallel_tasks,
                args=(
                    config,
                    dp_idx,
                    dp_aim_gpus,
                    data_subset,
                    file_lock,
                    barrier,
                ),
            )
            proc.start()
            procs.append(proc)
        exit_code = 0
        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                exit_code = proc.exitcode
        end_time = time.time()
        # Calculate total inference time (minutes)
        print(
            f"Total inference time: {(end_time - start_time) / 60:.2f} minutes"
        )
        exit(exit_code)


if __name__ == "__main__":
    main()
