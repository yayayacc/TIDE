import pandas as pd
from typing import Any, List
from torch import Tensor
import pandas as pd
import torch
from pathlib import Path
from typing import Any, List, Dict
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Define root directory for analysis results
DIR_ANALYSIS_ROOT = "./analysis_third_part"


def get_analysis_save_dir(data_dir: str, dataset: str) -> Path:
    """
    Map original data_dir to corresponding directory under analysis_third_part
    Example data_dir:
      /root/.../osworld/ui_tars_15_7b/chrome/xxxx
    Returns:
      ./analysis_third_part/osworld/ui_tars_15_7b/chrome/xxxx
    """
    data_dir = Path(data_dir)
    parts = list(data_dir.parts)

    # Try to find dataset position in path
    if dataset in parts:
        idx = parts.index(dataset)
        rel_from_dataset = Path(*parts[idx:])
    else:
        # If dataset name not found, try using last few directory levels
        # Assume structure is usually .../dataset/model/task/traj_id
        # If still not found, use last 4 levels of data_dir
        rel_from_dataset = Path(*parts[-4:])

    cache_root = Path(DIR_ANALYSIS_ROOT)
    return cache_root / rel_from_dataset


def format_criteria_for_title(selection_criteria: dict = None) -> str:
    """
    Format selection criteria as part of the title

    Args:
        selection_criteria: Selection criteria dictionary

    Returns:
        Formatted selection criteria string
    """
    if not selection_criteria:
        return ""

    criteria_parts = []
    for key, value in selection_criteria.items():
        if isinstance(value, bool):
            criteria_parts.append(f"{key}={'T' if value else 'F'}")
        elif isinstance(value, str):
            if key == "model_name":
                criteria_parts.append(f"{value}")
            else:
                criteria_parts.append(f"{key}='{value}'")
        else:
            criteria_parts.append(f"{key}={value}")

    return " | ".join(criteria_parts)


def get_config_label(row: pd.Series, exclude_keys: set = None) -> str:
    """
    Generate configuration label, excluding specified fields

    Args:
        row: Row containing configuration information
        exclude_keys: Set of field names to exclude
    """
    if exclude_keys is None:
        exclude_keys = set()

    config_mapping = {
        "model_name": row["model_name"],
    }
    # Only keep configuration items not in exclude_keys
    config_parts = [
        value
        for key, value in config_mapping.items()
        if key not in exclude_keys and value is not None
    ]

    label = "\n-".join(config_parts) if config_parts else "default"
    return label


def plot_metric_line(
    selected_df: pd.DataFrame,
    select_col: str = "success_by_loop",
    xlabel: str = "Specific Loop Count",
    ylabel: str = "Success Rate",
    title="Impact of Loop Count on Success Rate",
    selection_criteria: dict = None,  # New parameter
):
    # --- Line chart ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    if selected_df.empty:
        print("No experimental data matching the criteria found for plotting.")
    else:
        # Assign different colors to each experiment (using seaborn tab10)
        colors = sns.color_palette("tab10", n_colors=len(selected_df))

        # Get keys to exclude
        exclude_keys = set(selection_criteria.keys()
                           ) if selection_criteria else set()

        for idx, (index, row) in enumerate(selected_df.iterrows()):
            # success_by_loop column stores a dictionary
            data = row[f"{select_col}"]

            # Skip if data is empty
            if not isinstance(data, dict) or not data:
                print(
                    f"Warning: Experiment '{row['experiment']}' has empty or incorrectly formatted {select_col} data."
                )
                continue

            # Extract x (loop count) and y (success rate)
            # Sort if keys are int after JSON loading
            x_data = list(data.keys())
            y_data = [data[str(k)] for k in x_data]

            # Create legend label, excluding fields in selection criteria
            label = get_config_label(row, exclude_keys)
            # Plot curve
            ax.plot(
                x_data,
                y_data,
                # marker="o",
                linestyle="-",
                label=label,
                color=colors[idx],
            )

        # --- 4. Beautify chart ---
        # Add selection criteria to title
        criteria_str = format_criteria_for_title(selection_criteria)
        if criteria_str:
            full_title = f"{title}\n[{criteria_str}]"
        else:
            full_title = title

        ax.set_title(full_title, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        # Dynamically adjust y
        # ax.set_ylim(0, 1.05)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Add legend
        if len(selected_df) > 0:
            ax.legend(
                title="Experiment Configuration",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        plt.show()


def plot_metric_bar(
    selected_df: pd.DataFrame,
    select_cols,
    xlabel: str = "Metric",
    ylabel: str = "Value",
    title: str = "Metric Bar Comparison",
    figsize=(12, 6),
    dpi=150,
    selection_criteria: dict = None,  # New parameter
):
    """
    Plot bar chart. select_cols can be a single column name string or a list of column names.
    Each select_col is an x position, different experiments (rows in DataFrame) are plotted as side-by-side bars for comparison.
    Requirement: Each select_col value in selected_df must be numeric (or convertible to numeric).
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    if isinstance(select_cols, str):
        select_cols = [select_cols]
    elif not isinstance(select_cols, (list, tuple)):
        print("select_cols must be a string or a list of strings.")
        return

    if selected_df.empty:
        print("No experimental data matching the criteria found for plotting.")
        return

    # --- Data preparation ---
    # X-axis labels are metric names
    x_labels = select_cols
    n_metrics = len(x_labels)
    n_exps = len(selected_df)

    # Get keys to exclude
    exclude_keys = set(selection_criteria.keys()
                       ) if selection_criteria else set()

    # Extract data, shape is (n_exps, n_metrics)
    data_matrix = []
    exp_legend_labels = []
    for _, row in selected_df.iterrows():
        # Create experiment label for legend, excluding fields in selection criteria
        exp_legend_labels.append(get_config_label(row, exclude_keys))

        # Extract all metric values for this experiment
        metric_vals = []
        for col in select_cols:
            val = row.get(col, None)
            try:
                val = float(val) if val is not None else np.nan
            except (ValueError, TypeError):
                val = np.nan
            metric_vals.append(val)
        data_matrix.append(metric_vals)

    data_matrix = np.array(data_matrix)  # shape: (n_exps, n_metrics)

    # --- Plotting settings ---
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    colors = sns.color_palette("tab10", n_colors=n_exps)

    # Side-by-side bar width and position
    total_width = 0.8
    single_width = total_width / n_exps
    x = np.arange(n_metrics)  # x-axis positions

    for i in range(n_exps):  # Iterate through each experiment
        offsets = x - total_width / 2 + i * single_width + single_width / 2
        ax.bar(
            offsets,
            data_matrix[i],  # All metric values for the i-th experiment
            width=single_width * 0.9,
            label=exp_legend_labels[i],
            color=colors[i],
        )

    # --- Beautify chart ---
    # Add selection criteria to title
    criteria_str = format_criteria_for_title(selection_criteria)
    if criteria_str:
        full_title = f"{title}\n[{criteria_str}]"
    else:
        full_title = title

    ax.set_title(full_title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.legend(
        title="Experiment Configuration",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        ncol=1 if n_exps <= 4 else 2,
    )
    fig.tight_layout()
    plt.show()

    return fig, ax


def load_custom_data(data: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Load all trajectory data into a DataFrame.
    One sample contains multiple trajectories, and we log all steps from all trajectories.
    """
    records = []
    for i, sample in enumerate(data):
        for step_idx, step in enumerate(sample["trajectory"]):
            action: List[str] = step.get("action", [])
            observation_path = step.get("observation", "empty")

            # Load stored vector, mark as "empty" if empty
            if observation_path == "empty":
                observation = "empty"
            else:
                try:
                    # Load vector from .pt file
                    observation = torch.load(
                        observation_path, map_location="cpu")
                    # If shape is [1, 512], squeeze to [512]
                    if isinstance(observation, Tensor) and observation.dim() == 2:
                        observation = observation.squeeze(0)
                except Exception as e:
                    print(f"Warning: Failed to load {observation_path}: {e}")
                    observation = "empty"

            record = {
                "sample_idx": i,
                "data_dir": sample.get("data_dir", None),
                "seed": sample.get("data_dir", None),
                "traj_idx": 0,
                "success": sample.get("success"),
                "step_idx": step["step_idx"],
                "observation": observation,
                "action": action,
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df


def obs_equal(obs1, obs2, threshold=0.999):
    """
    Compare whether two observations are equal
    - If both are Tensors, use cosine similarity (threshold judgment)
    - Otherwise use strict equality
    """
    # If both are Tensors, use cosine similarity
    if isinstance(obs1, Tensor) and isinstance(obs2, Tensor):
        # Ensure 1D tensor
        if obs1.dim() > 1:
            obs1 = obs1.squeeze()
        if obs2.dim() > 1:
            obs2 = obs2.squeeze()

        # Calculate cosine similarity
        similarity = F.cosine_similarity(obs1.unsqueeze(0), obs2.unsqueeze(0))
        return similarity.item() >= threshold

    # If both are string "empty"
    elif obs1 == "empty" and obs2 == "empty":
        return True

    # Other cases (one is Tensor, one is "empty")
    else:
        return False


def identify_error_atomic_actions(group, similarity_threshold=0.999):
    """
    Identify error atomic actions and loop situations
    Returns marking information for each step
    """
    obs_list = group["observation"].tolist()
    act_list = group["action"].tolist()
    n = len(obs_list)

    # Initialize markers
    is_error_atomic = [False] * n
    is_looped = [False] * n
    error_atomic_id = [-1] * n
    is_repeating = [False] * n
    cycle_lengths = [0] * n  # New: record atomic action length

    i = 0
    error_id = 0

    while i < n:
        start_obs = obs_list[i]
        found_cycle = False
        cycle_end = -1

        # Find atomic action definition [i, j)
        for j in range(i + 1, n):
            if obs_equal(obs_list[j], start_obs, similarity_threshold):
                # Check path uniqueness
                path_obs = obs_list[i:j]
                is_unique = True
                for k1 in range(len(path_obs)):
                    for k2 in range(k1 + 1, len(path_obs)):
                        if obs_equal(path_obs[k1], path_obs[k2], similarity_threshold):
                            is_unique = False
                            break
                    if not is_unique:
                        break

                if is_unique:
                    found_cycle = True
                    cycle_end = j
                break

        if found_cycle:
            cycle_len = cycle_end - i
            cycle_lengths[i] = cycle_len

            # Mark prototype
            for k in range(i, cycle_end):
                error_atomic_id[k] = error_id
            is_error_atomic[i] = True

            # Greedily match subsequent loops
            current_idx = cycle_end

            while current_idx + cycle_len <= n:
                # Compare [i, cycle_end) and [current_idx, current_idx + cycle_len)
                current_obs = obs_list[i:cycle_end]
                next_obs = obs_list[current_idx: current_idx + cycle_len]
                current_act = act_list[i:cycle_end]
                next_act = act_list[current_idx: current_idx + cycle_len]

                # Check Obs
                obs_match = True
                for o1, o2 in zip(current_obs, next_obs):
                    if not obs_equal(o1, o2, similarity_threshold):
                        obs_match = False
                        break

                # Check Act
                act_match = (current_act == next_act)

                if obs_match and act_match:
                    is_looped[i] = True
                    # Mark repeating parts, assign same error_id
                    for k in range(current_idx, current_idx + cycle_len):
                        is_repeating[k] = True
                        error_atomic_id[k] = error_id
                    current_idx += cycle_len
                else:
                    break

            error_id += 1
            i = current_idx  # Skip all matched loops
        else:
            i += 1

    return pd.DataFrame(
        {
            "is_error_atomic": is_error_atomic,
            "is_looped": is_looped,
            "is_repeating": is_repeating,
            "error_atomic_id": error_atomic_id,
            "cycle_length": cycle_lengths,
        }
    )


def extract_error_details(group: pd.DataFrame, marks: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Extract detailed error atomic action information from marking results
    Only record atomic actions that have been looped (is_looped=True)
    """
    error_details = []
    unique_error_ids = sorted(list(set(marks["error_atomic_id"]) - {-1}))

    for error_id in unique_error_ids:
        error_rows = marks[marks["error_atomic_id"] == error_id]
        if error_rows.empty:
            continue

        if not error_rows["is_looped"].any():
            continue

        # Find definition starting point
        start_indices = error_rows[error_rows["is_error_atomic"]].index.tolist(
        )
        if not start_indices:
            continue
        first_start_idx = start_indices[0]

        # Get length
        length = int(marks.loc[first_start_idx, "cycle_length"])
        if length <= 0:
            continue

        # Get action sequence
        action_list = group.loc[first_start_idx: first_start_idx +
                                length - 1, "action"].tolist()

        # Get all instances
        # error_rows index is continuous blocks (prototype + loop1 + loop2 ...)
        all_indices = error_rows.index.tolist()

        loop_instances = []
        # Split by length
        for k in range(0, len(all_indices), length):
            chunk = all_indices[k: k + length]
            if len(chunk) == length:
                loop_instances.append(group.loc[chunk, "step_idx"].tolist())

        error_details.append({
            "error_id": int(error_id),
            "action_sequence": action_list,
            "occurrence_count": len(loop_instances),
            "loop_instances_step_ids": loop_instances
        })
    return error_details


def analyze_exp_df(df: pd.DataFrame, dataset_name: str = "unknown") -> pd.DataFrame:
    # Calculate pass
    df_traj = df.drop_duplicates(subset=["sample_idx", "traj_idx"])

    sample_success = df_traj.groupby("sample_idx")["success"].agg(
        pass_k="max", pass_mean="mean"
    )

    pass_k = sample_success["pass_k"].mean()
    pass_mean = sample_success["pass_mean"].mean()

    # Calculate Loop Ratio After Invalid Step
    df_sorted = df.sort_values(["sample_idx", "traj_idx", "step_idx"]).reset_index(
        drop=True
    )

    # Apply identification function to each trajectory
    result_list = []
    total_traj_count = 0
    looped_traj_count = 0
    # Iterate through each trajectory for analysis and saving
    for (sample_idx, traj_idx), group in df_sorted.groupby(["sample_idx", "traj_idx"]):
        total_traj_count += 1
        group = group.reset_index(drop=True)
        marks = identify_error_atomic_actions(group)
        if marks["is_looped"].any():
            looped_traj_count += 1
        # --- New: Extract detailed error information and save ---
        traj_error_details = extract_error_details(group, marks)

        # Get original data directory for this trajectory
        data_dir = group.iloc[0]["data_dir"]
        success = group.iloc[0]["success"]
        if data_dir:
            # Calculate save path
            save_dir = get_analysis_save_dir(data_dir, dataset_name)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Build analysis result object
            analysis_data = {
                "sample_idx": int(sample_idx),
                "traj_idx": int(traj_idx),
                "data_dir": str(data_dir),
                "success": float(success),
                "error_atomic_actions": traj_error_details
            }

            # Save to folder corresponding to this trajectory
            analysis_path = save_dir / "analysis.json"
            try:
                with open(analysis_path, "w", encoding="utf-8") as f:
                    json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Failed to save analysis.json to {analysis_path}: {e}")
        # -------------------------------------------

        marks["sample_idx"] = sample_idx
        marks["traj_idx"] = traj_idx
        marks["step_idx"] = group["step_idx"].values
        result_list.append(marks)

    marks_df = pd.concat(result_list, ignore_index=True)

    # Merge back to original data
    df_sorted = df_sorted.merge(
        marks_df, on=["sample_idx", "traj_idx", "step_idx"], how="left"
    )
    total_denominator_steps = len(df_sorted)
    total_numerator_steps = df_sorted["is_repeating"].sum()

    step_level_loop_ratio = total_numerator_steps / \
        (total_denominator_steps + 1e-8)

    traj_stats = (
        df_sorted.groupby(["sample_idx", "traj_idx"])
        .agg(
            denominator_count=("step_idx", "count"),
            numerator_count=("is_repeating", "sum"),
        )
        .reset_index()
    )

    # Filter out trajectories with denominator 0
    valid_trajs = traj_stats[traj_stats["denominator_count"] > 0].copy()

    # Calculate ratio for each traj
    valid_trajs["ratio"] = (
        valid_trajs["numerator_count"] / valid_trajs["denominator_count"]
    )

    # Calculate average ratio (in units of traj)
    traj_level_loop_ratio = valid_trajs["ratio"].mean() if len(
        valid_trajs) > 0 else 0

    # Calculate pass@1 @ T (Turns)
    traj_summary = df.groupby(["sample_idx", "traj_idx"]).agg(
        max_step=("step_idx", "max"),
        traj_success=("success", "max")  # Keep original success value
    )
    traj_summary["traj_length"] = traj_summary["max_step"] + 1
    traj_summary["first_success_step"] = np.where(
        traj_summary["traj_success"].astype(bool),
        traj_summary["traj_length"],
        np.inf,
    )
    # New: Mark whether each trajectory has loop
    traj_loop_status = marks_df.groupby(["sample_idx", "traj_idx"])[
        "is_looped"].any()
    traj_summary = traj_summary.merge(
        traj_loop_status.rename("has_loop"),
        left_index=True,
        right_index=True,
        how="left"
    )
    traj_summary["has_loop"] = traj_summary["has_loop"].fillna(False)
    max_T = int(traj_summary["traj_length"].max())
    pass1_at_T = pd.Series(
        [traj_summary.apply(
            lambda row: row["traj_success"] if row["traj_length"] <= T else 0.0,
            axis=1
        ).mean()
            for T in range(1, max_T + 1)],
        index=range(1, max_T + 1),
        name="pass@1",
    )
    pass1_at_T: Dict[int, float] = pass1_at_T.to_dict()

    # New: pass@1 @ T for trajectories with loop
    looped_trajs = traj_summary[traj_summary["has_loop"]]
    pass1_at_T_looped = pd.Series(
        [looped_trajs.apply(
            lambda row: row["traj_success"] if row["traj_length"] <= T else 0.0,
            axis=1
        ).mean() if len(looped_trajs) > 0 else 0.0
            for T in range(1, max_T + 1)],
        index=range(1, max_T + 1),
        name="pass@1_looped",
    )
    pass1_at_T_looped: Dict[int, float] = pass1_at_T_looped.to_dict()

    # New: pass@1 @ T for trajectories without loop
    non_looped_trajs = traj_summary[~traj_summary["has_loop"]]
    pass1_at_T_non_looped = pd.Series(
        [non_looped_trajs.apply(
            lambda row: row["traj_success"] if row["traj_length"] <= T else 0.0,
            axis=1
        ).mean() if len(non_looped_trajs) > 0 else 0.0
            for T in range(1, max_T + 1)],
        index=range(1, max_T + 1),
        name="pass@1_non_looped",
    )
    pass1_at_T_non_looped: Dict[int, float] = pass1_at_T_non_looped.to_dict()

    # Calculate pass@k @ T (Turns)
    sample_first_success = traj_summary.groupby("sample_idx")[
        "first_success_step"
    ].min()

    passk_at_T = pd.Series(
        [(sample_first_success <= T).mean() for T in range(1, max_T + 1)],
        index=range(1, max_T + 1),
        name="pass@k",
    )
    passk_at_T: Dict[int, float] = passk_at_T.to_dict()

    return {
        "total_traj_count": total_traj_count,
        "looped_traj_count": looped_traj_count,
        "pass_k": pass_k,
        "pass_mean": pass_mean,
        # error atomic action loop ratio
        "mean_loop_ratio_after_invalid_steps_stepnorm": step_level_loop_ratio,
        "mean_loop_ratio_after_invalid_steps_trajnorm": traj_level_loop_ratio,
        "pass1_at_T": pass1_at_T,
        "pass1_at_T_looped": pass1_at_T_looped,
        "pass1_at_T_non_looped": pass1_at_T_non_looped,
        "passk_at_T": passk_at_T,
    }


if __name__ == "__main__":
    import argparse
    import os
    import json
    import multiprocessing as mp
    from multiprocessing.pool import ThreadPool
    from omegaconf import OmegaConf
    import orjson
    import shutil

    def process_line(line):
        if not line.strip():
            return None
        return orjson.loads(line)

    def load_jsonl_parallel(data_path, n_workers=64):
        with open(data_path, "rb") as f:
            lines = f.readlines()
        # Use thread pool for IO-intensive and GIL-releasing CPU-intensive tasks
        with ThreadPool(n_workers) as pool:
            data = [x for x in pool.map(process_line, lines) if x is not None]
        return data

    def process_experiment_folder(exp_dir, rewrite):
        try:
            print(f"Starting processing: {exp_dir}")

            # Find jsonl files
            jsonl_files = [f for f in os.listdir(
                exp_dir) if f.endswith(".jsonl")]
            if not jsonl_files:
                print(f"Warning: No .jsonl files found in {exp_dir}")
                return
            jsonl_path = os.path.join(exp_dir, jsonl_files[0])
            # Pre-read to get dataset and model to determine save path
            with open(jsonl_path, "rb") as f:
                first_line = f.readline()
                if not first_line:
                    return
                first_sample = orjson.loads(first_line)
                dataset_name = first_sample.get("dataset", "unknown")
                model_name = first_sample.get("model", "unknown")

            # Define new save path: DIR_ANALYSIS_ROOT/dataset/model/result.json
            save_root = Path(DIR_ANALYSIS_ROOT) / dataset_name / model_name
            save_root.mkdir(parents=True, exist_ok=True)
            result_path = save_root / "result.json"

            if not rewrite and result_path.exists():
                print(f"Skipping (result.json already exists): {result_path}")
                return
            # Load and analyze data
            raw_data = load_jsonl_parallel(jsonl_path)
            df = load_custom_data(raw_data)

            # Get dataset name for path mapping
            dataset_name = raw_data[0].get("dataset", "unknown")

            # Pass dataset_name
            analysis_results = analyze_exp_df(df, dataset_name=dataset_name)

            config_data = {
                "dataset": dataset_name,
                "model_name": raw_data[0].get("model", "unknown"),
            }

            # Merge results and save
            final_result = {**config_data, **analysis_results}
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=4)

            print(f"Successfully saved result.json to: {save_root}")

        except Exception as e:
            print(f"Error processing {exp_dir}: {e}")

    parser = argparse.ArgumentParser(description="Analyze experiment data")
    parser.add_argument(
        "--rewrite",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to rewrite the analysis files",
    )
    parser.add_argument(
        "--all_exp_dir",
        type=str,
        help="Directory containing the all experiment data files",
        default="./res/alfworld",
    )
    parser.add_argument(
        "--single_exp_dir",
        type=str,
        help="Directory containing the single experiment data files",
        default=None,
    )
    args = parser.parse_args()

    if args.single_exp_dir:
        exp_dirs = [args.single_exp_dir]
    else:
        exp_dirs = [
            os.path.join(args.all_exp_dir, d)
            for d in os.listdir(args.all_exp_dir)
            if os.path.isdir(os.path.join(args.all_exp_dir, d))
        ]

    # Use multiprocessing to process folders
    with mp.Pool(processes=4) as pool:
        pool.starmap(process_experiment_folder, [
                     (d, args.rewrite) for d in exp_dirs])

    print("All processing tasks completed.")
