import pandas as pd
from typing import Any, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
        "chat_format": row["chat_format"],
        "enable_thinking": f"enable_thinking={'T' if row['enable_thinking'] else 'F'}",
        "history_has_cot": f"history_has_cot={'T' if row['history_has_cot'] else 'F'}",
        "state": f"state={row['state']}",
        "offer_feedback": f"offer_feedback={'T' if row['offer_feedback'] else 'F'}",
    }
    if row["chat_format"] == "user_assistant_format_part":
        config_mapping["history_window_size"] = (
            f"history_window_size={row['history_window_size']}"
        )
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
        exclude_keys = (
            set(selection_criteria.keys()) if selection_criteria else set()
        )

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
    exclude_keys = (
        set(selection_criteria.keys()) if selection_criteria else set()
    )

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


def load_right_data(data: list[dict[dict[str, any]]]) -> pd.DataFrame:
    records = []
    for i, sample in enumerate(data):
        # get first right trajectory
        selected_rollout = sample["traj_rollouts"][0]["rollout_results"]
        for rollout in sample["traj_rollouts"]:
            if rollout["rollout_results"]["success"] == True:
                selected_rollout = rollout["rollout_results"]
                break
        general_loop_log = selected_rollout["loop"]["general_loop_log"]
        specific_loop_log = selected_rollout["loop"]["specific_loop_log"]
        specific_loop_steps = [item["step"] for item in specific_loop_log]
        for step_idx, step in enumerate(selected_rollout["steps"]):

            record = {
                "sample_idx": i,
                "query": sample.get("query"),
                "success": selected_rollout.get("success"),
                "avg_accuracy": sample.get("avg_accuracy"),
                "step_idx": step_idx,
                "analysis": step.get("analysis", None),
                "action": step.get("action"),
                "analysis_token_entropy": step["token_entropy_stats"]
                .get("analysis_stats", {})
                .get("raw", None),
                "analysis_avg_entropy": step["token_entropy_stats"]
                .get("analysis_stats", {})
                .get("mean", None),
                "action_token_entropy": step["token_entropy_stats"]
                .get("action_stats", {})
                .get("raw", None),
                "action_avg_entropy": step["token_entropy_stats"]
                .get("action_stats", {})
                .get("mean", None),
                "action_space_entropy": step.get("action_space_entropy"),
                # "general_loop": True if step_idx in general_loop_log else False,
                "specific_loop": (
                    True if step_idx in specific_loop_steps else False
                ),
                # "action_executability": step.get("action_executability", None),
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df


def load_all_data(data: list[dict[dict[str, any]]]) -> pd.DataFrame:
    """
    Load all trajectory data into a DataFrame.
    One sample contains multiple trajectories, and we log all steps from all trajectories.
    """
    records = []
    for i, sample in enumerate(data):
        # for traj_idx, rollout in enumerate(sample["traj_rollouts"]):
        for traj_idx, rollout in enumerate(sample["traj_rollouts"][:1]):
            selected_rollout = rollout["rollout_results"]
            specific_loop_log = selected_rollout["loop"]
            specific_loop_steps = [item["step"] for item in specific_loop_log]
            for step_idx, step in enumerate(selected_rollout["steps"]):
                # if step["action"].strip().lower() == "stop" and step_idx == len(selected_rollout["steps"]) - 1:
                #     continue
                record = {
                    "sample_idx": i,
                    "seed": sample.get("seed", None),
                    "traj_idx": traj_idx,
                    "query": sample.get("query"),
                    "success": selected_rollout.get("success"),
                    "avg_accuracy": sample.get("avg_accuracy"),
                    "step_idx": step_idx,
                    "observation": step.get("observation", None),
                    "analysis": step.get("analysis", None),
                    "action": step.get("action"),
                    "stop_right": selected_rollout.get("stop_right", False),
                    # "general_loop": True if step_idx in general_loop_log else False,
                    "specific_loop": (
                        True if step_idx in specific_loop_steps else False
                    ),
                    "action_is_valid": step["env_feedback"]["action_is_valid"],
                }
                if step.get("is_random_injected", False) == False:
                    entropy_log = {
                        "is_random_injected": False,
                        "analysis_token_entropy": step["token_entropy_stats"]
                        .get("analysis_stats", {})
                        .get("raw", None),
                        "analysis_avg_entropy": step["token_entropy_stats"]
                        .get("analysis_stats", {})
                        .get("mean", None),
                        "action_token_entropy": step["token_entropy_stats"]
                        .get("action_stats", {})
                        .get("raw", None),
                        "action_avg_entropy": step["token_entropy_stats"]
                        .get("action_stats", {})
                        .get("mean", None),
                        "action_space_entropy": step.get(
                            "action_space_entropy", None
                        ),
                    }
                else:
                    entropy_log = {
                        "is_random_injected": True,
                        "analysis_token_entropy": None,
                        "analysis_avg_entropy": None,
                        "action_token_entropy": None,
                        "action_avg_entropy": None,
                        "action_space_entropy": None,
                    }
                record.update(entropy_log)
                records.append(record)

    df = pd.DataFrame(records)
    return df


def identify_error_atomic_actions(group):
    """
    Identify error atomic actions and loop situations
    Returns marking information for each step
    """
    obs_list = group["observation"].tolist()
    act_list = group["action"].tolist()
    n = len(obs_list)

    # Initialize markers
    is_error_atomic = [False] * n  # Whether it is the starting point of an error atomic action
    is_looped = [False] * n  # Whether this error atomic action is looped
    error_atomic_id = [-1] * n  # Mark which error atomic action it belongs to
    is_repeating = [False] * n  # Mark whether it is a repeating step (new)

    i = 0
    error_id = 0

    while i < n:
        # Find cycle starting from position i
        start_obs = obs_list[i]

        # Find the next position where obs == start_obs
        found_cycle = False
        cycle_end = -1

        for j in range(i + 1, n):
            if obs_list[j] == start_obs:
                # Found position returning to start
                # Check non-recursive constraint: no repeated states in the path
                # i.e., elements in obs_list[i:j] must be unique
                if len(set(obs_list[i:j])) == (j - i):
                    found_cycle = True
                    cycle_end = j
                # Regardless of constraint satisfaction, stop searching for further start points once returning to start
                break

        if found_cycle:
            # Found an error atomic action: [i, cycle_end)
            # Mark this error atomic action
            for k in range(i, cycle_end):
                error_atomic_id[k] = error_id

            is_error_atomic[i] = True  # Mark starting point

            # Check if next atomic action is same as current (cycle detection)
            # Need to compare [i, cycle_end) and [cycle_end, cycle_end + (cycle_end - i))
            cycle_length = cycle_end - i

            # Check if there are enough subsequent steps to determine cycle
            if cycle_end + cycle_length <= n:
                # Compare if two atomic actions are identical
                current_obs = obs_list[i:cycle_end]
                next_obs = obs_list[cycle_end : cycle_end + cycle_length]
                current_act = act_list[i:cycle_end]
                next_act = act_list[cycle_end : cycle_end + cycle_length]

                if current_obs == next_obs and current_act == next_act:
                    is_looped[i] = True  # Mark this error atomic action as looped
                    # Mark repeating parts (newly added)
                    for k in range(cycle_end, cycle_end + cycle_length):
                        is_repeating[k] = True

            error_id += 1
            i = cycle_end  # Move to end of cycle
        else:
            # No cycle found, continue to next step
            i += 1

    return pd.DataFrame(
        {
            "is_error_atomic": is_error_atomic,
            "is_looped": is_looped,
            "is_repeating": is_repeating,
            "error_atomic_id": error_atomic_id,
        }
    )


def analyze_exp_df(df: pd.DataFrame) -> pd.DataFrame:
    # cal entropy
    mean_analysis_avg_entropy = (
        df.groupby(["sample_idx", "traj_idx"])["analysis_avg_entropy"]
        .mean()
        .mean()
    )
    mean_action_avg_entropy = (
        df.groupby(["sample_idx", "traj_idx"])["action_avg_entropy"]
        .mean()
        .mean()
    )
    mean_action_space_entropy = (
        df.groupby("sample_idx")["action_space_entropy"].mean().mean()
    )

    # cal pass
    df_traj = df.drop_duplicates(subset=["sample_idx", "traj_idx"])

    sample_success = df_traj.groupby("sample_idx")["success"].agg(
        pass_k="max", pass_mean="mean"
    )

    pass_k = sample_success["pass_k"].mean()
    pass_mean = sample_success["pass_mean"].mean()
    # Count proportions of four traj scenarios
    # Need to find the last step's action for each traj
    df_traj = df.drop_duplicates(subset=["sample_idx", "traj_idx"])
    df_last_step = df.loc[
        df.groupby(["sample_idx", "traj_idx"])["step_idx"].idxmax()
    ]

    # Only take the last step's action, rename to avoid conflicts
    df_last_action = df_last_step[["sample_idx", "traj_idx", "action"]].rename(
        columns={"action": "last_action"}
    )

    # Merge traj info with last step's action
    df_traj_with_last_action = df_traj.merge(
        df_last_action, on=["sample_idx", "traj_idx"], how="left"
    )

    # Check if last step is stop
    df_traj_with_last_action["has_stop_action"] = (
        df_traj_with_last_action["last_action"].str.strip().str.lower()
        == "stop"
    )

    total_trajs = len(df_traj_with_last_action)

    # Case 1: Success and correct stop (success=True, stop_right=True)
    success_and_stop_right = df_traj_with_last_action[
        (df_traj_with_last_action["success"] == True)
        & (df_traj_with_last_action["stop_right"] == True)
    ].shape[0]

    # Case 2: Success but incorrect stop (success=True, stop_right=False)
    success_but_no_stop_right = df_traj_with_last_action[
        (df_traj_with_last_action["success"] == True)
        & (df_traj_with_last_action["stop_right"] == False)
    ].shape[0]

    # Case 3: Failure with stop action (incorrect early stop: success=False, has_stop_action=True)
    fail_with_stop = df_traj_with_last_action[
        (df_traj_with_last_action["success"] == False)
        & (df_traj_with_last_action["has_stop_action"] == True)
    ].shape[0]

    # Case 4: Failure without stop action (exceeded limit: success=False, has_stop_action=False)
    fail_without_stop = df_traj_with_last_action[
        (df_traj_with_last_action["success"] == False)
        & (df_traj_with_last_action["has_stop_action"] == False)
    ].shape[0]

    # Calculate proportions
    traj_stop_category_ratios = {
        "success_with_stop": (
            success_and_stop_right / total_trajs if total_trajs > 0 else 0
        ),
        "success_without_stop": (
            success_but_no_stop_right / total_trajs if total_trajs > 0 else 0
        ),
        "fail_with_stop": (
            fail_with_stop / total_trajs if total_trajs > 0 else 0
        ),
        "fail_without_stop": (
            fail_without_stop / total_trajs if total_trajs > 0 else 0
        ),
    }
    # Cal Loop Ratio After Invalid Step
    df_sorted = df.sort_values(
        ["sample_idx", "traj_idx", "step_idx"]
    ).reset_index(drop=True)

    # Apply identification function to each trajectory
    result_list = []
    for (sample_idx, traj_idx), group in df_sorted.groupby(
        ["sample_idx", "traj_idx"]
    ):
        group = group.reset_index(drop=True)
        marks = identify_error_atomic_actions(group)
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

    step_level_loop_ratio = total_numerator_steps / (
        total_denominator_steps + 1e-8
    )

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
    traj_level_loop_ratio = (
        valid_trajs["ratio"].mean() if len(valid_trajs) > 0 else 0
    )
    mean_entropy_stats = df_sorted.groupby("is_repeating")[
        ["action_avg_entropy", "analysis_avg_entropy"]
    ].mean()
    # Rename index
    mean_normal_step_action_entropy = (
        mean_entropy_stats.loc[False, "action_avg_entropy"]
        if False in mean_entropy_stats.index
        else 0
    )
    mean_loop_step_action_entropy = (
        mean_entropy_stats.loc[True, "action_avg_entropy"]
        if True in mean_entropy_stats.index
        else 0
    )
    mean_normal_step_analysis_entropy = (
        mean_entropy_stats.loc[False, "analysis_avg_entropy"]
        if False in mean_entropy_stats.index
        else 0
    )
    mean_loop_step_analysis_entropy = (
        mean_entropy_stats.loc[True, "analysis_avg_entropy"]
        if True in mean_entropy_stats.index
        else 0
    )

    # cal loop stats
    loop_counts = (
        df.groupby(["sample_idx", "traj_idx"])["specific_loop"]
        .sum()
        .rename("loop_count")
    )

    traj_success = df.drop_duplicates(
        subset=["sample_idx", "traj_idx"]
    ).set_index(["sample_idx", "traj_idx"])["success"]

    traj_stats_by_count = pd.concat(
        [loop_counts, traj_success], axis=1
    ).dropna()
    success_by_loop = traj_stats_by_count.groupby("loop_count")[
        "success"
    ].mean()
    # Fill missing loop_count, set to 0
    if not success_by_loop.empty:
        max_loop = int(success_by_loop.index.max())
        full_index = pd.RangeIndex(
            start=0, stop=max_loop + 1, name="loop_count"
        )
        success_by_loop = success_by_loop.reindex(
            full_index, fill_value=0.0
        ).sort_index()
    success_by_loop: Dict[int, float] = success_by_loop.to_dict()

    loop_stats = df.groupby(["sample_idx", "traj_idx"])["specific_loop"].agg(
        loop_actions="sum", total_actions="count"
    )
    loop_stats["loop_ratio"] = (
        loop_stats["loop_actions"] / loop_stats["total_actions"]
    )
    mean_loop_ratio_per_traj = loop_stats["loop_ratio"].mean()
    traj_stats_by_ratio = loop_stats.join(traj_success, how="inner").dropna(
        subset=["loop_ratio", "success"]
    )

    ratio_bins = np.linspace(0, 1, 21)
    traj_stats_by_ratio["loop_ratio_bin"] = pd.cut(
        traj_stats_by_ratio["loop_ratio"],
        bins=ratio_bins,
        include_lowest=True,
        labels=[
            f"{int(left * 100)}%-{int(right * 100)}%"
            for left, right in zip(ratio_bins[:-1], ratio_bins[1:])
        ],
    )

    success_by_ratio = (
        traj_stats_by_ratio.groupby("loop_ratio_bin", observed=False)["success"]
        .mean()
        .fillna(0.0)
    )
    success_by_loop_ratio: Dict[str, float] = success_by_ratio.to_dict()
    mean_loop_per_traj = loop_counts.mean()

    # cal pass@1 @ T (Turns)
    traj_summary = df.groupby(["sample_idx", "traj_idx"]).agg(
        max_step=("step_idx", "max"), traj_success=("success", "max")
    )
    traj_summary["traj_length"] = traj_summary["max_step"] + 1
    traj_summary["first_success_step"] = np.where(
        traj_summary["traj_success"].astype(bool),
        traj_summary["traj_length"],
        np.inf,
    )

    max_T = int(traj_summary["traj_length"].max())
    pass1_at_T = pd.Series(
        [
            (traj_summary["first_success_step"] <= T).mean()
            for T in range(1, max_T + 1)
        ],
        index=range(1, max_T + 1),
        name="pass@1",
    )
    pass1_at_T: Dict[int, float] = pass1_at_T.to_dict()
    # cal pass@k @ T (Turns)
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
        "mean_analysis_avg_entropy": mean_analysis_avg_entropy,
        "mean_action_avg_entropy": mean_action_avg_entropy,
        "mean_action_space_entropy": mean_action_space_entropy,
        "pass_k": pass_k,
        "pass_mean": pass_mean,
        **traj_stop_category_ratios,
        "mean_loop": mean_loop_per_traj,
        "mean_loop_ratio": mean_loop_ratio_per_traj,
        # error atomic action loop ratio
        "mean_loop_ratio_after_invalid_steps_stepnorm": step_level_loop_ratio,
        "mean_loop_ratio_after_invalid_steps_trajnorm": traj_level_loop_ratio,
        # entropy of reapted error atomic actions
        "mean_normal_step_action_entropy": mean_normal_step_action_entropy,
        "mean_loop_step_action_entropy": mean_loop_step_action_entropy,
        "mean_normal_step_analysis_entropy": mean_normal_step_analysis_entropy,
        "mean_loop_step_analysis_entropy": mean_loop_step_analysis_entropy,
        "success_by_loop": success_by_loop,
        "success_by_loop_ratio": success_by_loop_ratio,
        "pass1_at_T": pass1_at_T,
        "passk_at_T": passk_at_T,
    }


def get_config(exp_dir: str) -> Dict[str, Any]:
    from omegaconf import OmegaConf
    import os

    config_path = os.path.join(exp_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = OmegaConf.load(config_path)

    agent_proxy_config = config.get("agent_proxy", {})
    model_name = None
    if agent_proxy_config.get("type") == "vllm":
        model_name = (
            config.get("vllm_agent", {}).get("model_path", "").split("/")[-1]
        )
    elif agent_proxy_config.get("type") == "client":
        model_name = config.get("client_agent", {}).get("model_name")
    additional_exp = config.get("additional_exp", {})
    config_data = {
        "model_name": model_name,
        "task": config.get("task"),
        "data_path": config.get("data_path"),
        "alfworld_mode": config.env.alfworld.eval_mode,
        "webshop_mode": config.env.webshop.eval_mode,
        "max_steps": agent_proxy_config.max_steps,
        "chat_format": agent_proxy_config.get("chat_format"),
        "prompt_example": agent_proxy_config.get("prompt_example", "fewshot"),
        "history_window_size": agent_proxy_config.get("history_window_size"),
        "enable_thinking": agent_proxy_config.get("enable_thinking"),
        "stop_on_error": agent_proxy_config.get("stop_on_error"),
        "stop_by_self": additional_exp.get("stop_by_self", {}).get(
            "enable", False
        ),
        "offer_feedback": agent_proxy_config.get("offer_feedback"),
        "history_has_cot": agent_proxy_config.get("history_has_cot"),
        "state": agent_proxy_config.get("state"),
        "random_step_injection": additional_exp.get("random_step", {}).get(
            "enable", False
        ),
        "random_step_num": additional_exp.get("random_step", {}).get("num", 0),
    }
    return config_data


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
        # Use thread pool for IO-intensive and CPU-intensive tasks that release GIL
        with ThreadPool(n_workers) as pool:
            data = [x for x in pool.map(process_line, lines) if x is not None]
        return data

    def process_experiment_folder(exp_dir, rewrite):
        try:
            print(f"Starting processing: {exp_dir}")
            result_path = os.path.join(exp_dir, "result.json")

            if not rewrite and os.path.exists(result_path):
                print(f"Skipping (result.json already exists): {exp_dir}")
                return

            # Find jsonl files
            jsonl_files = [
                f for f in os.listdir(exp_dir) if f.endswith(".jsonl")
            ]
            if not jsonl_files:
                print(f"Warning: No .jsonl file found in {exp_dir}")
                # shutil.rmtree(exp_dir)
                # print(f"Deleted folder: {exp_dir}")
                return
            jsonl_path = os.path.join(exp_dir, jsonl_files[0])

            # Load and analyze data
            raw_data = load_jsonl_parallel(jsonl_path)
            df = load_all_data(raw_data)
            analysis_results = analyze_exp_df(df)

            # Read config.yaml
            config_path = os.path.join(exp_dir, "config.yaml")
            if not os.path.exists(config_path):
                print(f"Warning: config.yaml not found in {exp_dir}")
                return

            config = OmegaConf.load(config_path)

            # Extract configuration information
            config_data = get_config(exp_dir)

            # Merge results and save
            final_result = {**config_data, **analysis_results}
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=4)

            print(f"Successfully saved result.json to: {exp_dir}")

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
        pool.starmap(
            process_experiment_folder, [(d, args.rewrite) for d in exp_dirs]
        )

    print("All processing tasks completed.")
