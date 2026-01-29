import json
import orjson
import multiprocessing as mp
from functools import partial
import pandas as pd
import numpy as np
import os
import alfworld.gen.constants as constants
import re
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from omegaconf import OmegaConf
import argparse
import sys
from utils.analysis_files.analysis import load_all_data, get_config, format_criteria_for_title, get_config_label

LOC = [obj.lower() for obj in constants.VAL_RECEPTACLE_OBJECTS if obj not in constants.MOVABLE_RECEPTACLES]

# Mapping of implicit items and locations
IMPLICIT_LOCATIONS = {
    "cool": ["fridge"],
    "heat": ["microwave"],
    "clean": ["sink", "sinkbasin"],
}

# Regular expression templates for task types
TASK_PATTERNS = [
    # Pick Two & Place (most specific, match first)
    ("Pick&2", r"put two (\w+) in (\w+)", lambda obj, loc: (obj, loc)),
    ("Pick&2", r"find two (\w+) and put them (\w+)", lambda obj, loc: (obj, loc)),
    
    # Clean & Place
    ("Clean", r"put a clean (\w+) in (\w+)", lambda obj, loc: (obj, loc)),
    ("Clean", r"clean some (\w+) and put it in (\w+)", lambda obj, loc: (obj, loc)),
    
    # Heat & Place
    ("Heat", r"put a hot (\w+) in (\w+)", lambda obj, loc: (obj, loc)),
    ("Heat", r"heat some (\w+) and put it in (\w+)", lambda obj, loc: (obj, loc)),
    
    # Cool & Place
    ("Cool", r"put a cool (\w+) in (\w+)", lambda obj, loc: (obj, loc)),
    ("Cool", r"cool some (\w+) and put it in (\w+)", lambda obj, loc: (obj, loc)),
    
    # Examine in Light
    ("Examine", r"look at (\w+) under the (\w+)", lambda obj, loc: (obj, loc)),
    ("Examine", r"examine the (\w+) with the (\w+)", lambda obj, loc: (obj, loc)),
    
    # Pick & Place (most general, match last)
    ("Pick", r"put a (\w+) in (\w+)", lambda obj, loc: (obj, loc)),
    ("Pick", r"put some (\w+) on (\w+)", lambda obj, loc: (obj, loc)),
]

def plot_memory_recall_curves(
    selected_df: pd.DataFrame,
    recall_type: str = "by_task",  # "by_task" or "by_success"
    metric: str = "obj_recall_rate",  # "obj_recall_rate" or "loc_recall_rate"
    xlabel: str = "Number of Previous Steps",
    ylabel: str = "Recall Rate",
    title: str = "Memory Recall Curve",
    selection_criteria: dict = None,
    figsize=(12, 7),
    dpi=150
):
    """
    Plot memory recall curves
    
    Args:
        selected_df: DataFrame containing recall_curves data
        recall_type: Group by task type ("by_task") or success status ("by_success")
        metric: Metric to plot ("obj_recall_rate" or "loc_recall_rate")
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Chart title
        selection_criteria: Selection criteria dictionary
        figsize: Chart size
        dpi: Chart resolution
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    if selected_df.empty:
        print("No experimental data found matching the criteria for plotting.")
        return
    
    # Assign different colors to each experiment
    colors = sns.color_palette("tab10", n_colors=len(selected_df))
    
    # Get keys to exclude
    exclude_keys = set(selection_criteria.keys()) if selection_criteria else set()
    
    # Store all curve data to determine x-axis range
    all_x_values = set()
    
    for idx, (_, row) in enumerate(selected_df.iterrows()):
        # Get recall_curves data
        recall_curves = row.get('recall_curves', {})
        if not recall_curves:
            print(f"Warning: Experiment '{row.get('experiment', 'unknown')}' has no recall_curves data")
            continue
        
        # Select data based on recall_type
        curves_data = recall_curves.get(recall_type, {})
        if not curves_data:
            continue
        
        # Create legend label
        label = get_config_label(row, exclude_keys)
        
        # Plot curves for each subcategory
        if recall_type == "by_task":
            # Plot by task type, only plot Overall curve for each experiment
            overall_data = curves_data.get("Overall", {})
            if overall_data:
                x_data = [k for k in overall_data.keys()]

                y_data = [overall_data[k].get(metric, 0) for k in x_data]
                all_x_values.update(x_data)
                
                ax.plot(
                    x_data,
                    y_data,
                    linestyle="-",
                    marker="o",
                    markersize=3,
                    label=label,
                    color=colors[idx],
                    alpha=0.8
                )
        
        elif recall_type == "by_success":
            # Plot by success status, use different line styles to distinguish success/failure
            for success_status, line_style in [("true", "-"), ("false", "--")]:
                status_data = curves_data.get(success_status, {})
                if status_data:
                    x_data = [k for k in status_data.keys()]
                    y_data = [status_data[k].get(metric, 0) for k in x_data]
                    all_x_values.update(x_data)
                    
                    status_label = f"{label} ({'Success' if success_status == 'true' else 'Fail'})"
                    ax.plot(
                        x_data,
                        y_data,
                        linestyle=line_style,
                        marker="o" if success_status == "true" else "x",
                        markersize=3,
                        label=status_label,
                        color=colors[idx],
                        alpha=0.8
                    )
    
    # Beautify chart
    criteria_str = format_criteria_for_title(selection_criteria)
    if criteria_str:
        full_title = f"{title}\n[{criteria_str}]"
    else:
        full_title = title
    
    ax.set_title(full_title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.autoscale(enable=True, axis='y', tight=False)
    ax.set_ylim(bottom=0)  # Only fix lower bound to 0, upper bound automatic
    
    # Set x-axis range, interval of 5
    if all_x_values:
        ax.set_xlim(0, 50)  # Set x-axis range
        ax.set_xticks(range(0, 51, 5))  # Set ticks: 0, 5, 10, 15, ..., 50
    
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.3)
    
    # Add legend
    if len(selected_df) > 0:
        ax.legend(
            title="Experiment Configuration",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=9
        )
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_memory_recall_by_task(
    selected_df: pd.DataFrame,
    metric: str = "obj_recall_rate",
    xlabel: str = "Number of Previous Steps", 
    ylabel: str = "Recall Rate",
    title: str = "Memory Recall by Task Type",
    selection_criteria: dict = None,
    figsize=(10, 6),
    dpi=150
):
    """
    Plot separate charts for each task type
    
    Args:
        selected_df: DataFrame containing recall_curves data
        metric: Metric to plot ("obj_recall_rate" or "loc_recall_rate")
        Other parameters same as plot_memory_recall_curves
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from utils.analysis_files.analysis import format_criteria_for_title, get_config_label
    
    if selected_df.empty:
        print("No experimental data found matching the criteria for plotting.")
        return
    
    # Collect all task types
    all_task_types = set()
    for _, row in selected_df.iterrows():
        recall_curves = row.get('recall_curves', {})
        if recall_curves and 'by_task' in recall_curves:
            all_task_types.update(recall_curves['by_task'].keys())
    
    # Remove Overall, handle separately
    all_task_types.discard("Overall")
    task_types = sorted(list(all_task_types))
    
    # Assign colors to each experiment
    colors = sns.color_palette("tab10", n_colors=len(selected_df))
    exclude_keys = set(selection_criteria.keys()) if selection_criteria else set()
    
    # Plot Overall and each task type
    plot_tasks = ["Overall"] + task_types
    
    figures = []  # Store all charts
    
    for task_type in plot_tasks:
        # Create independent chart for each task type
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        has_data = False  # Mark if there is data
        
        for exp_idx, (_, row) in enumerate(selected_df.iterrows()):
            recall_curves = row.get('recall_curves', {})
            if not recall_curves or 'by_task' not in recall_curves:
                continue
            
            task_data = recall_curves['by_task'].get(task_type, {})
            if not task_data:
                continue
            
            x_data = [k for k in task_data.keys()]
            y_data = [task_data[k].get(metric, 0) for k in x_data]
            
            if x_data and y_data:  # Ensure there is data
                has_data = True
                label = get_config_label(row, exclude_keys)
                ax.plot(
                    x_data,
                    y_data,
                    linestyle="-",
                    marker="o",
                    markersize=3,
                    label=label,
                    color=colors[exp_idx],
                    alpha=0.8
                )
        
        if has_data:
            # Set chart title and labels
            criteria_str = format_criteria_for_title(selection_criteria)
            if criteria_str:
                full_title = f"{title} - {task_type}\n[{criteria_str}]"
            else:
                full_title = f"{title} - {task_type}"
            
            ax.set_title(full_title, fontsize=14)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.autoscale(enable=True, axis='y', tight=False)
            ax.set_ylim(bottom=0)  # Only fix lower bound to 0, upper bound automatic
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
            ax.set_xlim(0, 50)
            ax.set_xticks(range(0, 51, 10))
            
            # Add legend
            ax.legend(
                title="Experiment Configuration",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=9
            )
            
            plt.tight_layout()
            plt.show()
            
            figures.append((task_type, fig))
        else:
            plt.close(fig)  # If no data, close chart
    
    return figures

def extract_objects_and_locations(query):
    """Extract items and locations from query, and task type related information"""
    object_location_pattern = r"([a-zA-Z]+)\s(\d+)"
    query_obj_loc = re.findall(object_location_pattern, query)

    task_objects = []
    task_locations = []
    task_type = None
    
    for task_name, pattern, handler in TASK_PATTERNS:
        match = re.search(pattern, query)
        if match:
            task_type = task_name
            obj, loc = match.groups()
            if task_name == "Examine":
                task_objects.append(obj)
                task_objects.append(loc)
            else:
                task_objects.append(obj)
                task_locations.append(loc)
            implicit_locations = IMPLICIT_LOCATIONS.get(task_name.lower(),[])
            task_locations.extend(implicit_locations)
            break

    query_locations = [(obj_q,num_q) for obj_q, num_q in query_obj_loc if obj_q.lower() in LOC]
    query_objects = [(obj_q,num_q) for obj_q, num_q in query_obj_loc if obj_q.lower() not in LOC]
    
    query_objects = list(set(query_objects))
    query_locations = list(set(query_locations))

    return task_type, query_locations, query_objects, task_objects, task_locations

def analyze_memory_usage(df, sample_idx, traj_idx):
    """Analyze historical memory usage of specified trajectory (full history, no window limit)"""
    traj_df = df[(df['sample_idx'] == sample_idx) & (df['traj_idx'] == traj_idx)].copy()
    
    # Task information (query from step 0)
    task_type, query_locations, _, task_objects, task_locations = extract_objects_and_locations(traj_df.iloc[0]["query"])

    results = []
    # History storage, with index matching
    history_objs_with_id = []   # [(name, id), ...]
    history_locs_with_id = []
    step_obs_cache = {}
    for step_idx in sorted(traj_df['step_idx'].unique()):
        if step_idx == 0:
            # Step 0 completely skipped
            continue

        step_data = traj_df[traj_df['step_idx'] == step_idx]
        # Skip invalid action
        if not step_data['action_is_valid'].iloc[0]:
            continue

        obs = step_data["observation"].values[0]
        _, obs_locations, obs_objects, _, _ = extract_objects_and_locations(obs)
        _, action_locations, action_objects, _, _ = extract_objects_and_locations(step_data["action"].values[0])
        obs_obj_set = set(obs_objects)
        obs_loc_set = set(obs_locations)
        step_obs_cache[step_idx] = {
            "objs": obs_obj_set,
            "locs": obs_loc_set
        }
        # === 1. Overall historical recall (excluding current obs) ===
        hist_objs_exact = set(history_objs_with_id)
        hist_locs_exact = set(history_locs_with_id)

        # Recall match (boolean: whether there is recall)
        history_obj_hit = any(obj in hist_objs_exact for obj in action_objects)
        history_loc_hit = any(loc in hist_locs_exact for loc in action_locations)

        # Task hit (boolean: whether recalled items hit task requirements)
        recalled_task_obj_hit = any(
            obj in hist_objs_exact and obj[0].lower() in [t.lower() for t in task_objects]
            for obj in action_objects
        )
        recalled_task_loc_hit = any(
            loc in hist_locs_exact and loc[0].lower() in [t.lower() for t in task_locations]
            for loc in action_locations
        )

        # === 2. Detailed step-by-step recall analysis (including current obs) ===
        # Get all valid steps (including current step)
        valid_steps = traj_df[(traj_df['step_idx'] > 0) & 
                             (traj_df['step_idx'] <= step_idx) & 
                             (traj_df['action_is_valid'] == True)]['step_idx'].values
        
        # Single step recall record
        single_step_recall = {}
        for valid_step in valid_steps:
            cached = step_obs_cache.get(valid_step)
            if cached is None:
                continue
            step_objs_set = cached["objs"]
            step_locs_set = cached["locs"]
            
            single_step_recall[valid_step] = {
                "obj_hit": any(obj in step_objs_set for obj in action_objects),
                "loc_hit": any(loc in step_locs_set for loc in action_locations)
            }

        # === 3. Cumulative recall analysis ===
        cumulative_recall = {}
        for n in range(50):  # n=0 means only current step, n=1 means current + previous 1 step, ...
            if n <= len(valid_steps):
                if n == 0:
                    # Only current step
                    cum_objs = set(obs_objects)
                    cum_locs = set(obs_locations)
                else:
                    # Accumulate previous n steps (counting backwards from current)
                    cum_objs = set()
                    cum_locs = set()
                    for i in range(min(n+1, len(valid_steps))):
                        target_step = valid_steps[-(i+1)]  # The (i+1)th from the end
                        target_obs = traj_df.loc[traj_df['step_idx'] == target_step, "observation"].values[0]
                        _, target_locs, target_objs, _, _ = extract_objects_and_locations(target_obs)
                        cum_objs.update(target_objs)
                        cum_locs.update(target_locs)
                
                cumulative_recall[n] = {
                    "obj_hit": any(obj in cum_objs for obj in action_objects),
                    "loc_hit": any(loc in cum_locs for loc in action_locations)
                }
            else:

                    cumulative_recall[n] = cumulative_recall[max(len(valid_steps)-1,0)]

        # === 4. Current obs vs historical obs recall ===
        current_obj_hit = any(obj in obs_obj_set for obj in action_objects)
        current_loc_hit = any(loc in obs_loc_set for loc in action_locations)

        results.append({
            "sample_idx": sample_idx,
            "traj_idx": traj_idx,
            "step_idx": step_idx,
            "success": step_data['success'].iloc[0],
            "task_type": task_type,
            "task_objects": task_objects,
            "task_locations": task_locations,
            # Basic information
            "obs_objects": obs_objects,
            "obs_locations": obs_locations,
            "action_objects": action_objects,
            "action_locations": action_locations,
            # Whether hit task
            "task_obj_hit": recalled_task_obj_hit,
            "task_loc_hit": recalled_task_loc_hit,
            # Detailed recall analysis
            "single_step_recall": single_step_recall,
            "cumulative_recall": cumulative_recall,
            # Current vs history (boolean values)
            "current_obj_hit": current_obj_hit,
            "current_loc_hit": current_loc_hit,
            "history_obj_hit": history_obj_hit,
            "history_loc_hit": history_loc_hit,
        })

        # Update history (after analysis)
        history_objs_with_id.extend(obs_objects)
        history_locs_with_id.extend(obs_locations)

    return pd.DataFrame(results)

def analyze_all_trajectories(df, n_workers=16):
    """Batch analyze memory usage of all trajectories"""
    unique_trajs = df[['sample_idx', 'traj_idx']].drop_duplicates()
    
    total = len(unique_trajs)
    
    # Use multiprocessing to accelerate analysis
    from functools import partial
    analyze_func = partial(analyze_memory_usage, df)
    
    with mp.Pool(n_workers) as pool:
        results = pool.starmap(analyze_func, unique_trajs.values)
    
    # Merge all results
    all_results = pd.concat(results, ignore_index=True)
    
    print(f"Analysis completed! Analyzed {total} trajectories in total")
    return all_results

def summarize_memory_analysis(mem_df):
    """Summarize memory usage analysis (all recalls are boolean values)"""
    traj_success = mem_df.groupby(['sample_idx', 'traj_idx'])['success'].first().reset_index()
    # 1. Overall recall rate
    total_recall_stats = {
        "obj_recall_rate": mem_df["history_obj_hit"].mean(),
        "loc_recall_rate": mem_df["history_loc_hit"].mean(),
        "task_obj_hit_rate": mem_df["task_obj_hit"].mean(),
        "task_loc_hit_rate": mem_df["task_loc_hit"].mean(),
        "success_rate": traj_success["success"].mean(),
        "step_count": len(mem_df),
    }

    # 2. Statistics by task type
    traj_task = mem_df.groupby(['sample_idx', 'traj_idx'])['task_type'].first().reset_index()
    traj_with_task = traj_success.merge(traj_task, on=['sample_idx', 'traj_idx'])
    task_stats = mem_df.groupby('task_type').apply(
        lambda x: pd.Series({
            'obj_recall_rate': x['history_obj_hit'].mean(),
            'loc_recall_rate': x['history_loc_hit'].mean(),
            'task_obj_hit_rate': x['task_obj_hit'].mean(),
            'task_loc_hit_rate': x['task_loc_hit'].mean(),
            'success_rate': traj_with_task[traj_with_task['task_type'] == x.name]['success'].mean(),
            'step_count': len(x)
        })
    )
    summary_table = pd.concat(
        [task_stats, pd.DataFrame(total_recall_stats, index=["Overall"])]
    )

    # 3. Success/failure comparison
    success_comparison = mem_df.groupby('success').apply(
        lambda x: pd.Series({
            'obj_recall_rate': x['history_obj_hit'].mean(),
            'loc_recall_rate': x['history_loc_hit'].mean(),
            'task_obj_hit_rate': x['task_obj_hit'].mean(),
            'task_loc_hit_rate': x['task_loc_hit'].mean(),
            'current_obj_rate': x['current_obj_hit'].mean(),
            'history_obj_rate': x['history_obj_hit'].mean(),
            'count': len(x)
        })
    )

    return {
        "by_task": summary_table.to_dict(orient='index'),
        'by_success': success_comparison.to_dict(orient='index')
    }

def calculate_cumulative_recall_curves(mem_df):
    """Calculate cumulative recall curve data"""
    
    # Overall curve
    overall_agg = defaultdict(lambda: {'count': 0, 'obj_hit': 0, 'loc_hit': 0})
    for _, row in mem_df.iterrows():
        record = row['cumulative_recall']
        if not isinstance(record, dict):
            continue
        for n, metrics in record.items():
            overall_agg[n]['count'] += 1
            if metrics.get('obj_hit'):
                overall_agg[n]['obj_hit'] += 1
            if metrics.get('loc_hit'):
                overall_agg[n]['loc_hit'] += 1
    
    overall_curve = {}
    for n, agg_data in sorted(overall_agg.items()):
        if agg_data['count'] > 0:
            overall_curve[int(n)] = {
                'obj_recall_rate': float(agg_data['obj_hit'] / agg_data['count']),
                'loc_recall_rate': float(agg_data['loc_hit'] / agg_data['count'])
            }
    
    # Curves by task type
    task_curves = {}
    task_types = [t for t in mem_df['task_type'].unique() if t is not None]
    
    for task_type in sorted(task_types):
        task_df = mem_df[mem_df['task_type'] == task_type]
        task_agg = defaultdict(lambda: {'count': 0, 'obj_hit': 0, 'loc_hit': 0})
        
        for _, row in task_df.iterrows():
            record = row['cumulative_recall']
            if not isinstance(record, dict):
                continue
            for n, metrics in record.items():
                task_agg[n]['count'] += 1
                if metrics.get('obj_hit'):
                    task_agg[n]['obj_hit'] += 1
                if metrics.get('loc_hit'):
                    task_agg[n]['loc_hit'] += 1
        
        if task_agg:
            task_curve = {}
            for n, agg_data in sorted(task_agg.items()):
                if agg_data['count'] > 0:
                    task_curve[int(n)] = {
                        'obj_recall_rate': float(agg_data['obj_hit'] / agg_data['count']),
                        'loc_recall_rate': float(agg_data['loc_hit'] / agg_data['count'])
                    }
            task_curves[task_type] = task_curve
    by_task_curves = {
        **task_curves,
        "Overall": overall_curve
    }
    success_curves = {}
    
    for success_status in [False, True]:
        status_df = mem_df[mem_df['success'] == success_status]
        status_agg = defaultdict(lambda: {'count': 0, 'obj_hit': 0, 'loc_hit': 0})
        
        for _, row in status_df.iterrows():
            record = row['cumulative_recall']
            if not isinstance(record, dict):
                continue
            for n, metrics in record.items():
                status_agg[n]['count'] += 1
                if metrics.get('obj_hit'):
                    status_agg[n]['obj_hit'] += 1
                if metrics.get('loc_hit'):
                    status_agg[n]['loc_hit'] += 1
        
        if status_agg:
            status_curve = {}
            for n, agg_data in sorted(status_agg.items()):
                if agg_data['count'] > 0:
                    status_curve[int(n)] = {
                        'obj_recall_rate': float(agg_data['obj_hit'] / agg_data['count']),
                        'loc_recall_rate': float(agg_data['loc_hit'] / agg_data['count'])
                    }
            success_curves[str(success_status).lower()] = status_curve
    return {
        "by_task": by_task_curves,
        "by_success": success_curves
    }

def process_line(line):
    if not line.strip():
        return None
    return orjson.loads(line)

def load_jsonl_parallel(data_path, n_workers=32):
    with open(data_path, "rb") as f:
        lines = f.readlines()
    with ThreadPool(n_workers) as pool:
        data = [x for x in pool.map(process_line, lines) if x is not None]
    return data

def process_experiment_folder(exp_dir, rewrite):
    """Process memory recall analysis for a single experiment folder"""
    try:
        print(f"Starting memory recall analysis: {exp_dir}")
        result_path = os.path.join(exp_dir, "result_mem_recall.json")

        if not rewrite and os.path.exists(result_path):
            print(f"Skipping (result_mem_recall.json already exists): {exp_dir}")
            return

        # Find jsonl files
        jsonl_files = [f for f in os.listdir(exp_dir) if f.endswith('.jsonl')]
        if not jsonl_files:
            print(f"Warning: No .jsonl file found in {exp_dir}")
            return
        jsonl_path = os.path.join(exp_dir, jsonl_files[0])

        # Load data
        raw_data = load_jsonl_parallel(jsonl_path)
        df = load_all_data(raw_data)
        
        # Analyze memory recall
        mem_df = analyze_all_trajectories(df, n_workers=64)
        
        if mem_df.empty:
            print(f"Warning: Memory analysis result for {exp_dir} is empty")
            return
        else:
            print(f"Memory analysis completed, result rows: {len(mem_df)}")
        # Get summary and curve data
        summary = summarize_memory_analysis(mem_df)
        recall_curves = calculate_cumulative_recall_curves(mem_df)
        
        config_data = get_config(exp_dir)
        
        # Merge results
        final_result = {
            **config_data,
            "memory_recall_summary": summary,
            "recall_curves": recall_curves
        }
        
        # Save results
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"Successfully saved result_mem_recall.json to: {exp_dir}")
        
    except Exception as e:
        print(f"Error processing {exp_dir}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ALFWorld memory recall")
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

    for exp_dir in exp_dirs:
        process_experiment_folder(exp_dir, rewrite=args.rewrite)

    print("All memory recall analysis tasks completed.")