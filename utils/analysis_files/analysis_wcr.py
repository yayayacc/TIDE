import pandas as pd
def cal_weighted_corrected_rate(df_no_state: pd.DataFrame, df_env_state: pd.DataFrame) -> float:
    """
    Calculate the Weighted Corrected Rate (WCR) between two datasets: one without environmental state and one with environmental state.
    Args:
        df_no_state (pd.DataFrame): DataFrame containing results without environmental state.
        df_env_state (pd.DataFrame): DataFrame containing results with environmental state.
    Returns:
        float: The calculated WCR value.
    """
    def get_unique_id(row):
        if row['seed'] is not None and row['seed'] != 'none':
            return row['seed']
        else:
            return row['query']

    # Add unique identifier column for both datasets
    df_no_state['unique_id'] = df_no_state.apply(get_unique_id, axis=1)
    df_env_state['unique_id'] = df_env_state.apply(get_unique_id, axis=1)

    # Get trajectory count (assuming each sample has the same number of trajectories)
    traj_rollout_n = df_no_state.groupby('unique_id')['traj_idx'].nunique().max()
    min_difficulty = 1 / traj_rollout_n
    
    # Calculate difficulty for each sample (difficulty = max(1 - avg_accuracy, 1/traj_rollout_n))
    df_no_state['difficulty'] = (1 - df_no_state['avg_accuracy']).clip(lower=min_difficulty)

    # First complete the construction of merged_df, add accuracy under env_state condition
    merged_df = df_no_state.drop_duplicates(subset=['unique_id'])
    merged_df = merged_df[merged_df['difficulty'] > 0][['unique_id', 'avg_accuracy', 'difficulty']]
    merged_df = merged_df.rename(columns={'avg_accuracy': 'avg_acc_no_state'})

    # Get average accuracy under environment state condition from df_env_state
    env_state_acc = df_env_state.drop_duplicates(subset=['unique_id'])[['unique_id', 'avg_accuracy']]
    env_state_acc = env_state_acc.rename(columns={'avg_accuracy': 'avg_acc_env_state'})

    # Merge the two datasets
    merged_df = merged_df.merge(env_state_acc, on='unique_id', how='inner')

    # Calculate improvement for each sample
    merged_df['improvement'] = merged_df['avg_acc_env_state'] - merged_df['avg_acc_no_state']

    # Calculate potential improvement space for each sample
    merged_df['potential_improvement'] = 1 - merged_df['avg_acc_no_state']

    # Calculate weighted improvement
    merged_df['weighted_improvement'] = merged_df['difficulty'] * merged_df['improvement']
    merged_df['weighted_potential'] = merged_df['difficulty'] * merged_df['potential_improvement']

    # Calculate WCR
    wcr = merged_df['weighted_improvement'].sum() / merged_df['weighted_potential'].sum()

    return wcr
