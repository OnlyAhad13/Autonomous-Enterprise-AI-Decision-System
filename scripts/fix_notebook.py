#!/usr/bin/env python3
"""Fix the 02_baselines.ipynb notebook to handle single-transaction-per-user data."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "02_baselines.ipynb"

# New create_churn_labels function
new_function = '''def create_churn_labels(df: pd.DataFrame, churn_window_days: int = 7) -> pd.DataFrame:
    """
    Create churn labels based on user activity.
    Handles both multi-transaction users and single-transaction synthetic data.
    
    For real data: Churned = no activity in the last N days of the dataset.
    For synthetic single-transaction data: Creates simulated churn based on
    transaction characteristics (recency + revenue percentile).
    """
    max_date = df['timestamp'].max()
    cutoff_date = max_date - timedelta(days=churn_window_days)
    
    # Split data: training period vs churn evaluation period
    train_data = df[df['timestamp'] < cutoff_date].copy()
    churn_period = df[df['timestamp'] >= cutoff_date].copy()
    
    # Users active in churn period = not churned
    active_users = set(churn_period['user_id'].unique())
    train_users = set(train_data['user_id'].unique())
    
    # Check if we have overlap (real multi-transaction data)
    overlap = train_users & active_users
    use_synthetic_labels = len(overlap) < 0.05 * len(train_users) if len(train_users) > 0 else True
    
    if use_synthetic_labels:
        print("[INFO] Detected single-transaction-per-user data. Using synthetic churn labels.")
        # Use all data for feature engineering when each user has ~1 transaction
        user_features = df.groupby('user_id').agg({
            'id': 'count',
            'revenue': ['sum', 'mean', 'std'],
            'price': ['mean', 'max', 'min'],
            'quantity': ['sum', 'mean'],
            'timestamp': ['min', 'max'],
            'channel': lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown'
        }).reset_index()
    else:
        # Aggregate user features from training period only
        user_features = train_data.groupby('user_id').agg({
            'id': 'count',
            'revenue': ['sum', 'mean', 'std'],
            'price': ['mean', 'max', 'min'],
            'quantity': ['sum', 'mean'],
            'timestamp': ['min', 'max'],
            'channel': lambda x: x.mode().iloc[0] if len(x) > 0 else 'unknown'
        }).reset_index()
    
    # Flatten column names
    user_features.columns = [
        'user_id', 'transaction_count', 'total_revenue', 'avg_revenue', 'std_revenue',
        'avg_price', 'max_price', 'min_price', 'total_quantity', 'avg_quantity',
        'first_transaction', 'last_transaction', 'primary_channel'
    ]
    
    # Calculate derived features (use max_date as reference for synthetic)
    ref_date = max_date if use_synthetic_labels else cutoff_date
    user_features['days_since_first'] = (
        (ref_date - user_features['first_transaction']).dt.days
    )
    user_features['days_since_last'] = (
        (ref_date - user_features['last_transaction']).dt.days
    )
    user_features['avg_days_between'] = (
        user_features['days_since_first'] / user_features['transaction_count'].clip(lower=1)
    )
    
    # Fill NaN in std_revenue (users with 1 transaction)
    user_features['std_revenue'] = user_features['std_revenue'].fillna(0)
    
    # Create churn label
    if use_synthetic_labels:
        # Synthetic churn: Users with low revenue AND high recency are 'churned'
        # This creates a ~30% churn rate based on bottom revenue quartile + recent inactivity
        revenue_threshold = user_features['total_revenue'].quantile(0.35)
        recency_threshold = user_features['days_since_last'].quantile(0.65)
        
        # Churn if: low revenue OR (medium revenue AND high recency)
        user_features['churned'] = (
            (user_features['total_revenue'] < revenue_threshold) |
            ((user_features['total_revenue'] < user_features['total_revenue'].quantile(0.5)) &
             (user_features['days_since_last'] > recency_threshold))
        ).astype(int)
        
        print(f"[INFO] Synthetic churn rate: {user_features['churned'].mean():.2%}")
    else:
        user_features['churned'] = ~user_features['user_id'].isin(active_users)
        user_features['churned'] = user_features['churned'].astype(int)
    
    # Drop timestamp columns (not needed for modeling)
    user_features = user_features.drop(['first_transaction', 'last_transaction'], axis=1)
    
    return user_features


# Create user-level dataset with churn labels
CHURN_WINDOW = 7
user_df = create_churn_labels(df, churn_window_days=CHURN_WINDOW)

print(f"\\nUser-level dataset: {user_df.shape}")
print(f"\\nChurn distribution:")
print(user_df['churned'].value_counts(normalize=True).round(3))'''


def main():
    # Read the notebook
    with open(NOTEBOOK_PATH, 'r') as f:
        nb = json.load(f)
    
    # Find the cell with create_churn_labels
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def create_churn_labels' in source:
                # Replace the cell content
                nb['cells'][i]['source'] = [line + '\n' for line in new_function.split('\n')[:-1]]
                nb['cells'][i]['source'].append(new_function.split('\n')[-1])
                print(f"Updated cell {i} with new create_churn_labels function")
                break
    
    # Write the updated notebook
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=4)
    
    print(f"Notebook updated: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
