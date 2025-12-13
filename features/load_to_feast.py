"""
Feast Feature Ingestion Script

Reads from Delta Lake / Parquet, computes aggregated features,
and writes to Feast offline store (Parquet) and online store (SQLite/Redis).

Usage:
    python features/load_to_feast.py [--source delta|parquet] [--materialize]
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_events_from_parquet(path: str) -> pd.DataFrame:
    """Load events from Parquet file."""
    print(f"Loading events from Parquet: {path}")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} records")
    return df


def load_events_from_delta(path: str) -> pd.DataFrame:
    """Load events from Delta Lake using PySpark."""
    try:
        from pyspark.sql import SparkSession
        
        print(f"Loading events from Delta Lake: {path}")
        spark = SparkSession.builder \
            .appName("FeastIngestion") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        df = spark.read.format("delta").load(path).toPandas()
        spark.stop()
        print(f"  Loaded {len(df):,} records")
        return df
    except ImportError:
        print("Warning: PySpark not available, falling back to Parquet")
        # Try reading as Parquet (Delta stores as Parquet internally)
        parquet_path = Path(path) / "*.parquet"
        return pd.read_parquet(str(parquet_path))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and types for processing."""
    # Rename columns if needed
    column_mapping = {
        "id": "event_id",
        "timestamp": "event_timestamp",
        "total_amount": "total_amount",
    }
    
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    
    # Parse timestamp
    if "event_timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
    elif "timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Calculate total_amount if not present
    if "total_amount" not in df.columns and "price" in df.columns and "quantity" in df.columns:
        df["total_amount"] = df["price"] * df["quantity"]
    
    # Normalize user_id prefix
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].apply(
            lambda x: x if str(x).startswith("usr_") else f"usr_{x}"
        )
    
    # Normalize product_id prefix
    if "product_id" in df.columns:
        df["product_id"] = df["product_id"].apply(
            lambda x: x if str(x).startswith("prod_") else f"prod_{x}"
        )
    
    return df


def compute_user_features(df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """
    Compute user-level aggregated features.
    
    Args:
        df: Events DataFrame
        lookback_days: Number of days to look back
    
    Returns:
        User features DataFrame
    """
    print(f"Computing user features (lookback: {lookback_days} days)...")
    
    # Filter to lookback window
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - timedelta(days=lookback_days)
    df_window = df[df["event_timestamp"] >= cutoff].copy()
    
    if len(df_window) == 0:
        print("  Warning: No data in lookback window, using all data")
        df_window = df.copy()
    
    # Compute aggregations per user
    user_agg = df_window.groupby("user_id").agg(
        user_total_spend_30d=("total_amount", "sum"),
        user_avg_order_value=("total_amount", "mean"),
        user_avg_quantity=("quantity", "mean"),
        user_total_quantity_30d=("quantity", "sum"),
        user_transaction_count_30d=("event_id" if "event_id" in df_window.columns else "user_id", "count"),
        user_unique_products_30d=("product_id", "nunique"),
        last_purchase=("event_timestamp", "max"),
    ).reset_index()
    
    # Calculate days since last purchase
    user_agg["user_days_since_last_purchase"] = (
        (now - user_agg["last_purchase"]).dt.days
    ).fillna(999).astype(int)
    user_agg = user_agg.drop(columns=["last_purchase"])
    
    # Compute favorite channel
    if "channel" in df_window.columns:
        channel_counts = df_window.groupby(["user_id", "channel"]).size().reset_index(name="count")
        idx = channel_counts.groupby("user_id")["count"].idxmax()
        favorite_channel = channel_counts.loc[idx][["user_id", "channel"]]
        favorite_channel = favorite_channel.rename(columns={"channel": "user_favorite_channel"})
        user_agg = user_agg.merge(favorite_channel, on="user_id", how="left")
    else:
        user_agg["user_favorite_channel"] = "unknown"
    
    # Fill NaN
    user_agg["user_favorite_channel"] = user_agg["user_favorite_channel"].fillna("unknown")
    
    # Add timestamps required by Feast
    user_agg["event_timestamp"] = now
    user_agg["created_timestamp"] = now
    
    # Round numeric columns
    for col in ["user_total_spend_30d", "user_avg_order_value", "user_avg_quantity"]:
        if col in user_agg.columns:
            user_agg[col] = user_agg[col].round(2)
    
    print(f"  Computed features for {len(user_agg):,} users")
    return user_agg


def compute_product_features(df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
    """
    Compute product-level aggregated features.
    
    Args:
        df: Events DataFrame
        lookback_days: Number of days to look back
    
    Returns:
        Product features DataFrame
    """
    print(f"Computing product features (lookback: {lookback_days} days)...")
    
    # Filter to lookback window
    now = pd.Timestamp.now(tz="UTC")
    cutoff = now - timedelta(days=lookback_days)
    df_window = df[df["event_timestamp"] >= cutoff].copy()
    
    if len(df_window) == 0:
        print("  Warning: No data in lookback window, using all data")
        df_window = df.copy()
    
    # Compute aggregations per product
    product_agg = df_window.groupby("product_id").agg(
        product_median_price_30d=("price", "median"),
        product_avg_price_30d=("price", "mean"),
        product_min_price_30d=("price", "min"),
        product_max_price_30d=("price", "max"),
        product_total_sales_30d=("quantity", "sum"),
        product_total_revenue_30d=("total_amount", "sum"),
        product_unique_buyers_30d=("user_id", "nunique"),
        product_avg_quantity_per_order=("quantity", "mean"),
        product_transaction_count_30d=("product_id", "count"),
    ).reset_index()
    
    # Compute top region
    if "region" in df_window.columns:
        region_counts = df_window.groupby(["product_id", "region"]).size().reset_index(name="count")
        idx = region_counts.groupby("product_id")["count"].idxmax()
        top_region = region_counts.loc[idx][["product_id", "region"]]
        top_region = top_region.rename(columns={"region": "product_top_region"})
        product_agg = product_agg.merge(top_region, on="product_id", how="left")
    elif "location" in df_window.columns:
        # Extract country from location
        df_window["region"] = df_window["location"].apply(
            lambda x: x.split(",")[-1].strip() if isinstance(x, str) and "," in x else "Unknown"
        )
        region_counts = df_window.groupby(["product_id", "region"]).size().reset_index(name="count")
        idx = region_counts.groupby("product_id")["count"].idxmax()
        top_region = region_counts.loc[idx][["product_id", "region"]]
        top_region = top_region.rename(columns={"region": "product_top_region"})
        product_agg = product_agg.merge(top_region, on="product_id", how="left")
    else:
        product_agg["product_top_region"] = "Unknown"
    
    # Fill NaN
    product_agg["product_top_region"] = product_agg["product_top_region"].fillna("Unknown")
    
    # Add timestamps required by Feast
    product_agg["event_timestamp"] = now
    product_agg["created_timestamp"] = now
    
    # Round numeric columns
    numeric_cols = [
        "product_median_price_30d", "product_avg_price_30d",
        "product_min_price_30d", "product_max_price_30d",
        "product_total_revenue_30d", "product_avg_quantity_per_order"
    ]
    for col in numeric_cols:
        if col in product_agg.columns:
            product_agg[col] = product_agg[col].round(2)
    
    print(f"  Computed features for {len(product_agg):,} products")
    return product_agg


def write_to_offline_store(df: pd.DataFrame, output_path: str) -> None:
    """Write features to Parquet file (offline store)."""
    print(f"Writing to offline store: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Written {len(df):,} records")


def materialize_to_online_store(repo_path: str, feature_views: list = None) -> None:
    """
    Materialize features to online store using Feast CLI.
    
    Args:
        repo_path: Path to feature repository
        feature_views: List of feature view names to materialize (None = all)
    """
    try:
        from feast import FeatureStore
        
        print(f"Materializing to online store...")
        store = FeatureStore(repo_path=repo_path)
        
        # Materialize from 30 days ago to now
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        if feature_views:
            store.materialize(
                start_date=start_date,
                end_date=end_date,
                feature_views=feature_views
            )
        else:
            store.materialize(
                start_date=start_date,
                end_date=end_date
            )
        
        print("  Materialization complete")
    except ImportError:
        print("Warning: Feast not installed. Run: pip install feast")
    except Exception as e:
        print(f"Warning: Materialization failed: {e}")
        print("  Run 'feast apply' first to register feature views")


def main():
    parser = argparse.ArgumentParser(description="Load features to Feast")
    parser.add_argument(
        "--source", "-s",
        choices=["parquet", "delta"],
        default="parquet",
        help="Data source type (default: parquet)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input path (default: data/sample/events.parquet)"
    )
    parser.add_argument(
        "--materialize", "-m",
        action="store_true",
        help="Materialize to online store after writing offline"
    )
    parser.add_argument(
        "--lookback-days", "-l",
        type=int,
        default=30,
        help="Lookback window in days (default: 30)"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    project_root = Path(__file__).parent.parent
    repo_path = project_root / "features" / "feature_repo"
    
    if args.input:
        input_path = args.input
    elif args.source == "delta":
        input_path = str(project_root / "data" / "lake" / "delta" / "events" / "streaming")
    else:
        input_path = str(project_root / "data" / "sample" / "events.parquet")
    
    # Validate input exists
    if not Path(input_path).exists():
        print(f"Error: Input path not found: {input_path}")
        print("Available sample data: data/sample/events.parquet")
        sys.exit(1)
    
    print("=" * 60)
    print("Feast Feature Ingestion")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Input:  {input_path}")
    print(f"Lookback: {args.lookback_days} days")
    print("=" * 60)
    
    # Load events
    if args.source == "delta":
        events_df = load_events_from_delta(input_path)
    else:
        events_df = load_events_from_parquet(input_path)
    
    # Normalize columns
    events_df = normalize_columns(events_df)
    
    # Compute features
    user_features = compute_user_features(events_df, args.lookback_days)
    product_features = compute_product_features(events_df, args.lookback_days)
    
    # Write to offline store
    user_output = repo_path / "data" / "user_features.parquet"
    product_output = repo_path / "data" / "product_features.parquet"
    
    write_to_offline_store(user_features, str(user_output))
    write_to_offline_store(product_features, str(product_output))
    
    # Materialize to online store if requested
    if args.materialize:
        materialize_to_online_store(str(repo_path))
    
    print("=" * 60)
    print("Ingestion complete!")
    print("=" * 60)
    print(f"\nOffline store files:")
    print(f"  {user_output}")
    print(f"  {product_output}")
    print(f"\nNext steps:")
    print(f"  1. cd {repo_path} && feast apply")
    print(f"  2. python features/load_to_feast.py --materialize")
    print(f"  3. python -m pytest tests/test_feast_features.py -v")


if __name__ == "__main__":
    main()
