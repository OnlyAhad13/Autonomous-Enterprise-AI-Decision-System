"""
User Feature Definitions

Aggregated user-level features computed from transaction data.
"""

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

from entities import user


# Data source for user features (Parquet file)
user_features_source = FileSource(
    name="user_features_source",
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)


# User feature view with 30-day aggregations
user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=1),  # Features expire after 1 day in online store
    schema=[
        # Spending patterns
        Field(name="user_total_spend_30d", dtype=Float64, 
              description="Total spending amount in the last 30 days"),
        Field(name="user_avg_order_value", dtype=Float64,
              description="Average order value (total_amount per transaction)"),
        
        # Quantity patterns
        Field(name="user_avg_quantity", dtype=Float64,
              description="Average quantity per transaction"),
        Field(name="user_total_quantity_30d", dtype=Int64,
              description="Total items purchased in last 30 days"),
        
        # Activity metrics
        Field(name="user_transaction_count_30d", dtype=Int64,
              description="Number of transactions in last 30 days"),
        Field(name="user_unique_products_30d", dtype=Int64,
              description="Number of unique products purchased"),
        
        # Channel preference
        Field(name="user_favorite_channel", dtype=String,
              description="Most frequently used sales channel"),
        
        # Recency
        Field(name="user_days_since_last_purchase", dtype=Int64,
              description="Days since last transaction"),
    ],
    source=user_features_source,
    online=True,  # Materialize to online store
    tags={
        "owner": "data-engineering",
        "domain": "user-behavior",
    },
)
