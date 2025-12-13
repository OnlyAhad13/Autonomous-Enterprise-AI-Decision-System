"""
Product Feature Definitions

Aggregated product-level features computed from transaction data.
"""

from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

from entities import product


# Data source for product features (Parquet file)
product_features_source = FileSource(
    name="product_features_source",
    path="data/product_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)


# Product feature view with 30-day aggregations
product_features = FeatureView(
    name="product_features",
    entities=[product],
    ttl=timedelta(days=1),  # Features expire after 1 day in online store
    schema=[
        # Price statistics
        Field(name="product_median_price_30d", dtype=Float64,
              description="Median transaction price in the last 30 days"),
        Field(name="product_avg_price_30d", dtype=Float64,
              description="Average transaction price in the last 30 days"),
        Field(name="product_min_price_30d", dtype=Float64,
              description="Minimum price in the last 30 days"),
        Field(name="product_max_price_30d", dtype=Float64,
              description="Maximum price in the last 30 days"),
        
        # Sales volume
        Field(name="product_total_sales_30d", dtype=Int64,
              description="Total units sold in the last 30 days"),
        Field(name="product_total_revenue_30d", dtype=Float64,
              description="Total revenue from this product"),
        
        # Customer metrics
        Field(name="product_unique_buyers_30d", dtype=Int64,
              description="Number of unique customers who purchased"),
        Field(name="product_avg_quantity_per_order", dtype=Float64,
              description="Average quantity per transaction"),
        
        # Transaction count
        Field(name="product_transaction_count_30d", dtype=Int64,
              description="Number of transactions for this product"),
        
        # Top region
        Field(name="product_top_region", dtype=String,
              description="Region with most sales"),
    ],
    source=product_features_source,
    online=True,  # Materialize to online store
    tags={
        "owner": "data-engineering",
        "domain": "product-analytics",
    },
)
