"""
Streaming Transforms Module

Unit-testable transform functions for the streaming pipeline.
These functions are pure transformations that can be tested without Spark infrastructure.
"""

import json
from datetime import datetime
from typing import Any, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType, MapType
)


# Schema for raw Kafka events (matches kafka_producer.py EVENT_SCHEMA)
RAW_EVENT_SCHEMA = StructType([
    StructField("id", StringType(), nullable=True),
    StructField("timestamp", StringType(), nullable=True),
    StructField("user_id", StringType(), nullable=True),
    StructField("product_id", StringType(), nullable=True),
    StructField("price", DoubleType(), nullable=True),
    StructField("quantity", IntegerType(), nullable=True),
    StructField("location", StringType(), nullable=True),
    StructField("metadata", MapType(StringType(), StringType()), nullable=True),
])

# Required fields for validation
REQUIRED_FIELDS = ["id", "timestamp", "user_id", "product_id", "price", "quantity", "location"]


def parse_kafka_value(df: DataFrame) -> DataFrame:
    """
    Parse Kafka message value from binary JSON to structured columns.
    
    Args:
        df: DataFrame with 'value' column (binary)
    
    Returns:
        DataFrame with parsed event columns
    """
    # Convert binary to string
    df = df.withColumn("value_str", F.col("value").cast("string"))
    
    # Parse JSON to struct
    df = df.withColumn(
        "parsed",
        F.from_json(F.col("value_str"), RAW_EVENT_SCHEMA)
    )
    
    # Expand struct fields
    df = df.select(
        F.col("key").cast("string").alias("kafka_key"),
        F.col("topic").alias("kafka_topic"),
        F.col("partition").alias("kafka_partition"),
        F.col("offset").alias("kafka_offset"),
        F.col("timestamp").alias("kafka_timestamp"),
        F.col("parsed.*")
    )
    
    return df


def validate_event(event: dict) -> tuple[bool, list[str]]:
    """
    Validate a single event dictionary.
    
    This is a pure Python function for unit testing.
    
    Args:
        event: Event dictionary
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in event or event[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate price
    if "price" in event and event["price"] is not None:
        if not isinstance(event["price"], (int, float)) or event["price"] < 0:
            errors.append(f"Invalid price: {event['price']}")
    
    # Validate quantity
    if "quantity" in event and event["quantity"] is not None:
        if not isinstance(event["quantity"], int) or event["quantity"] < 1:
            errors.append(f"Invalid quantity: {event['quantity']}")
    
    # Validate timestamp format
    if "timestamp" in event and event["timestamp"]:
        try:
            datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            errors.append(f"Invalid timestamp format: {event['timestamp']}")
    
    return len(errors) == 0, errors


def drop_malformed_records(df: DataFrame) -> DataFrame:
    """
    Filter out records that don't meet validation criteria.
    
    Args:
        df: DataFrame with parsed event columns
    
    Returns:
        DataFrame with only valid records
    """
    # Mark records with validation issues
    df = df.withColumn(
        "_is_valid",
        (F.col("id").isNotNull()) &
        (F.col("timestamp").isNotNull()) &
        (F.col("user_id").isNotNull()) &
        (F.col("product_id").isNotNull()) &
        (F.col("price").isNotNull()) & (F.col("price") >= 0) &
        (F.col("quantity").isNotNull()) & (F.col("quantity") >= 1) &
        (F.col("location").isNotNull())
    )
    
    # Log invalid record count (for metrics)
    # In production, you'd send this to a monitoring system
    
    # Filter and drop validation column
    df_valid = df.filter(F.col("_is_valid")).drop("_is_valid")
    
    return df_valid


def normalize_event(df: DataFrame) -> DataFrame:
    """
    Normalize event fields for consistency.
    
    Transformations:
    - Parse timestamp to TimestampType
    - Extract date for partitioning
    - Normalize user_id and product_id prefixes
    - Extract city/country from location
    - Calculate total_amount
    - Add processing metadata
    
    Args:
        df: DataFrame with validated event columns
    
    Returns:
        DataFrame with normalized fields
    """
    # Parse timestamp
    df = df.withColumn(
        "event_timestamp",
        F.to_timestamp(F.col("timestamp"))
    )
    
    # Extract partition date
    df = df.withColumn(
        "dt",
        F.date_format(F.col("event_timestamp"), "yyyy-MM-dd")
    )
    
    # Extract hour for potential sub-partitioning
    df = df.withColumn(
        "event_hour",
        F.hour(F.col("event_timestamp"))
    )
    
    # Normalize user_id prefix
    df = df.withColumn(
        "user_id",
        F.when(
            ~F.col("user_id").startswith("usr_"),
            F.concat(F.lit("usr_"), F.col("user_id"))
        ).otherwise(F.col("user_id"))
    )
    
    # Normalize product_id prefix
    df = df.withColumn(
        "product_id",
        F.when(
            ~F.col("product_id").startswith("prod_"),
            F.concat(F.lit("prod_"), F.col("product_id"))
        ).otherwise(F.col("product_id"))
    )
    
    # Extract city and country from location
    df = df.withColumn(
        "city",
        F.trim(F.split(F.col("location"), ",").getItem(0))
    )
    df = df.withColumn(
        "country",
        F.trim(F.split(F.col("location"), ",").getItem(1))
    )
    
    # Calculate total amount
    df = df.withColumn(
        "total_amount",
        F.round(F.col("price") * F.col("quantity"), 2)
    )
    
    # Extract metadata fields
    df = df.withColumn(
        "channel",
        F.col("metadata").getItem("channel")
    )
    df = df.withColumn(
        "device_type",
        F.col("metadata").getItem("device_type")
    )
    df = df.withColumn(
        "session_id",
        F.col("metadata").getItem("session_id")
    )
    
    # Add processing metadata
    df = df.withColumn("processed_at", F.current_timestamp())
    df = df.withColumn("processing_date", F.current_date())
    
    return df


def normalize_event_dict(event: dict) -> dict:
    """
    Normalize a single event dictionary.
    
    This is a pure Python function for unit testing.
    
    Args:
        event: Raw event dictionary
    
    Returns:
        Normalized event dictionary
    """
    normalized = event.copy()
    
    # Normalize user_id
    if normalized.get("user_id") and not normalized["user_id"].startswith("usr_"):
        normalized["user_id"] = f"usr_{normalized['user_id']}"
    
    # Normalize product_id
    if normalized.get("product_id") and not normalized["product_id"].startswith("prod_"):
        normalized["product_id"] = f"prod_{normalized['product_id']}"
    
    # Extract city/country
    if normalized.get("location"):
        parts = normalized["location"].split(",")
        normalized["city"] = parts[0].strip() if len(parts) > 0 else None
        normalized["country"] = parts[1].strip() if len(parts) > 1 else None
    
    # Calculate total
    if normalized.get("price") is not None and normalized.get("quantity") is not None:
        normalized["total_amount"] = round(normalized["price"] * normalized["quantity"], 2)
    
    return normalized


def to_canonical_event(event: dict, region: str = None) -> dict:
    """
    Convert event to canonical format for output Kafka topic.
    
    This is a pure Python function for unit testing.
    
    Args:
        event: Normalized event dictionary
        region: Geo region code
    
    Returns:
        Canonical event dictionary
    """
    canonical = {
        "event_id": event.get("id"),
        "event_type": "transaction",
        "event_time": event.get("timestamp"),
        "user": {
            "id": event.get("user_id"),
            "session_id": event.get("session_id") or event.get("metadata", {}).get("session_id"),
        },
        "product": {
            "id": event.get("product_id"),
        },
        "transaction": {
            "price": event.get("price"),
            "quantity": event.get("quantity"),
            "total": event.get("total_amount"),
        },
        "location": {
            "raw": event.get("location"),
            "city": event.get("city"),
            "country": event.get("country"),
            "region": region,
        },
        "channel": event.get("channel") or event.get("metadata", {}).get("channel"),
        "device_type": event.get("device_type") or event.get("metadata", {}).get("device_type"),
        "partition_date": event.get("dt"),
    }
    
    return canonical


def prepare_canonical_output(df: DataFrame) -> DataFrame:
    """
    Prepare DataFrame for canonical Kafka output.
    
    Creates a JSON string in the canonical format.
    
    Args:
        df: Normalized DataFrame with region column
    
    Returns:
        DataFrame with 'key' and 'value' columns for Kafka
    """
    # Build canonical struct
    canonical_struct = F.struct(
        F.col("id").alias("event_id"),
        F.lit("transaction").alias("event_type"),
        F.col("timestamp").alias("event_time"),
        F.struct(
            F.col("user_id").alias("id"),
            F.col("session_id")
        ).alias("user"),
        F.struct(
            F.col("product_id").alias("id")
        ).alias("product"),
        F.struct(
            F.col("price"),
            F.col("quantity"),
            F.col("total_amount").alias("total")
        ).alias("transaction"),
        F.struct(
            F.col("location").alias("raw"),
            F.col("city"),
            F.col("country"),
            F.col("region")
        ).alias("location"),
        F.col("channel"),
        F.col("device_type"),
        F.col("dt").alias("partition_date")
    )
    
    # Convert to JSON string for Kafka
    df = df.withColumn("value", F.to_json(canonical_struct))
    
    # Use user_id as key for partitioning
    df = df.withColumn("key", F.col("user_id"))
    
    return df.select("key", "value")


def select_delta_columns(df: DataFrame) -> DataFrame:
    """
    Select and order columns for Delta Lake output.
    
    Args:
        df: Normalized DataFrame
    
    Returns:
        DataFrame with Delta output schema
    """
    return df.select(
        # Primary identifiers
        "id",
        "event_timestamp",
        "dt",
        "event_hour",
        
        # Kafka metadata
        "kafka_offset",
        "kafka_partition",
        "kafka_timestamp",
        
        # Entity references
        "user_id",
        "product_id",
        
        # Transaction data
        "price",
        "quantity",
        "total_amount",
        
        # Location
        "location",
        "city",
        "country",
        "region",
        
        # Metadata
        "channel",
        "device_type",
        "session_id",
        
        # Processing info
        "processed_at",
        "processing_date",
    )
