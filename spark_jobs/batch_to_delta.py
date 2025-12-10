"""
Batch to Delta Lake Spark Job

Reads events from JSONL/Parquet files, normalizes fields, and writes
partitioned Delta files.

Usage:
    spark-submit batch_to_delta.py \
        --input-path data/sample \
        --output-path data/lake/delta/events \
        --date 2024-01-15
"""

import argparse
import sys
from datetime import datetime
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, TimestampType, MapType
)


# Define schema for business events
EVENT_SCHEMA = StructType([
    StructField("id", StringType(), nullable=False),
    StructField("timestamp", StringType(), nullable=False),
    StructField("user_id", StringType(), nullable=False),
    StructField("product_id", StringType(), nullable=False),
    StructField("price", DoubleType(), nullable=False),
    StructField("quantity", IntegerType(), nullable=False),
    StructField("location", StringType(), nullable=False),
    StructField("metadata", StringType(), nullable=True),  # JSON string
])


def create_spark_session(app_name: str = "BatchToDelta") -> SparkSession:
    """Create and configure Spark session with Delta Lake support."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


def read_source_data(
    spark: SparkSession,
    input_path: str,
    file_format: str = "auto"
) -> DataFrame:
    """
    Read source data from JSONL or Parquet files.
    
    Args:
        spark: SparkSession instance
        input_path: Path to input directory or file
        file_format: 'jsonl', 'parquet', or 'auto' (detect from files)
    
    Returns:
        DataFrame with raw event data
    """
    import os
    
    # Auto-detect format
    if file_format == "auto":
        if os.path.exists(f"{input_path}/events.parquet"):
            file_format = "parquet"
            source_path = f"{input_path}/events.parquet"
        elif os.path.exists(f"{input_path}/events.jsonl"):
            file_format = "jsonl"
            source_path = f"{input_path}/events.jsonl"
        elif os.path.isdir(input_path):
            # Check for parquet files in directory
            if any(f.endswith('.parquet') for f in os.listdir(input_path)):
                file_format = "parquet"
                source_path = input_path
            else:
                file_format = "jsonl"
                source_path = input_path
        else:
            raise FileNotFoundError(f"No valid source files found in {input_path}")
    else:
        source_path = input_path
    
    print(f"Reading {file_format} data from: {source_path}")
    
    if file_format == "parquet":
        df = spark.read.parquet(source_path)
    elif file_format in ("jsonl", "json"):
        df = spark.read.json(source_path, schema=EVENT_SCHEMA)
    else:
        raise ValueError(f"Unsupported format: {file_format}")
    
    print(f"Read {df.count()} records from source")
    return df


def normalize_fields(df: DataFrame) -> DataFrame:
    """
    Normalize and clean event fields.
    
    Transformations:
    - Parse timestamp to TimestampType
    - Extract date for partitioning
    - Trim string fields
    - Parse metadata JSON if string
    - Add processing metadata
    """
    # Parse timestamp string to timestamp type
    df = df.withColumn(
        "event_timestamp",
        F.to_timestamp(F.col("timestamp"))
    )
    
    # Extract date for partitioning (dt = YYYY-MM-DD)
    df = df.withColumn(
        "dt",
        F.date_format(F.col("event_timestamp"), "yyyy-MM-dd")
    )
    
    # Trim string fields
    string_cols = ["id", "user_id", "product_id", "location"]
    for col_name in string_cols:
        df = df.withColumn(col_name, F.trim(F.col(col_name)))
    
    # Normalize user_id (ensure consistent prefix)
    df = df.withColumn(
        "user_id",
        F.when(
            ~F.col("user_id").startswith("usr_"),
            F.concat(F.lit("usr_"), F.col("user_id"))
        ).otherwise(F.col("user_id"))
    )
    
    # Normalize product_id (ensure consistent prefix)
    df = df.withColumn(
        "product_id",
        F.when(
            ~F.col("product_id").startswith("prod_"),
            F.concat(F.lit("prod_"), F.col("product_id"))
        ).otherwise(F.col("product_id"))
    )
    
    # Extract location components
    df = df.withColumn(
        "city",
        F.trim(F.split(F.col("location"), ",").getItem(0))
    )
    df = df.withColumn(
        "country",
        F.trim(F.split(F.col("location"), ",").getItem(1))
    )
    
    # Parse metadata JSON string to struct
    df = df.withColumn(
        "metadata_parsed",
        F.from_json(
            F.col("metadata"),
            MapType(StringType(), StringType())
        )
    )
    
    # Extract common metadata fields
    df = df.withColumn(
        "channel",
        F.col("metadata_parsed").getItem("channel")
    )
    df = df.withColumn(
        "device_type",
        F.col("metadata_parsed").getItem("device_type")
    )
    df = df.withColumn(
        "session_id",
        F.col("metadata_parsed").getItem("session_id")
    )
    
    # Calculate total amount
    df = df.withColumn(
        "total_amount",
        F.round(F.col("price") * F.col("quantity"), 2)
    )
    
    # Add processing metadata
    df = df.withColumn(
        "processed_at",
        F.current_timestamp()
    )
    df = df.withColumn(
        "processing_date",
        F.current_date()
    )
    
    return df


def validate_data(df: DataFrame) -> DataFrame:
    """
    Filter out invalid records and log validation metrics.
    
    Validation rules:
    - id must not be null
    - timestamp must be valid
    - price must be >= 0
    - quantity must be >= 1
    """
    initial_count = df.count()
    
    # Filter valid records
    df_valid = df.filter(
        (F.col("id").isNotNull()) &
        (F.col("event_timestamp").isNotNull()) &
        (F.col("price") >= 0) &
        (F.col("quantity") >= 1)
    )
    
    valid_count = df_valid.count()
    invalid_count = initial_count - valid_count
    
    print(f"Validation: {valid_count} valid, {invalid_count} invalid out of {initial_count} total")
    
    if invalid_count > 0:
        # Log sample of invalid records for debugging
        print("Sample invalid records:")
        invalid_df = df.filter(
            (F.col("id").isNull()) |
            (F.col("event_timestamp").isNull()) |
            (F.col("price") < 0) |
            (F.col("quantity") < 1)
        ).limit(5)
        invalid_df.show(truncate=False)
    
    return df_valid


def select_output_columns(df: DataFrame) -> DataFrame:
    """Select and order columns for Delta output."""
    return df.select(
        # Primary identifiers
        "id",
        "event_timestamp",
        "dt",
        
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
        
        # Metadata
        "channel",
        "device_type",
        "session_id",
        "metadata",
        
        # Processing info
        "processed_at",
        "processing_date",
    )


def write_delta(
    df: DataFrame,
    output_path: str,
    partition_by: str = "dt",
    mode: str = "append"
) -> None:
    """
    Write DataFrame to Delta Lake with partitioning.
    
    Args:
        df: DataFrame to write
        output_path: Path to Delta table
        partition_by: Partition column(s)
        mode: Write mode ('append', 'overwrite')
    """
    print(f"Writing {df.count()} records to Delta table: {output_path}")
    print(f"Partition by: {partition_by}, Mode: {mode}")
    
    (
        df.write
        .format("delta")
        .mode(mode)
        .partitionBy(partition_by)
        .option("mergeSchema", "true")
        .save(output_path)
    )
    
    print(f"Successfully wrote to {output_path}")


def run_batch_job(
    input_path: str,
    output_path: str,
    date: Optional[str] = None,
    mode: str = "append"
) -> dict:
    """
    Main batch processing pipeline.
    
    Args:
        input_path: Path to source data
        output_path: Path to Delta Lake output
        date: Optional date filter (YYYY-MM-DD)
        mode: Write mode
    
    Returns:
        Processing statistics
    """
    print("=" * 60)
    print("Batch to Delta Lake Processing")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Date:   {date or 'all'}")
    print(f"Mode:   {mode}")
    print("=" * 60)
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Read source data
        df_raw = read_source_data(spark, input_path)
        
        # Normalize fields
        df_normalized = normalize_fields(df_raw)
        
        # Filter by date if specified
        if date:
            df_normalized = df_normalized.filter(F.col("dt") == date)
            print(f"Filtered to date {date}: {df_normalized.count()} records")
        
        # Validate data
        df_valid = validate_data(df_normalized)
        
        # Select output columns
        df_output = select_output_columns(df_valid)
        
        # Write to Delta
        write_delta(df_output, output_path, mode=mode)
        
        # Collect statistics
        stats = {
            "input_count": df_raw.count(),
            "output_count": df_output.count(),
            "partitions": df_output.select("dt").distinct().count(),
        }
        
        print("\n" + "=" * 60)
        print("Processing Complete")
        print("=" * 60)
        print(f"Input records:  {stats['input_count']}")
        print(f"Output records: {stats['output_count']}")
        print(f"Partitions:     {stats['partitions']}")
        
        return stats
        
    finally:
        spark.stop()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Process events and write to Delta Lake"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input data (JSONL or Parquet)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to Delta Lake output"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Filter to specific date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="append",
        choices=["append", "overwrite"],
        help="Write mode (default: append)"
    )
    
    args = parser.parse_args()
    
    try:
        stats = run_batch_job(
            input_path=args.input_path,
            output_path=args.output_path,
            date=args.date,
            mode=args.mode
        )
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
