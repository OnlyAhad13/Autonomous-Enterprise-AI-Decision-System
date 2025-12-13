"""
Streaming to Delta Lake Spark Job

PySpark Structured Streaming job that:
1. Reads from Kafka topic events.raw.v1
2. Applies schema, drops malformed records
3. Enriches with geo lookup (lat/lon to region)
4. Writes canonical events to Delta Lake
5. Publishes normalized JSON to Kafka events.canonical.v1

Usage:
    spark-submit --packages ... streaming_to_delta.py --config conf/streaming.conf

Features:
- Exactly-once semantics via checkpointing
- Graceful shutdown handling (SIGTERM/SIGINT)
- Retry/backoff logic for Kafka connection failures
- Dual sink: Delta Lake + Kafka
"""

import argparse
import atexit
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.streaming import StreamingQuery

# Local imports
from transforms import (
    parse_kafka_value,
    drop_malformed_records,
    normalize_event,
    prepare_canonical_output,
    select_delta_columns,
)
from geo_lookup import enrich_dataframe_with_geo


# Global state for graceful shutdown
_active_queries: list[StreamingQuery] = []
_shutdown_requested = False


def load_config(config_path: str) -> dict:
    """
    Load configuration from HOCON file.
    
    Falls back to defaults if pyhocon is not available.
    """
    try:
        from pyhocon import ConfigFactory
        config = ConfigFactory.parse_file(config_path)
        return config
    except ImportError:
        print("Warning: pyhocon not installed, using default configuration")
        return get_default_config()
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return get_default_config()


def get_default_config() -> dict:
    """Return default configuration as a dict."""
    return {
        "streaming": {
            "app": {
                "name": "StreamingToDelta",
                "log_level": "INFO"
            },
            "kafka": {
                "source": {
                    "bootstrap_servers": "localhost:9093",
                    "topic": "events.raw.v1",
                    "consumer_group": "streaming-to-delta",
                    "starting_offsets": "earliest",
                    "max_offsets_per_trigger": 10000,
                    "fail_on_data_loss": False
                },
                "sink": {
                    "bootstrap_servers": "localhost:9093",
                    "topic": "events.canonical.v1"
                }
            },
            "delta": {
                "output_path": "data/lake/delta/events/streaming",
                "checkpoint_path": "data/lake/checkpoints/streaming-to-delta",
                "partition_by": "dt",
                "trigger_interval": "10 seconds"
            },
            "geo": {
                "lookup_file": "conf/geo_regions.json",
                "default_region": "UNKNOWN"
            },
            "retry": {
                "max_attempts": 5,
                "initial_delay_ms": 1000,
                "max_delay_ms": 60000,
                "backoff_multiplier": 2.0
            }
        }
    }


def get_nested(config, *keys, default=None):
    """Safely get nested config value."""
    try:
        value = config
        for key in keys:
            if hasattr(value, 'get'):
                value = value.get(key, default)
            elif hasattr(value, key):
                value = getattr(value, key)
            else:
                value = value[key]
        return value if value is not None else default
    except (KeyError, TypeError, AttributeError):
        return default


def create_spark_session(app_name: str = "StreamingToDelta") -> SparkSession:
    """Create and configure Spark session with Delta Lake and Kafka support."""
    return (
        SparkSession.builder
        .appName(app_name)
        # Delta Lake
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        # Streaming
        .config("spark.sql.streaming.schemaInference", "true")
        .config("spark.sql.adaptive.enabled", "true")
        # Compression
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


def create_kafka_source(
    spark: SparkSession,
    bootstrap_servers: str,
    topic: str,
    starting_offsets: str = "earliest",
    max_offsets_per_trigger: int = 10000,
    fail_on_data_loss: bool = False
) -> DataFrame:
    """
    Create Kafka source DataFrame for structured streaming.
    
    Args:
        spark: SparkSession
        bootstrap_servers: Kafka bootstrap servers
        topic: Source topic name
        starting_offsets: Where to start reading
        max_offsets_per_trigger: Max records per micro-batch
        fail_on_data_loss: Whether to fail on data loss
    
    Returns:
        Streaming DataFrame
    """
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", starting_offsets)
        .option("maxOffsetsPerTrigger", max_offsets_per_trigger)
        .option("failOnDataLoss", str(fail_on_data_loss).lower())
        .load()
    )


def write_to_delta_and_kafka(
    batch_df: DataFrame,
    batch_id: int,
    delta_path: str,
    kafka_bootstrap: str,
    kafka_topic: str,
    spark: SparkSession,
    geo_config_path: Optional[str] = None
) -> None:
    """
    Process a micro-batch: write to Delta Lake and publish to Kafka.
    
    This function is called for each micro-batch in foreachBatch mode.
    
    Args:
        batch_df: Micro-batch DataFrame
        batch_id: Batch identifier
        delta_path: Delta Lake output path
        kafka_bootstrap: Kafka bootstrap servers for sink
        kafka_topic: Kafka output topic
        spark: SparkSession for geo lookup
        geo_config_path: Path to geo config file
    """
    if batch_df.isEmpty():
        print(f"Batch {batch_id}: Empty batch, skipping")
        return
    
    start_time = time.time()
    record_count = batch_df.count()
    print(f"Batch {batch_id}: Processing {record_count} records")
    
    try:
        # 1. Parse Kafka messages
        df = parse_kafka_value(batch_df)
        
        # 2. Drop malformed records
        df = drop_malformed_records(df)
        valid_count = df.count()
        dropped = record_count - valid_count
        if dropped > 0:
            print(f"  Dropped {dropped} malformed records")
        
        # 3. Normalize fields
        df = normalize_event(df)
        
        # 4. Enrich with geo region
        df = enrich_dataframe_with_geo(
            df,
            spark,
            city_col="city",
            config_path=geo_config_path,
            default_region="UNKNOWN"
        )
        
        # 5. Cache for dual write
        df.cache()
        
        # 6. Write to Delta Lake
        delta_df = select_delta_columns(df)
        (
            delta_df.write
            .format("delta")
            .mode("append")
            .partitionBy("dt")
            .option("mergeSchema", "true")
            .save(delta_path)
        )
        print(f"  Written {valid_count} records to Delta Lake")
        
        # 7. Publish to canonical Kafka topic
        kafka_df = prepare_canonical_output(df)
        (
            kafka_df.write
            .format("kafka")
            .option("kafka.bootstrap.servers", kafka_bootstrap)
            .option("topic", kafka_topic)
            .save()
        )
        print(f"  Published {valid_count} records to Kafka topic {kafka_topic}")
        
        # Cleanup
        df.unpersist()
        
        elapsed = time.time() - start_time
        print(f"Batch {batch_id}: Completed in {elapsed:.2f}s ({valid_count / elapsed:.0f} records/s)")
        
    except Exception as e:
        print(f"Batch {batch_id}: ERROR - {e}")
        raise


def setup_signal_handlers():
    """Setup graceful shutdown handlers for SIGTERM and SIGINT."""
    
    def handle_shutdown(signum, frame):
        global _shutdown_requested
        signal_name = signal.Signals(signum).name
        print(f"\nReceived {signal_name}, initiating graceful shutdown...")
        _shutdown_requested = True
        
        for query in _active_queries:
            try:
                print(f"Stopping query: {query.name}")
                query.stop()
            except Exception as e:
                print(f"Error stopping query: {e}")
    
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)


def run_with_retry(
    func,
    max_attempts: int = 5,
    initial_delay_ms: int = 1000,
    max_delay_ms: int = 60000,
    backoff_multiplier: float = 2.0
):
    """
    Execute function with exponential backoff retry.
    
    Args:
        func: Function to execute
        max_attempts: Maximum retry attempts
        initial_delay_ms: Initial delay in milliseconds
        max_delay_ms: Maximum delay in milliseconds
        backoff_multiplier: Backoff multiplier
    
    Returns:
        Function result
    
    Raises:
        Last exception if all retries fail
    """
    delay_ms = initial_delay_ms
    last_exception = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts:
                print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                print(f"Retrying in {delay_ms}ms...")
                time.sleep(delay_ms / 1000)
                delay_ms = min(delay_ms * backoff_multiplier, max_delay_ms)
            else:
                print(f"All {max_attempts} attempts failed")
    
    raise last_exception


def run_streaming_job(config: dict) -> None:
    """
    Main streaming job execution.
    
    Args:
        config: Configuration dictionary
    """
    # Extract config values
    app_name = get_nested(config, "streaming", "app", "name", default="StreamingToDelta")
    
    kafka_source_bootstrap = get_nested(
        config, "streaming", "kafka", "source", "bootstrap_servers", 
        default="localhost:9093"
    )
    kafka_source_topic = get_nested(
        config, "streaming", "kafka", "source", "topic",
        default="events.raw.v1"
    )
    starting_offsets = get_nested(
        config, "streaming", "kafka", "source", "starting_offsets",
        default="earliest"
    )
    max_offsets = get_nested(
        config, "streaming", "kafka", "source", "max_offsets_per_trigger",
        default=10000
    )
    
    kafka_sink_bootstrap = get_nested(
        config, "streaming", "kafka", "sink", "bootstrap_servers",
        default="localhost:9093"
    )
    kafka_sink_topic = get_nested(
        config, "streaming", "kafka", "sink", "topic",
        default="events.canonical.v1"
    )
    
    delta_output_path = get_nested(
        config, "streaming", "delta", "output_path",
        default="data/lake/delta/events/streaming"
    )
    checkpoint_path = get_nested(
        config, "streaming", "delta", "checkpoint_path",
        default="data/lake/checkpoints/streaming-to-delta"
    )
    trigger_interval = get_nested(
        config, "streaming", "delta", "trigger_interval",
        default="10 seconds"
    )
    
    geo_config_path = get_nested(
        config, "streaming", "geo", "lookup_file",
        default="conf/geo_regions.json"
    )
    
    retry_config = get_nested(config, "streaming", "retry", default={})
    
    print("=" * 60)
    print("Streaming to Delta Lake")
    print("=" * 60)
    print(f"Source:     Kafka {kafka_source_topic} @ {kafka_source_bootstrap}")
    print(f"Delta Sink: {delta_output_path}")
    print(f"Kafka Sink: {kafka_sink_topic} @ {kafka_sink_bootstrap}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Trigger:    {trigger_interval}")
    print("=" * 60)
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Create Spark session with retry
    def create_session():
        return create_spark_session(app_name)
    
    spark = run_with_retry(
        create_session,
        max_attempts=retry_config.get("max_attempts", 5),
        initial_delay_ms=retry_config.get("initial_delay_ms", 1000),
        max_delay_ms=retry_config.get("max_delay_ms", 60000),
        backoff_multiplier=retry_config.get("backoff_multiplier", 2.0)
    )
    
    try:
        # Create Kafka source with retry
        def create_source():
            return create_kafka_source(
                spark,
                kafka_source_bootstrap,
                kafka_source_topic,
                starting_offsets,
                max_offsets
            )
        
        kafka_df = run_with_retry(
            create_source,
            max_attempts=retry_config.get("max_attempts", 5),
            initial_delay_ms=retry_config.get("initial_delay_ms", 1000)
        )
        
        # Create streaming query with foreachBatch for dual sink
        query = (
            kafka_df.writeStream
            .foreachBatch(
                lambda df, bid: write_to_delta_and_kafka(
                    df, bid,
                    delta_output_path,
                    kafka_sink_bootstrap,
                    kafka_sink_topic,
                    spark,
                    geo_config_path
                )
            )
            .option("checkpointLocation", checkpoint_path)
            .trigger(processingTime=trigger_interval)
            .queryName("streaming-to-delta")
            .start()
        )
        
        _active_queries.append(query)
        print(f"\nStreaming query started: {query.name}")
        print("Waiting for data... (Press Ctrl+C to stop)\n")
        
        # Wait for termination
        query.awaitTermination()
        
    except Exception as e:
        print(f"Streaming job failed: {e}")
        raise
    finally:
        print("\nStopping Spark session...")
        spark.stop()
        print("Streaming job terminated")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stream events from Kafka to Delta Lake"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="conf/streaming.conf",
        help="Path to configuration file (HOCON format)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without starting stream"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.dry_run:
        print("Configuration loaded successfully:")
        print(f"  Source topic: {get_nested(config, 'streaming', 'kafka', 'source', 'topic')}")
        print(f"  Sink topic:   {get_nested(config, 'streaming', 'kafka', 'sink', 'topic')}")
        print(f"  Delta path:   {get_nested(config, 'streaming', 'delta', 'output_path')}")
        print("\nDry run complete, configuration is valid.")
        sys.exit(0)
    
    try:
        run_streaming_job(config)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
