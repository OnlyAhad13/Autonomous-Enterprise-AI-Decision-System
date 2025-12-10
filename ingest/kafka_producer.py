"""
Kafka Producer for Business Events

Reads events from JSONL file, registers JSON schema in Schema Registry,
and publishes events to Kafka topic.

Usage:
    python kafka_producer.py --input data/sample/events.jsonl
    python kafka_producer.py --input data/sample/events.jsonl --topic events.raw.v1 --batch-size 1000
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from confluent_kafka import Producer
    from confluent_kafka.admin import AdminClient, NewTopic
    from confluent_kafka.schema_registry import SchemaRegistryClient, Schema
    from confluent_kafka.serialization import (
        SerializationContext,
        MessageField,
        StringSerializer,
    )
    from confluent_kafka.schema_registry.json_schema import JSONSerializer
except ImportError:
    print("confluent-kafka is required. Install with: pip install confluent-kafka[json,avro]")
    sys.exit(1)


# JSON Schema for business events (compatible with Schema Registry)
EVENT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "BusinessEvent",
    "description": "Schema for streaming business transaction events",
    "type": "object",
    "properties": {
        "id": {
            "type": "string",
            "description": "Unique event identifier (UUID)"
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp"
        },
        "user_id": {
            "type": "string",
            "description": "User identifier"
        },
        "product_id": {
            "type": "string",
            "description": "Product identifier"
        },
        "price": {
            "type": "number",
            "minimum": 0,
            "description": "Transaction price"
        },
        "quantity": {
            "type": "integer",
            "minimum": 1,
            "description": "Item quantity"
        },
        "location": {
            "type": "string",
            "description": "Geographic location"
        },
        "metadata": {
            "type": "object",
            "description": "Additional event metadata",
            "properties": {
                "channel": {
                    "type": "string",
                    "enum": ["web", "mobile", "api", "pos"]
                },
                "device_type": {"type": "string"},
                "session_id": {"type": "string"},
                "referrer": {"type": "string"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    },
    "required": ["id", "timestamp", "user_id", "product_id", "price", "quantity", "location"]
}


def delivery_report(err: Any, msg: Any) -> None:
    """Callback for message delivery reports."""
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")


def create_topic_if_not_exists(
    admin_client: AdminClient,
    topic_name: str,
    num_partitions: int = 3,
    replication_factor: int = 1
) -> None:
    """Create Kafka topic if it doesn't exist."""
    existing_topics = admin_client.list_topics().topics
    
    if topic_name not in existing_topics:
        print(f"Creating topic: {topic_name}")
        new_topic = NewTopic(
            topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor
        )
        futures = admin_client.create_topics([new_topic])
        
        for topic, future in futures.items():
            try:
                future.result()
                print(f"Topic '{topic}' created successfully")
            except Exception as e:
                print(f"Failed to create topic '{topic}': {e}")
    else:
        print(f"Topic '{topic_name}' already exists")


def register_schema(
    schema_registry_url: str,
    topic_name: str,
    schema: dict
) -> JSONSerializer:
    """Register JSON schema with Schema Registry and return serializer."""
    schema_registry_client = SchemaRegistryClient({
        "url": schema_registry_url
    })
    
    json_schema = Schema(json.dumps(schema), schema_type="JSON")
    
    # Subject name follows topic naming convention
    subject_name = f"{topic_name}-value"
    
    # Register schema
    schema_id = schema_registry_client.register_schema(
        subject_name,
        json_schema
    )
    print(f"Schema registered with ID: {schema_id} for subject: {subject_name}")
    
    # Create serializer
    json_serializer = JSONSerializer(
        json.dumps(schema),
        schema_registry_client
    )
    
    return json_serializer


def read_events(input_path: Path) -> list[dict[str, Any]]:
    """Read events from JSONL file."""
    events = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def produce_events(
    producer: Producer,
    topic: str,
    events: list[dict[str, Any]],
    serializer: JSONSerializer | None = None,
    batch_size: int = 1000,
    key_field: str = "user_id"
) -> int:
    """Produce events to Kafka topic."""
    string_serializer = StringSerializer("utf-8")
    produced_count = 0
    
    print(f"Producing {len(events):,} events to topic: {topic}")
    start_time = time.time()
    
    for i, event in enumerate(events):
        # Use user_id as message key for partitioning
        key = event.get(key_field, str(i))
        
        # Serialize value
        if serializer:
            value = serializer(
                event,
                SerializationContext(topic, MessageField.VALUE)
            )
        else:
            value = json.dumps(event).encode("utf-8")
        
        # Produce message
        producer.produce(
            topic=topic,
            key=string_serializer(key),
            value=value,
            callback=delivery_report
        )
        
        produced_count += 1
        
        # Flush periodically
        if produced_count % batch_size == 0:
            producer.flush()
            elapsed = time.time() - start_time
            rate = produced_count / elapsed if elapsed > 0 else 0
            print(f"  Produced {produced_count:,} / {len(events):,} events ({rate:.0f} events/sec)")
    
    # Final flush
    producer.flush()
    
    elapsed = time.time() - start_time
    rate = produced_count / elapsed if elapsed > 0 else 0
    print(f"\nCompleted: {produced_count:,} events in {elapsed:.2f}s ({rate:.0f} events/sec)")
    
    return produced_count


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Produce business events to Kafka"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "sample" / "events.jsonl",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "-t", "--topic",
        type=str,
        default="events.raw.v1",
        help="Kafka topic name (default: events.raw.v1)"
    )
    parser.add_argument(
        "-b", "--bootstrap-servers",
        type=str,
        default="localhost:9093",
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "-s", "--schema-registry",
        type=str,
        default="http://localhost:8081",
        help="Schema Registry URL"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for flushing (default: 1000)"
    )
    parser.add_argument(
        "--skip-schema-registry",
        action="store_true",
        help="Skip Schema Registry and send raw JSON"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of events to produce"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run 'python data/sample_generator.py' first to generate sample data.")
        sys.exit(1)
    
    print("Configuration:")
    print(f"  Input:            {args.input}")
    print(f"  Topic:            {args.topic}")
    print(f"  Bootstrap:        {args.bootstrap_servers}")
    print(f"  Schema Registry:  {args.schema_registry}")
    print(f"  Batch Size:       {args.batch_size}")
    print()
    
    # Create admin client and ensure topic exists
    admin_client = AdminClient({
        "bootstrap.servers": args.bootstrap_servers
    })
    create_topic_if_not_exists(admin_client, args.topic)
    
    # Register schema (if not skipped)
    serializer = None
    if not args.skip_schema_registry:
        try:
            serializer = register_schema(
                args.schema_registry,
                args.topic,
                EVENT_SCHEMA
            )
        except Exception as e:
            print(f"Warning: Schema Registry registration failed: {e}")
            print("Continuing without schema validation...")
    
    # Create producer
    producer = Producer({
        "bootstrap.servers": args.bootstrap_servers,
        "client.id": "enterprise-ai-event-producer",
        "acks": "all",
        "retries": 3,
        "retry.backoff.ms": 100,
        "linger.ms": 5,
        "batch.num.messages": 1000,
        "compression.type": "snappy",
    })
    
    # Read events
    print("Reading events from file...")
    events = read_events(args.input)
    print(f"Loaded {len(events):,} events")
    
    # Apply limit if specified
    if args.limit:
        events = events[:args.limit]
        print(f"Limited to {len(events):,} events")
    
    # Produce events
    print()
    produced_count = produce_events(
        producer,
        args.topic,
        events,
        serializer,
        args.batch_size
    )
    
    print(f"\nDone! Produced {produced_count:,} events to {args.topic}")


if __name__ == "__main__":
    main()
