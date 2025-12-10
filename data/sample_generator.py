"""
Sample Event Generator

Generates synthetic business events using Faker for testing and development.
Outputs to Parquet and JSONL formats.

Usage:
    python sample_generator.py --count 50000 --seed 42
    python sample_generator.py -n 1000 -s 123 --output-dir ./output
"""

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from faker import Faker
except ImportError:
    raise ImportError("Faker is required. Install with: pip install faker")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("PyArrow is required. Install with: pip install pyarrow")


# Schema field definitions for validation
SCHEMA_FIELDS = {
    "id": {"type": str, "required": True},
    "timestamp": {"type": str, "required": True},
    "user_id": {"type": str, "required": True},
    "product_id": {"type": str, "required": True},
    "price": {"type": float, "required": True},
    "quantity": {"type": int, "required": True},
    "location": {"type": str, "required": True},
    "metadata": {"type": dict, "required": False},
}

CHANNELS = ["web", "mobile", "api", "pos"]
DEVICE_TYPES = ["desktop", "mobile", "tablet", "smart_tv", "iot"]
REFERRERS = ["google", "facebook", "twitter", "direct", "email", "affiliate", None]
PRODUCT_CATEGORIES = ["electronics", "clothing", "food", "home", "sports", "books"]


def generate_event(fake: Faker, event_time: datetime) -> dict[str, Any]:
    """Generate a single business event."""
    channel = random.choice(CHANNELS)
    
    # Build metadata
    metadata = {
        "channel": channel,
        "device_type": random.choice(DEVICE_TYPES) if channel != "pos" else "pos_terminal",
        "session_id": f"sess_{fake.uuid4()[:12]}",
    }
    
    # Add optional fields
    referrer = random.choice(REFERRERS)
    if referrer:
        metadata["referrer"] = referrer
    
    # Add tags occasionally
    if random.random() > 0.7:
        metadata["tags"] = random.sample(
            PRODUCT_CATEGORIES,
            k=random.randint(1, 3)
        )
    
    return {
        "id": fake.uuid4(),
        "timestamp": event_time.isoformat(),
        "user_id": f"usr_{fake.uuid4()[:12]}",
        "product_id": f"prod_{fake.uuid4()[:8]}",
        "price": round(random.uniform(0.99, 999.99), 2),
        "quantity": random.randint(1, 10),
        "location": f"{fake.city()}, {fake.country()}",
        "metadata": metadata,
    }


def generate_events(count: int, seed: int | None = None) -> list[dict[str, Any]]:
    """Generate a list of synthetic business events."""
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)
    
    fake = Faker()
    events = []
    
    # Generate events spread over the last 30 days
    # Use fixed reference time when seed is set for reproducibility
    if seed is not None:
        end_time = datetime(2024, 12, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=30)
    time_range_seconds = int((end_time - start_time).total_seconds())
    
    print(f"Generating {count:,} events...")
    
    for i in range(count):
        # Random timestamp within the last 30 days
        random_offset = random.randint(0, time_range_seconds)
        event_time = start_time + timedelta(seconds=random_offset)
        
        event = generate_event(fake, event_time)
        events.append(event)
        
        # Progress indicator
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1:,} / {count:,} events...")
    
    # Sort by timestamp
    events.sort(key=lambda x: x["timestamp"])
    
    return events


def write_jsonl(events: list[dict[str, Any]], output_path: Path) -> None:
    """Write events to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    
    print(f"Written {len(events):,} events to {output_path}")


def write_parquet(events: list[dict[str, Any]], output_path: Path) -> None:
    """Write events to Parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define PyArrow schema
    schema = pa.schema([
        ("id", pa.string()),
        ("timestamp", pa.string()),
        ("user_id", pa.string()),
        ("product_id", pa.string()),
        ("price", pa.float64()),
        ("quantity", pa.int32()),
        ("location", pa.string()),
        ("metadata", pa.string()),  # JSON serialized
    ])
    
    # Convert metadata to JSON strings for Parquet storage
    rows = []
    for event in events:
        row = event.copy()
        row["metadata"] = json.dumps(event["metadata"])
        rows.append(row)
    
    # Create table
    table = pa.Table.from_pylist(rows, schema=schema)
    
    # Write with compression
    pq.write_table(
        table,
        output_path,
        compression="snappy",
        row_group_size=10000,
    )
    
    print(f"Written {len(events):,} events to {output_path}")


def validate_event(event: dict[str, Any]) -> bool:
    """Validate an event against the schema."""
    for field, spec in SCHEMA_FIELDS.items():
        if spec["required"] and field not in event:
            return False
        if field in event and not isinstance(event[field], spec["type"]):
            return False
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic business events for testing"
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=50000,
        help="Number of events to generate (default: 50000)"
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).parent / "sample",
        help="Output directory (default: data/sample)"
    )
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Count: {args.count:,}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output_dir}")
    print()
    
    # Generate events
    events = generate_events(args.count, args.seed)
    
    # Validate sample
    sample_size = min(100, len(events))
    valid_count = sum(1 for e in events[:sample_size] if validate_event(e))
    print(f"\nValidation: {valid_count}/{sample_size} sampled events are schema-compliant")
    
    # Write outputs
    print()
    write_parquet(events, args.output_dir / "events.parquet")
    write_jsonl(events, args.output_dir / "events.jsonl")
    
    print(f"\nDone! Generated {len(events):,} events.")


if __name__ == "__main__":
    main()
