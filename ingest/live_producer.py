#!/usr/bin/env python3
"""
Live Data Generator - Continuous Event Producer

Generates realistic business events and sends them to Kafka in real-time.
This creates a continuous stream of events for the dashboard to display.

Usage:
    python live_producer.py --rate 10  # 10 events per second
"""

import argparse
import json
import random
import signal
import sys
import time
import uuid
from datetime import datetime
from typing import Any

try:
    from confluent_kafka import Producer
except ImportError:
    print("confluent-kafka required. Install: pip install confluent-kafka")
    sys.exit(1)


# Configuration
PRODUCTS = [
    {"id": f"P{i:03d}", "name": f"Product {i}", "category": cat, "base_price": random.uniform(10, 500)}
    for i, cat in enumerate(["Electronics", "Clothing", "Home", "Sports", "Books"] * 20, 1)
]

USERS = [f"U{i:04d}" for i in range(1, 1001)]

LOCATIONS = [
    {"city": "New York", "region": "Northeast", "lat": 40.7128, "lon": -74.0060},
    {"city": "Los Angeles", "region": "West", "lat": 34.0522, "lon": -118.2437},
    {"city": "Chicago", "region": "Midwest", "lat": 41.8781, "lon": -87.6298},
    {"city": "Houston", "region": "South", "lat": 29.7604, "lon": -95.3698},
    {"city": "Phoenix", "region": "West", "lat": 33.4484, "lon": -112.0740},
    {"city": "Philadelphia", "region": "Northeast", "lat": 39.9526, "lon": -75.1652},
    {"city": "San Antonio", "region": "South", "lat": 29.4241, "lon": -98.4936},
    {"city": "San Diego", "region": "West", "lat": 32.7157, "lon": -117.1611},
    {"city": "Dallas", "region": "South", "lat": 32.7767, "lon": -96.7970},
    {"city": "Seattle", "region": "West", "lat": 47.6062, "lon": -122.3321},
]

EVENT_TYPES = ["order_placed", "page_view", "item_click", "add_to_cart", "purchase", "checkout_started"]


class LiveProducer:
    """Continuous event producer for Kafka."""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9093",
        topic: str = "events.raw.v1",
        rate: float = 10.0,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.rate = rate  # events per second
        self.running = False
        self.events_sent = 0
        self.start_time = None
        
        # Create producer
        self.producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "client.id": "live-producer",
            "acks": "all",
            "linger.ms": 5,
            "batch.size": 16384,
        })
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals."""
        print("\n🛑 Shutting down producer...")
        self.running = False
    
    def _delivery_callback(self, err, msg):
        """Callback for delivery reports."""
        if err:
            print(f"❌ Delivery failed: {err}")
        else:
            self.events_sent += 1
    
    def generate_event(self) -> dict[str, Any]:
        """Generate a single realistic event."""
        event_type = random.choice(EVENT_TYPES)
        product = random.choice(PRODUCTS)
        location = random.choice(LOCATIONS)
        
        # Price variation
        price = product["base_price"] * random.uniform(0.9, 1.1)
        quantity = random.choices([1, 2, 3, 4, 5], weights=[50, 25, 15, 7, 3])[0]
        
        return {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": random.choice(USERS),
            "product_id": product["id"],
            "product_name": product["name"],
            "category": product["category"],
            "price": round(price, 2),
            "quantity": quantity,
            "total": round(price * quantity, 2),
            "location": {
                "city": location["city"],
                "region": location["region"],
                "latitude": location["lat"],
                "longitude": location["lon"],
            },
            "metadata": {
                "session_id": f"sess_{random.randint(100000, 999999)}",
                "device": random.choice(["mobile", "desktop", "tablet"]),
                "browser": random.choice(["Chrome", "Firefox", "Safari", "Edge"]),
            }
        }
    
    def run(self):
        """Start producing events continuously."""
        self.running = True
        self.start_time = time.time()
        interval = 1.0 / self.rate
        
        print(f"🚀 Starting live producer")
        print(f"   Bootstrap: {self.bootstrap_servers}")
        print(f"   Topic: {self.topic}")
        print(f"   Rate: {self.rate} events/sec")
        print(f"   Press Ctrl+C to stop\n")
        
        last_status = time.time()
        
        while self.running:
            try:
                # Generate and send event
                event = self.generate_event()
                
                self.producer.produce(
                    topic=self.topic,
                    key=event["user_id"].encode("utf-8"),
                    value=json.dumps(event).encode("utf-8"),
                    callback=self._delivery_callback,
                )
                
                # Poll for delivery callbacks
                self.producer.poll(0)
                
                # Status update every 5 seconds
                if time.time() - last_status >= 5:
                    elapsed = time.time() - self.start_time
                    rate = self.events_sent / elapsed if elapsed > 0 else 0
                    print(f"📊 Sent: {self.events_sent:,} events | Rate: {rate:.1f}/sec | Elapsed: {elapsed:.0f}s")
                    last_status = time.time()
                
                # Rate limiting
                time.sleep(interval)
                
            except BufferError:
                # Queue is full, wait a bit
                self.producer.poll(100)
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(1)
        
        # Flush remaining messages
        print("📤 Flushing remaining messages...")
        self.producer.flush(timeout=10)
        
        elapsed = time.time() - self.start_time
        print(f"\n✅ Producer stopped")
        print(f"   Total events: {self.events_sent:,}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Avg rate: {self.events_sent / elapsed:.1f}/sec")


def main():
    parser = argparse.ArgumentParser(description="Live event producer for Kafka")
    parser.add_argument("--bootstrap", default="localhost:9093", help="Kafka bootstrap servers")
    parser.add_argument("--topic", default="events.raw.v1", help="Kafka topic")
    parser.add_argument("--rate", type=float, default=10.0, help="Events per second")
    
    args = parser.parse_args()
    
    producer = LiveProducer(
        bootstrap_servers=args.bootstrap,
        topic=args.topic,
        rate=args.rate,
    )
    producer.run()


if __name__ == "__main__":
    main()
