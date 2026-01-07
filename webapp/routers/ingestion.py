"""
Ingestion Router - Real-time Kafka Event Streaming with Live Data.
"""

import json
import asyncio
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Try to import Kafka consumer
try:
    from confluent_kafka import Consumer, KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("⚠️ confluent-kafka not available, using mock data")


router = APIRouter()

# Kafka configuration
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9093")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "events.raw.v1")

# In-memory event buffer for history
event_buffer: List[Dict] = []
MAX_BUFFER_SIZE = 1000


# ============================================================================
# Response Models
# ============================================================================

class KafkaEvent(BaseModel):
    """Single Kafka event."""
    id: str
    topic: str
    partition: int
    offset: int
    timestamp: str
    key: Optional[str]
    value: Dict[str, Any]


class IngestionStats(BaseModel):
    """Ingestion statistics."""
    events_per_second: float
    total_events_buffered: int
    partitions: int
    consumer_lag: int
    topics: List[str]
    kafka_connected: bool


class EventPage(BaseModel):
    """Paginated events response."""
    events: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int


# ============================================================================
# Kafka Consumer Helper
# ============================================================================

def create_kafka_consumer():
    """Create a Kafka consumer instance."""
    if not KAFKA_AVAILABLE:
        return None
    
    try:
        consumer = Consumer({
            "bootstrap.servers": KAFKA_BOOTSTRAP,
            "group.id": "webapp-consumer",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
        })
        consumer.subscribe([KAFKA_TOPIC])
        return consumer
    except Exception as e:
        print(f"❌ Failed to create Kafka consumer: {e}")
        return None


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/stats", response_model=IngestionStats)
async def get_ingestion_stats():
    """Get current ingestion statistics."""
    kafka_connected = False
    
    if KAFKA_AVAILABLE:
        try:
            # Quick check for Kafka connectivity
            consumer = Consumer({
                "bootstrap.servers": KAFKA_BOOTSTRAP,
                "group.id": "webapp-stats-check",
            })
            metadata = consumer.list_topics(timeout=2)
            kafka_connected = True
            consumer.close()
        except:
            kafka_connected = False
    
    return IngestionStats(
        events_per_second=len(event_buffer) / 60 if event_buffer else 0,
        total_events_buffered=len(event_buffer),
        partitions=3,
        consumer_lag=0,
        topics=[KAFKA_TOPIC, "events.canonical.v1"],
        kafka_connected=kafka_connected,
    )


@router.get("/events", response_model=EventPage)
async def get_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Get paginated events from buffer."""
    total = len(event_buffer)
    start = (page - 1) * page_size
    end = start + page_size
    
    events = event_buffer[start:end] if event_buffer else []
    
    return EventPage(
        events=events,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total > 0 else 1,
    )


@router.get("/stream")
async def stream_events():
    """Stream real-time events via Server-Sent Events (SSE)."""
    
    async def event_generator():
        consumer = None
        
        if KAFKA_AVAILABLE:
            try:
                consumer = Consumer({
                    "bootstrap.servers": KAFKA_BOOTSTRAP,
                    "group.id": f"webapp-stream-{datetime.now().timestamp()}",
                    "auto.offset.reset": "latest",
                    "enable.auto.commit": False,
                })
                consumer.subscribe([KAFKA_TOPIC])
                print(f"✅ Connected to Kafka: {KAFKA_BOOTSTRAP}")
            except Exception as e:
                print(f"❌ Kafka connection failed: {e}")
                consumer = None
        
        try:
            while True:
                if consumer:
                    # Try to get message from Kafka
                    msg = consumer.poll(timeout=0.5)
                    
                    if msg is not None and not msg.error():
                        try:
                            value = json.loads(msg.value().decode("utf-8"))
                            event = {
                                "id": value.get("id", f"evt_{msg.offset()}"),
                                "topic": msg.topic(),
                                "partition": msg.partition(),
                                "offset": msg.offset(),
                                "timestamp": value.get("timestamp", datetime.now().isoformat()),
                                "key": msg.key().decode("utf-8") if msg.key() else None,
                                "value": value,
                            }
                            
                            # Add to buffer
                            event_buffer.insert(0, event)
                            if len(event_buffer) > MAX_BUFFER_SIZE:
                                event_buffer.pop()
                            
                            yield f"data: {json.dumps(event)}\n\n"
                        except json.JSONDecodeError:
                            pass
                    elif msg and msg.error():
                        if msg.error().code() != KafkaError._PARTITION_EOF:
                            print(f"Kafka error: {msg.error()}")
                else:
                    # Mock event when Kafka not available
                    import random
                    mock_event = {
                        "id": f"mock_{int(datetime.now().timestamp() * 1000)}",
                        "topic": "events.raw.v1",
                        "partition": random.randint(0, 2),
                        "offset": random.randint(10000, 99999),
                        "timestamp": datetime.now().isoformat(),
                        "key": f"U{random.randint(1000, 9999)}",
                        "value": {
                            "event_type": random.choice(["order_placed", "page_view", "purchase"]),
                            "user_id": f"U{random.randint(1000, 9999)}",
                            "product_id": f"P{random.randint(100, 999)}",
                            "price": round(random.uniform(10, 500), 2),
                            "quantity": random.randint(1, 5),
                            "location": {"city": random.choice(["New York", "LA", "Chicago"])},
                        }
                    }
                    yield f"data: {json.dumps(mock_event)}\n\n"
                    await asyncio.sleep(0.5)
                
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spin
                
        finally:
            if consumer:
                consumer.close()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/topics")
async def get_topics():
    """Get list of Kafka topics."""
    topics = []
    
    if KAFKA_AVAILABLE:
        try:
            consumer = Consumer({
                "bootstrap.servers": KAFKA_BOOTSTRAP,
                "group.id": "webapp-topics-check",
            })
            metadata = consumer.list_topics(timeout=5)
            for topic in metadata.topics:
                if not topic.startswith("_"):  # Skip internal topics
                    topics.append({
                        "name": topic,
                        "partitions": len(metadata.topics[topic].partitions),
                        "status": "active",
                    })
            consumer.close()
        except Exception as e:
            print(f"Failed to list topics: {e}")
    
    if not topics:
        topics = [
            {"name": "events.raw.v1", "partitions": 3, "status": "unknown"},
            {"name": "events.canonical.v1", "partitions": 3, "status": "unknown"},
        ]
    
    return {"topics": topics}
