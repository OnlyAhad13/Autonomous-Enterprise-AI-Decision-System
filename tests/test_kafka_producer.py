"""Tests for Kafka producer functionality."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest


# Skip all tests if Kafka is not running
def kafka_is_running() -> bool:
    """Check if Kafka is accessible."""
    try:
        result = subprocess.run(
            [
                "docker", "exec", "enterprise-ai-kafka",
                "kafka-topics", "--list", "--bootstrap-server", "localhost:9093"
            ],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


pytestmark = pytest.mark.skipif(
    not kafka_is_running(),
    reason="Kafka is not running. Start with: ./infra/run.sh start"
)


class TestKafkaProducer:
    """Integration tests for Kafka producer."""

    @pytest.fixture
    def test_topic(self) -> str:
        """Create a test topic and return its name."""
        topic_name = f"test.events.{int(time.time())}"
        
        # Create topic
        subprocess.run(
            [
                "docker", "exec", "enterprise-ai-kafka",
                "kafka-topics", "--create",
                "--bootstrap-server", "localhost:9093",
                "--topic", topic_name,
                "--partitions", "1",
                "--replication-factor", "1",
                "--if-not-exists"
            ],
            check=True,
            capture_output=True
        )
        
        yield topic_name
        
        # Cleanup - delete topic
        subprocess.run(
            [
                "docker", "exec", "enterprise-ai-kafka",
                "kafka-topics", "--delete",
                "--bootstrap-server", "localhost:9093",
                "--topic", topic_name
            ],
            capture_output=True
        )

    @pytest.fixture
    def sample_events_file(self, tmp_path: Path) -> Path:
        """Create a sample events file for testing."""
        events = [
            {
                "id": f"test-{i}",
                "timestamp": "2024-01-01T00:00:00Z",
                "user_id": f"user_{i}",
                "product_id": f"prod_{i}",
                "price": 10.0 + i,
                "quantity": i + 1,
                "location": "Test City, Test Country",
                "metadata": {"channel": "web", "device_type": "desktop", "session_id": f"sess_{i}"}
            }
            for i in range(5)
        ]
        
        events_file = tmp_path / "test_events.jsonl"
        with open(events_file, "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
        
        return events_file

    def test_producer_sends_messages_to_topic(
        self,
        test_topic: str,
        sample_events_file: Path
    ) -> None:
        """Test that producer successfully sends messages to Kafka topic."""
        # Run the producer
        result = subprocess.run(
            [
                sys.executable, "ingest/kafka_producer.py",
                "--input", str(sample_events_file),
                "--topic", test_topic,
                "--skip-schema-registry"
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=30
        )
        
        assert result.returncode == 0, f"Producer failed: {result.stderr}"
        assert "Produced 5 events" in result.stdout or "Done! Produced 5" in result.stdout

    def test_messages_appear_in_topic(
        self,
        test_topic: str,
        sample_events_file: Path
    ) -> None:
        """Test that produced messages can be consumed from the topic."""
        # Produce messages
        subprocess.run(
            [
                sys.executable, "ingest/kafka_producer.py",
                "--input", str(sample_events_file),
                "--topic", test_topic,
                "--skip-schema-registry"
            ],
            capture_output=True,
            cwd=Path(__file__).parent.parent,
            timeout=30,
            check=True
        )
        
        # Give Kafka a moment to commit
        time.sleep(2)
        
        # Consume messages
        result = subprocess.run(
            [
                "docker", "exec", "enterprise-ai-kafka",
                "kafka-console-consumer",
                "--bootstrap-server", "localhost:9093",
                "--topic", test_topic,
                "--from-beginning",
                "--max-messages", "5",
                "--timeout-ms", "10000"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Verify messages were received
        assert result.returncode == 0, f"Consumer failed: {result.stderr}"
        
        # Parse consumed messages
        messages = [line for line in result.stdout.strip().split("\n") if line]
        assert len(messages) == 5, f"Expected 5 messages, got {len(messages)}"
        
        # Verify message content
        for msg in messages:
            event = json.loads(msg)
            assert "id" in event
            assert "timestamp" in event
            assert "user_id" in event
            assert "product_id" in event
            assert "price" in event
            assert "quantity" in event
            assert "location" in event
            assert "metadata" in event

    def test_message_key_is_user_id(
        self,
        test_topic: str,
        sample_events_file: Path
    ) -> None:
        """Test that messages are keyed by user_id."""
        # Produce messages
        subprocess.run(
            [
                sys.executable, "ingest/kafka_producer.py",
                "--input", str(sample_events_file),
                "--topic", test_topic,
                "--skip-schema-registry"
            ],
            capture_output=True,
            cwd=Path(__file__).parent.parent,
            timeout=30,
            check=True
        )
        
        time.sleep(2)
        
        # Consume with key printing
        result = subprocess.run(
            [
                "docker", "exec", "enterprise-ai-kafka",
                "kafka-console-consumer",
                "--bootstrap-server", "localhost:9093",
                "--topic", test_topic,
                "--from-beginning",
                "--max-messages", "5",
                "--timeout-ms", "10000",
                "--property", "print.key=true",
                "--property", "key.separator=|"
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0
        
        messages = [line for line in result.stdout.strip().split("\n") if line]
        for msg in messages:
            key, value = msg.split("|", 1)
            assert key.startswith("user_"), f"Key should be user_id, got: {key}"


class TestKafkaTopicCreation:
    """Tests for topic creation functionality."""

    def test_create_topics_script(self) -> None:
        """Test that create-topics command works."""
        result = subprocess.run(
            ["./run.sh", "create-topics"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent / "infra",
            timeout=30
        )
        
        # Should succeed or topic already exists
        assert result.returncode == 0 or "already exists" in result.stdout.lower()

    def test_list_topics_includes_events_topic(self) -> None:
        """Test that events.raw.v1 topic exists after creation."""
        # First ensure topic is created
        subprocess.run(
            ["./run.sh", "create-topics"],
            capture_output=True,
            cwd=Path(__file__).parent.parent / "infra",
            timeout=30
        )
        
        # List topics
        result = subprocess.run(
            [
                "docker", "exec", "enterprise-ai-kafka",
                "kafka-topics", "--list",
                "--bootstrap-server", "localhost:9093"
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        assert result.returncode == 0
        assert "events.raw.v1" in result.stdout
