"""
Kafka Tool Wrapper.

Provides functions to interact with Apache Kafka for monitoring and management.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy import for kafka-python
try:
    from kafka import KafkaAdminClient, KafkaConsumer
    from kafka.admin import ConfigResource, ConfigResourceType
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False


@dataclass
class ToolResult:
    """Standard result from tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    requires_confirmation: bool = False
    action_description: Optional[str] = None


class KafkaTool:
    """
    Tool wrapper for Apache Kafka operations.
    
    Provides safe functions for LLM agents to monitor Kafka clusters.
    """
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        timeout_ms: int = 10000,
    ):
        """
        Initialize Kafka tool.
        
        Args:
            bootstrap_servers: Kafka broker addresses.
            timeout_ms: Request timeout in milliseconds.
        """
        self.bootstrap_servers = bootstrap_servers
        self.timeout_ms = timeout_ms
        self._admin_client: Optional[Any] = None
    
    def _get_admin_client(self) -> Any:
        """Get or create admin client."""
        if not HAS_KAFKA:
            raise ImportError("kafka-python not installed. Run: pip install kafka-python")
        
        if self._admin_client is None:
            self._admin_client = KafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers,
                request_timeout_ms=self.timeout_ms,
            )
        return self._admin_client
    
    def get_consumer_lag(
        self,
        group_id: str,
        topics: Optional[List[str]] = None,
    ) -> ToolResult:
        """
        Check consumer group lag for specified topics.
        
        Args:
            group_id: Consumer group ID to check.
            topics: Optional list of topics. If None, checks all subscribed topics.
            
        Returns:
            ToolResult with lag information per partition.
            
        Example:
            >>> tool.get_consumer_lag("my-consumer-group")
            ToolResult(success=True, data={"total_lag": 1500, "partitions": [...]})
        """
        if not HAS_KAFKA:
            return ToolResult(
                success=False,
                data=None,
                error="kafka-python not installed",
            )
        
        try:
            admin = self._get_admin_client()
            
            # Get consumer group offsets
            group_offsets = admin.list_consumer_group_offsets(group_id)
            
            # Get topic end offsets
            consumer = KafkaConsumer(
                bootstrap_servers=self.bootstrap_servers,
                group_id=f"_lag_checker_{group_id}",
            )
            
            partitions = list(group_offsets.keys())
            if topics:
                partitions = [p for p in partitions if p.topic in topics]
            
            end_offsets = consumer.end_offsets(partitions)
            consumer.close()
            
            # Calculate lag
            lag_data = []
            total_lag = 0
            
            for tp, committed in group_offsets.items():
                if tp in end_offsets:
                    end = end_offsets[tp]
                    current = committed.offset if committed else 0
                    partition_lag = max(0, end - current)
                    total_lag += partition_lag
                    
                    lag_data.append({
                        "topic": tp.topic,
                        "partition": tp.partition,
                        "current_offset": current,
                        "end_offset": end,
                        "lag": partition_lag,
                    })
            
            return ToolResult(
                success=True,
                data={
                    "group_id": group_id,
                    "total_lag": total_lag,
                    "partitions": lag_data,
                    "partition_count": len(lag_data),
                },
            )
            
        except Exception as e:
            logger.error(f"Kafka error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def get_topic_stats(
        self,
        topic: str,
    ) -> ToolResult:
        """
        Get statistics for a Kafka topic.
        
        Args:
            topic: Topic name.
            
        Returns:
            ToolResult with topic partition information.
            
        Example:
            >>> tool.get_topic_stats("events.raw.v1")
            ToolResult(success=True, data={"partitions": 12, ...})
        """
        if not HAS_KAFKA:
            return ToolResult(
                success=False,
                data=None,
                error="kafka-python not installed",
            )
        
        try:
            admin = self._get_admin_client()
            
            # Get topic metadata
            metadata = admin.describe_topics([topic])
            
            if not metadata:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Topic '{topic}' not found",
                )
            
            topic_info = metadata[0]
            partitions = topic_info.get("partitions", [])
            
            # Get offsets for message count estimate
            consumer = KafkaConsumer(
                bootstrap_servers=self.bootstrap_servers,
            )
            
            from kafka import TopicPartition
            tps = [TopicPartition(topic, p["partition"]) for p in partitions]
            
            beginning = consumer.beginning_offsets(tps)
            end = consumer.end_offsets(tps)
            consumer.close()
            
            total_messages = sum(end[tp] - beginning[tp] for tp in tps)
            
            partition_data = []
            for p in partitions:
                tp = TopicPartition(topic, p["partition"])
                partition_data.append({
                    "partition": p["partition"],
                    "leader": p.get("leader"),
                    "replicas": p.get("replicas", []),
                    "isr": p.get("isr", []),
                    "messages": end[tp] - beginning[tp],
                })
            
            return ToolResult(
                success=True,
                data={
                    "topic": topic,
                    "partition_count": len(partitions),
                    "total_messages": total_messages,
                    "partitions": partition_data,
                },
            )
            
        except Exception as e:
            logger.error(f"Kafka error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def list_topics(
        self,
        include_internal: bool = False,
    ) -> ToolResult:
        """
        List all Kafka topics.
        
        Args:
            include_internal: Whether to include internal topics (starting with _).
            
        Returns:
            ToolResult with list of topics.
            
        Example:
            >>> tool.list_topics()
            ToolResult(success=True, data={"topics": ["events.raw.v1", ...]})
        """
        if not HAS_KAFKA:
            return ToolResult(
                success=False,
                data=None,
                error="kafka-python not installed",
            )
        
        try:
            admin = self._get_admin_client()
            topics = admin.list_topics()
            
            if not include_internal:
                topics = [t for t in topics if not t.startswith("_")]
            
            return ToolResult(
                success=True,
                data={
                    "topics": sorted(topics),
                    "count": len(topics),
                },
            )
            
        except Exception as e:
            logger.error(f"Kafka error: {e}")
            return ToolResult(success=False, data=None, error=str(e))
    
    def list_consumer_groups(self) -> ToolResult:
        """
        List all consumer groups.
        
        Returns:
            ToolResult with list of consumer groups.
        """
        if not HAS_KAFKA:
            return ToolResult(
                success=False,
                data=None,
                error="kafka-python not installed",
            )
        
        try:
            admin = self._get_admin_client()
            groups = admin.list_consumer_groups()
            
            group_data = [
                {
                    "group_id": g[0],
                    "protocol_type": g[1],
                }
                for g in groups
            ]
            
            return ToolResult(
                success=True,
                data={
                    "groups": group_data,
                    "count": len(group_data),
                },
            )
            
        except Exception as e:
            logger.error(f"Kafka error: {e}")
            return ToolResult(success=False, data=None, error=str(e))


# Convenience functions
_default_tool: Optional[KafkaTool] = None


def get_tool(bootstrap_servers: str = "localhost:9092") -> KafkaTool:
    """Get or create default Kafka tool instance."""
    global _default_tool
    if _default_tool is None:
        _default_tool = KafkaTool(bootstrap_servers=bootstrap_servers)
    return _default_tool


def get_consumer_lag(
    group_id: str,
    topics: Optional[List[str]] = None,
) -> ToolResult:
    """Get consumer lag. See KafkaTool.get_consumer_lag for details."""
    return get_tool().get_consumer_lag(group_id, topics)


def get_topic_stats(topic: str) -> ToolResult:
    """Get topic stats. See KafkaTool.get_topic_stats for details."""
    return get_tool().get_topic_stats(topic)
