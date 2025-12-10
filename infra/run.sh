# Usage: ./run.sh [start|stop|restart|status|logs|create-topics|health]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.kafka.yml"
PROJECT_NAME="enterprise-ai-kafka"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Start Kafka infrastructure
start() {
    log_info "Starting Kafka infrastructure..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" up -d
    log_info "Waiting for services to be healthy..."
    sleep 5
    health_check
}

# Start with Kafka UI
start_with_ui() {
    log_info "Starting Kafka infrastructure with UI..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" --profile ui up -d
    log_info "Waiting for services to be healthy..."
    sleep 5
    health_check
    log_info "Kafka UI available at: http://localhost:8080"
}

# Stop Kafka infrastructure
stop() {
    log_info "Stopping Kafka infrastructure..."
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" --profile ui down
    log_info "Kafka infrastructure stopped."
}

# Restart Kafka infrastructure
restart() {
    stop
    sleep 2
    start
}

# Show status of services
status() {
    log_info "Service Status:"
    docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" ps
}

# Show logs
logs() {
    local service="${1:-}"
    if [ -n "$service" ]; then
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f "$service"
    else
        docker compose -f "$COMPOSE_FILE" -p "$PROJECT_NAME" logs -f
    fi
}

# Health check for all services
health_check() {
    log_info "Running health checks..."
    
    # Check Zookeeper
    if docker exec enterprise-ai-zookeeper nc -z localhost 2181 2>/dev/null; then
        echo -e "  Zookeeper:       ${GREEN}✓ Healthy${NC}"
    else
        echo -e "  Zookeeper:       ${RED}✗ Unhealthy${NC}"
    fi
    
    # Check Kafka
    if docker exec enterprise-ai-kafka kafka-broker-api-versions --bootstrap-server localhost:29092 2>/dev/null | grep -q "kafka:29092"; then
        echo -e "  Kafka:           ${GREEN}✓ Healthy${NC}"
    else
        echo -e "  Kafka:           ${RED}✗ Unhealthy${NC}"
    fi
    
    # Check Schema Registry
    if curl -s http://localhost:8081/subjects >/dev/null 2>&1; then
        echo -e "  Schema Registry: ${GREEN}✓ Healthy${NC}"
    else
        echo -e "  Schema Registry: ${RED}✗ Unhealthy${NC}"
    fi
    
    echo ""
    log_info "Endpoints:"
    echo "  Kafka Bootstrap:  localhost:9093"
    echo "  Schema Registry:  http://localhost:8081"
    echo "  Zookeeper:        localhost:2181"
}

# Create default topics
create_topics() {
    log_info "Creating default topics..."
    
    # Create events.raw.v1 topic
    docker exec enterprise-ai-kafka kafka-topics --create \
        --bootstrap-server localhost:29092 \
        --topic events.raw.v1 \
        --partitions 3 \
        --replication-factor 1 \
        --if-not-exists
    log_info "Created topic: events.raw.v1"
    
    # Create events.processed.v1 topic
    docker exec enterprise-ai-kafka kafka-topics --create \
        --bootstrap-server localhost:29092 \
        --topic events.processed.v1 \
        --partitions 3 \
        --replication-factor 1 \
        --if-not-exists
    log_info "Created topic: events.processed.v1"
    
    # List all topics
    log_info "Available topics:"
    docker exec enterprise-ai-kafka kafka-topics --list --bootstrap-server localhost:29092
}

# List topics
list_topics() {
    log_info "Listing topics..."
    docker exec enterprise-ai-kafka kafka-topics --list --bootstrap-server localhost:29092
}

# Consume messages from a topic
consume() {
    local topic="${1:-events.raw.v1}"
    local count="${2:-10}"
    log_info "Consuming $count messages from topic: $topic"
    docker exec enterprise-ai-kafka kafka-console-consumer \
        --bootstrap-server localhost:29092 \
        --topic "$topic" \
        --from-beginning \
        --max-messages "$count"
}

# Show help
show_help() {
    echo "Kafka Infrastructure Management Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start         Start Kafka infrastructure"
    echo "  start-ui      Start Kafka infrastructure with Kafka UI"
    echo "  stop          Stop Kafka infrastructure"
    echo "  restart       Restart Kafka infrastructure"
    echo "  status        Show service status"
    echo "  logs [svc]    Show logs (optionally for specific service)"
    echo "  health        Run health checks"
    echo "  create-topics Create default topics"
    echo "  list-topics   List all topics"
    echo "  consume [topic] [count]  Consume messages from topic"
    echo "  help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs kafka"
    echo "  $0 consume events.raw.v1 5"
}

# Main command dispatcher
case "${1:-help}" in
    start)
        start
        ;;
    start-ui)
        start_with_ui
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    health)
        health_check
        ;;
    create-topics)
        create_topics
        ;;
    list-topics)
        list_topics
        ;;
    consume)
        consume "$2" "$3"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
