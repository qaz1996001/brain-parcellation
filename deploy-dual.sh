#!/bin/bash

set -e

ACTION=$1  # start|stop|restart|status

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

start_production() {
    log_info "Starting Production instance..."

    # Start Docker services for production
    docker-compose --env-file .env.production --project-name brain-prod up -d

    # Wait for services to be ready
    sleep 5

    # Start backend application
    log_info "Starting production backend on port 8000..."
    # Note: Adjust this command based on your actual startup method
    # ENV=production APP_PORT=8000 ./brain-parcellation-start.sh production &

    log_info "Production instance started"
}

start_testing() {
    log_info "Starting Testing instance..."

    # Start Docker services for testing
    docker-compose --env-file .env.testing --project-name brain-test up -d

    # Wait for services to be ready
    sleep 5

    # Start backend application
    log_info "Starting testing backend on port 8001..."
    # Note: Adjust this command based on your actual startup method
    # ENV=testing APP_PORT=8001 ./brain-parcellation-start.sh testing &

    log_info "Testing instance started"
}

stop_production() {
    log_info "Stopping Production instance..."
    docker-compose --project-name brain-prod down
    log_info "Production instance stopped"
}

stop_testing() {
    log_info "Stopping Testing instance..."
    docker-compose --project-name brain-test down
    log_info "Testing instance stopped"
}

show_status() {
    echo ""
    log_info "=== Production Instance Status ==="
    echo ""

    # Check production health
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Health Check:"
        curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "Backend not responding"
    else
        log_warn "Production backend (port 8000) not responding"
    fi

    echo ""
    echo "Docker Containers:"
    docker-compose --project-name brain-prod ps

    echo ""
    log_info "=== Testing Instance Status ==="
    echo ""

    # Check testing health
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo "Health Check:"
        curl -s http://localhost:8001/health | python -m json.tool 2>/dev/null || echo "Backend not responding"
    else
        log_warn "Testing backend (port 8001) not responding"
    fi

    echo ""
    echo "Docker Containers:"
    docker-compose --project-name brain-test ps

    echo ""
    log_info "=== Port Usage ==="
    echo "Checking port occupancy..."
    netstat -tuln 2>/dev/null | grep -E ':8000|:8001|:5672|:5673|:6379|:6380|:15672|:15673|:15433' || echo "No ports in use"
}

case $ACTION in
  start)
    log_info "Starting both Production and Testing instances..."
    start_production
    echo ""
    start_testing
    echo ""
    log_info "Both instances started successfully!"
    echo ""
    log_info "Access points:"
    echo "  - Production: http://localhost:8000/health"
    echo "  - Testing:    http://localhost:8001/health"
    ;;

  stop)
    log_info "Stopping both instances..."
    stop_production
    echo ""
    stop_testing
    echo ""
    log_info "Both instances stopped"
    ;;

  restart)
    log_info "Restarting both instances..."
    $0 stop
    sleep 3
    $0 start
    ;;

  status)
    show_status
    ;;

  start-prod)
    start_production
    ;;

  stop-prod)
    stop_production
    ;;

  start-test)
    start_testing
    ;;

  stop-test)
    stop_testing
    ;;

  *)
    echo "Usage: $0 {start|stop|restart|status|start-prod|stop-prod|start-test|stop-test}"
    echo ""
    echo "Commands:"
    echo "  start       - Start both production and testing instances"
    echo "  stop        - Stop both instances"
    echo "  restart     - Restart both instances"
    echo "  status      - Show status of both instances"
    echo "  start-prod  - Start only production instance"
    echo "  stop-prod   - Stop only production instance"
    echo "  start-test  - Start only testing instance"
    echo "  stop-test   - Stop only testing instance"
    exit 1
    ;;
esac
