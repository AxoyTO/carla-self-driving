#!/bin/bash

# CARLA RL Training Docker Deployment Script
# Manages Docker Compose services for CARLA RL training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    echo "CARLA RL Training Docker Deployment"
    echo "==================================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "BASIC COMMANDS:"
    echo "  up          Start all services"
    echo "  down        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show status of all services"
    echo "  build       Build and start services"
    echo ""
    echo "SERVICE-SPECIFIC COMMANDS:"
    echo "  start-training    Start only training service"
    echo "  stop-training     Stop only training service"
    echo "  start-carla       Start only CARLA server"
    echo "  stop-carla        Stop only CARLA server"
    echo "  restart-carla     Restart CARLA server"
    echo ""
    echo "MONITORING COMMANDS:"
    echo "  logs             Show training application logs"
    echo "  logs-carla       Show CARLA server logs"
    echo "  logs-tensorboard Show TensorBoard logs"
    echo "  logs-all         Show all service logs"
    echo ""
    echo "DEVELOPMENT COMMANDS:"
    echo "  shell            Access training container shell"
    echo "  shell-carla      Access CARLA server container shell"
    echo "  health           Check health of all services"
    echo "  cleanup          Clean up stopped containers and volumes"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 up                    # Start all services"
    echo "  $0 logs                  # View training logs"
    echo "  $0 restart-carla         # Restart CARLA server"
    echo "  $0 shell                 # Access training container"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running or not accessible"
        print_error "Please start Docker and try again"
        exit 1
    fi
}

# Function to check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
        print_error "docker-compose or 'docker compose' command not found"
        print_error "Please install Docker Compose and try again"
        exit 1
    fi
    
    # Use docker compose if available, otherwise docker-compose
    if command -v docker compose &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
}

# Function to wait for service to be healthy
wait_for_service() {
    local service="$1"
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service to be healthy..."
    
    while [ $attempt -le $max_attempts ]; do
        if $COMPOSE_CMD ps --format table | grep -q "$service.*healthy\|$service.*running"; then
            print_success "$service is ready"
            return 0
        fi
        
        print_status "Waiting for $service... (attempt $attempt/$max_attempts)"
        sleep 5
        attempt=$((attempt + 1))
    done
    
    print_warning "$service is not healthy after $max_attempts attempts"
    return 1
}

# Change to project root
cd "$PROJECT_ROOT"

# Check prerequisites
check_docker
check_docker_compose

print_status "Using compose command: $COMPOSE_CMD"
print_status "Project root: $PROJECT_ROOT"

# Parse command
case "${1:-help}" in
    up)
        print_status "Starting CARLA RL training services..."
        $COMPOSE_CMD up -d
        print_success "Services started"
        
        # Wait for CARLA server to be ready
        wait_for_service "carla-server"
        
        print_status "Services are ready!"
        print_status "Training logs: $0 logs"
        print_status "TensorBoard: http://localhost:6007"
        ;;
        
    down)
        print_status "Stopping CARLA RL training services..."
        $COMPOSE_CMD down
        print_success "Services stopped"
        ;;
        
    restart)
        print_status "Restarting CARLA RL training services..."
        $COMPOSE_CMD restart
        print_success "Services restarted"
        ;;
        
    status)
        print_status "Service status:"
        $COMPOSE_CMD ps
        ;;
        
    build)
        print_status "Building and starting services..."
        $COMPOSE_CMD up -d --build
        print_success "Services built and started"
        ;;
        
    start-training)
        print_status "Starting training service..."
        $COMPOSE_CMD up -d carla-rl-training
        print_success "Training service started"
        ;;
        
    stop-training)
        print_status "Stopping training service..."
        $COMPOSE_CMD stop carla-rl-training
        print_success "Training service stopped"
        ;;
        
    start-carla)
        print_status "Starting CARLA server..."
        $COMPOSE_CMD up -d carla-server
        wait_for_service "carla-server"
        print_success "CARLA server started and ready"
        ;;
        
    stop-carla)
        print_status "Stopping CARLA server..."
        $COMPOSE_CMD stop carla-server
        print_success "CARLA server stopped"
        ;;
        
    restart-carla)
        print_status "Restarting CARLA server..."
        $COMPOSE_CMD restart carla-server
        wait_for_service "carla-server"
        print_success "CARLA server restarted and ready"
        ;;
        
    logs)
        print_status "Showing training application logs..."
        $COMPOSE_CMD logs -f carla-rl-training
        ;;
        
    logs-carla)
        print_status "Showing CARLA server logs..."
        $COMPOSE_CMD logs -f carla-server
        ;;
        
    logs-tensorboard)
        print_status "Showing TensorBoard logs..."
        $COMPOSE_CMD logs -f tensorboard
        ;;
        
    logs-all)
        print_status "Showing all service logs..."
        $COMPOSE_CMD logs -f
        ;;
        
    shell)
        print_status "Accessing training container shell..."
        $COMPOSE_CMD exec carla-rl-training /bin/bash
        ;;
        
    shell-carla)
        print_status "Accessing CARLA server container shell..."
        $COMPOSE_CMD exec carla-server /bin/bash
        ;;
        
    health)
        print_status "Checking service health..."
        echo ""
        
        # Check each service
        services=("carla-server" "carla-rl-training" "tensorboard")
        for service in "${services[@]}"; do
            if $COMPOSE_CMD ps --format table | grep -q "$service.*running"; then
                print_success "$service: Running"
            else
                print_error "$service: Not running"
            fi
        done
        
        echo ""
        print_status "Detailed status:"
        $COMPOSE_CMD ps
        ;;
        
    cleanup)
        print_status "Cleaning up stopped containers and unused volumes..."
        $COMPOSE_CMD down --remove-orphans
        docker system prune -f
        docker volume prune -f
        print_success "Cleanup completed"
        ;;
        
    help|--help|-h)
        show_help
        ;;
        
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac 