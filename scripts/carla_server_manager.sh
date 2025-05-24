#!/bin/zsh

# CARLA Server Manager Script
# Manages CARLA server for local development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default configuration
CARLA_DEFAULT_ROOT="/opt/carla"
CARLA_DEFAULT_PORT=2000
CARLA_PID_FILE="$PROJECT_ROOT/.carla_server.pid"
CARLA_LOG_FILE="$PROJECT_ROOT/data/logs/carla_server.log"

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
    echo "CARLA Server Manager"
    echo "==================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "COMMANDS:"
    echo "  start       Start CARLA server"
    echo "  stop        Stop CARLA server"
    echo "  restart     Restart CARLA server"
    echo "  status      Check CARLA server status"
    echo "  logs        Show CARLA server logs"
    echo "  kill        Force kill CARLA server"
    echo "  help        Show this help message"
    echo ""
    echo "OPTIONS:"
    echo "  --carla-root PATH    Path to CARLA installation (default: $CARLA_DEFAULT_ROOT)"
    echo "  --port PORT          CARLA server port (default: $CARLA_DEFAULT_PORT)"
    echo "  --headless           Run in headless mode with -RenderOffScreen"
    echo "  --quality LEVEL      Graphics quality: Low, Medium, High (default: Low)"
    echo "  --world-port PORT    World port (default: same as --port)"
    echo "  --resolution WxH     Screen resolution (default: 800x600)"
    echo "  --background         Run in background (daemon mode)"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 start --headless"
    echo "  $0 start --port 2001 --quality Medium"
    echo "  $0 status"
    echo "  $0 stop"
}

# Function to get CARLA root from config or environment
get_carla_root() {
    local carla_root=""
    
    # Try to get from app config
    if [ -f "$PROJECT_ROOT/app/config.py" ]; then
        carla_root=$(python3 -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
try:
    import app.config as config
    print(config.CARLA_ROOT)
except:
    pass
" 2>/dev/null)
    fi
    
    # Fallback to environment variable
    if [ -z "$carla_root" ] && [ ! -z "$CARLA_ROOT" ]; then
        carla_root="$CARLA_ROOT"
    fi
    
    # Final fallback to default
    if [ -z "$carla_root" ]; then
        carla_root="$CARLA_DEFAULT_ROOT"
    fi
    
    echo "$carla_root"
}

# Function to check if CARLA server is running
is_carla_running() {
    local port=${1:-$CARLA_DEFAULT_PORT}
    
    # Check if PID file exists and process is running
    if [ -f "$CARLA_PID_FILE" ]; then
        local pid=$(cat "$CARLA_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            # Process exists, check if it's actually CARLA and responsive
            if python3 -c "
import carla
import sys
try:
    client = carla.Client('localhost', $port)
    client.set_timeout(5.0)
    client.get_server_version()
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; then
                return 0  # CARLA is running and responsive
            fi
        else
            # PID file exists but process is dead, clean it up
            rm -f "$CARLA_PID_FILE"
        fi
    fi
    
    # Check if something is listening on the port
    if timeout 2 bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
        # Something is on the port, try to connect with CARLA client
        if python3 -c "
import carla
import sys
try:
    client = carla.Client('localhost', $port)
    client.set_timeout(5.0)
    client.get_server_version()
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; then
            return 0  # CARLA is running
        fi
    fi
    
    return 1  # CARLA is not running
}

# Function to start CARLA server
start_carla() {
    local carla_root="$1"
    local port="$2"
    local headless="$3"
    local quality="$4"
    local resolution="$5"
    local background="$6"
    
    print_status "Starting CARLA server..."
    print_status "CARLA Root: $carla_root"
    print_status "Port: $port"
    print_status "Headless: $headless"
    print_status "Quality: $quality"
    print_status "Resolution: $resolution"
    
    # Check if CARLA is already running
    if is_carla_running "$port"; then
        print_warning "CARLA server is already running on port $port"
        return 0
    fi
    
    # Validate CARLA installation
    if [ ! -d "$carla_root" ]; then
        print_error "CARLA directory not found: $carla_root"
        print_error "Please install CARLA or update the path with --carla-root"
        return 1
    fi
    
    if [ ! -f "$carla_root/CarlaUE4.sh" ]; then
        print_error "CarlaUE4.sh not found in: $carla_root"
        return 1
    fi
    
    # Create log directory
    mkdir -p "$(dirname "$CARLA_LOG_FILE")"
    
    # Build CARLA command
    local carla_cmd="$carla_root/CarlaUE4.sh"
    local carla_args="-carla-server -world-port=$port"
    
    # Add quality setting
    carla_args="$carla_args -quality-level=$quality"
    
    # Add resolution
    local width=$(echo "$resolution" | cut -d'x' -f1)
    local height=$(echo "$resolution" | cut -d'x' -f2)
    carla_args="$carla_args -resx=$width -resy=$height"
    
    # Add headless mode arguments
    if [ "$headless" = "true" ]; then
        carla_args="$carla_args -RenderOffScreen -opengl"
        print_status "Running in headless mode"
    fi
    
    print_status "Command: $carla_cmd $carla_args"
    
    # Change to CARLA directory
    cd "$carla_root"
    
    if [ "$background" = "true" ]; then
        # Run in background
        print_status "Starting CARLA server in background..."
        nohup $carla_cmd $carla_args > "$CARLA_LOG_FILE" 2>&1 &
        local pid=$!
        echo $pid > "$CARLA_PID_FILE"
        print_success "CARLA server started with PID: $pid"
        print_status "Log file: $CARLA_LOG_FILE"
        
        # Wait a bit and check if it's running
        sleep 5
        if is_carla_running "$port"; then
            print_success "CARLA server is running and responsive"
        else
            print_warning "CARLA server may not have started properly. Check logs: $0 logs"
        fi
    else
        # Run in foreground
        print_status "Starting CARLA server in foreground..."
        print_status "Press Ctrl+C to stop the server"
        exec $carla_cmd $carla_args
    fi
}

# Function to stop CARLA server
stop_carla() {
    print_status "Stopping CARLA server..."
    
    if [ -f "$CARLA_PID_FILE" ]; then
        local pid=$(cat "$CARLA_PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            print_status "Stopping CARLA server (PID: $pid)"
            kill "$pid"
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            if kill -0 "$pid" 2>/dev/null; then
                print_warning "CARLA server didn't stop gracefully, force killing..."
                kill -9 "$pid" 2>/dev/null || true
            fi
            
            print_success "CARLA server stopped"
        else
            print_warning "PID file exists but process is not running"
        fi
        rm -f "$CARLA_PID_FILE"
    else
        print_warning "No PID file found, CARLA server may not be running"
    fi
}

# Function to force kill CARLA processes
kill_carla() {
    print_status "Force killing all CARLA processes..."
    
    # Kill by process name
    pkill -f "CarlaUE4" 2>/dev/null || true
    pkill -f "carla" 2>/dev/null || true
    
    # Clean up PID file
    rm -f "$CARLA_PID_FILE"
    
    print_success "All CARLA processes killed"
}

# Function to check CARLA status
check_status() {
    local port=${1:-$CARLA_DEFAULT_PORT}
    
    print_status "Checking CARLA server status on port $port..."
    
    if is_carla_running "$port"; then
        print_success "CARLA server is running and responsive"
        
        # Get server version if possible
        local version=$(python3 -c "
import carla
try:
    client = carla.Client('localhost', $port)
    client.set_timeout(5.0)
    print(client.get_server_version())
except Exception as e:
    print('Unknown')
" 2>/dev/null)
        
        print_status "CARLA Version: $version"
        
        # Show PID if available
        if [ -f "$CARLA_PID_FILE" ]; then
            local pid=$(cat "$CARLA_PID_FILE")
            if kill -0 "$pid" 2>/dev/null; then
                print_status "Process ID: $pid"
            fi
        fi
    else
        print_warning "CARLA server is not running or not responsive on port $port"
        return 1
    fi
}

# Function to show logs
show_logs() {
    if [ -f "$CARLA_LOG_FILE" ]; then
        print_status "Showing CARLA server logs from: $CARLA_LOG_FILE"
        echo "========================================"
        tail -f "$CARLA_LOG_FILE"
    else
        print_warning "No log file found: $CARLA_LOG_FILE"
    fi
}

# Parse command line arguments
COMMAND=""
CARLA_ROOT=""
PORT=""
HEADLESS="false"
QUALITY="Low"
RESOLUTION="800x600"
BACKGROUND="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        start|stop|restart|status|logs|kill|help)
            COMMAND="$1"
            shift
            ;;
        --carla-root)
            CARLA_ROOT="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --world-port)
            PORT="$2"  # world-port is the same as port for our purposes
            shift 2
            ;;
        --headless)
            HEADLESS="true"
            shift
            ;;
        --quality)
            QUALITY="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --background)
            BACKGROUND="true"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set defaults
if [ -z "$CARLA_ROOT" ]; then
    CARLA_ROOT=$(get_carla_root)
fi

if [ -z "$PORT" ]; then
    PORT="$CARLA_DEFAULT_PORT"
fi

# Execute command
case "$COMMAND" in
    start)
        start_carla "$CARLA_ROOT" "$PORT" "$HEADLESS" "$QUALITY" "$RESOLUTION" "$BACKGROUND"
        ;;
    stop)
        stop_carla
        ;;
    restart)
        stop_carla
        sleep 2
        start_carla "$CARLA_ROOT" "$PORT" "$HEADLESS" "$QUALITY" "$RESOLUTION" "$BACKGROUND"
        ;;
    status)
        check_status "$PORT"
        ;;
    logs)
        show_logs
        ;;
    kill)
        kill_carla
        ;;
    help)
        show_help
        ;;
    "")
        print_error "No command specified"
        show_help
        exit 1
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac 