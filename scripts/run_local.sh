#!/bin/zsh

# Local Development Runner for CARLA RL Training Application
# This script sets up the environment and runs the application locally

set -e

echo "CARLA RL Local Development Runner"
echo "===================================="

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Project root: $PROJECT_ROOT"

# Initialize conda for zsh
# Try common conda initialization paths
CONDA_INIT_ATTEMPTED=false

# Method 1: Try sourcing zshrc
if [ -f ~/.zshrc ]; then
    source ~/.zshrc 2>/dev/null || true
    CONDA_INIT_ATTEMPTED=true
fi

# Method 2: Try common conda installation paths
if ! command -v conda &> /dev/null && [ "$CONDA_INIT_ATTEMPTED" = false ]; then
    for conda_path in ~/miniconda3 ~/anaconda3 /opt/conda /usr/local/miniconda3 /usr/local/anaconda3; do
        if [ -f "$conda_path/etc/profile.d/conda.sh" ]; then
            source "$conda_path/etc/profile.d/conda.sh"
            echo "Initialized conda from: $conda_path"
            break
        fi
    done
fi

# Check if conda is now available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda not found."
    echo "   Please ensure conda is installed and properly initialized."
    echo "   Try running: conda init zsh"
    echo "   Then restart your shell and try again."
    exit 1
fi

# Check if carla4 environment exists
if ! conda env list | grep -q "^carla4 "; then
    echo "Error: conda environment 'carla4' not found."
    echo "   Please create the environment or adjust the script to use your environment name."
    echo "   Available environments:"
    conda env list | grep -v "^#"
    exit 1
fi

# Activate carla4 environment
echo "Activating conda environment: carla4"
conda activate carla4

# Parse command line arguments first
if [ "$1" = "help" ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "COMMANDS:"
    echo "  (none)      Run the main training application"
    echo "  help        Show this help message"
    echo "  test        Run basic import tests"
    echo "  config      Show current configuration"
    echo "  tensorboard Start TensorBoard server"
    echo "  viz         Run with pygame visualization"
    echo "  carla       Manage CARLA server (start/stop/status)"
    echo ""
    echo "CARLA SERVER COMMANDS:"
    echo "  carla start     Start CARLA server in background (headless)"
    echo "  carla stop      Stop CARLA server"
    echo "  carla status    Check CARLA server status"
    echo "  carla restart   Restart CARLA server"
    echo "  carla logs      Show CARLA server logs"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                    # Run training"
    echo "  $0 viz               # Run with visualization"
    echo "  $0 test              # Test imports"
    echo "  $0 carla start       # Start CARLA server"
    echo "  $0 carla status      # Check CARLA status"
    echo "  $0 tensorboard       # Start TensorBoard"
    exit 0
elif [ "$1" = "test" ]; then
    echo "Running basic import tests..."
    python -c "
try:
    import app.config as config
    print('app.config imported successfully')
    
    from utils.config_loader import load_training_config
    print('utils.config_loader imported successfully')
    
    from app.environments.carla_env import CarlaEnv
    print('app.environments.carla_env imported successfully')
    
    print('All basic imports successful!')
except ImportError as e:
    print(f'Import error: {e}')
    exit(1)
"
    exit 0
elif [ "$1" = "config" ]; then
    echo "Current configuration:"
    python -c "
import app.config as config
from utils.config_loader import load_training_config, load_environment_config

print('Core Configuration:')
print(f'   CARLA_ROOT: {config.CARLA_ROOT}')
print(f'   DEVICE: {config.DEVICE}')
print(f'   LOG_LEVEL: {config.LOG_LEVEL}')

print('\\nTraining Configuration:')
train_config = load_training_config()
if train_config and 'training' in train_config:
    tc = train_config['training']
    print(f'   Total Episodes: {tc.get(\"total_episodes\", \"N/A\")}')
    print(f'   Learning Rate: {tc.get(\"learning_rate\", \"N/A\")}')
    print(f'   Batch Size: {tc.get(\"batch_size\", \"N/A\")}')
else:
    print('   No training configuration found')

print('\\nEnvironment Configuration:')
env_config = load_environment_config()
if env_config and 'environment' in env_config:
    ec = env_config['environment']
    print(f'   CARLA Host: {ec.get(\"carla_server\", {}).get(\"host\", \"N/A\")}')
    print(f'   CARLA Port: {ec.get(\"carla_server\", {}).get(\"port\", \"N/A\")}')
    print(f'   Town: {ec.get(\"world\", {}).get(\"town\", \"N/A\")}')
else:
    print('   No environment configuration found')
"
    exit 0
elif [ "$1" = "tensorboard" ]; then
    echo "Starting TensorBoard..."
    if [ -d "$PROJECT_ROOT/data/tensorboard_logs" ]; then
        echo "   Logs directory: $PROJECT_ROOT/data/tensorboard_logs"
        echo "   URL: http://localhost:6006"
        tensorboard --logdir="$PROJECT_ROOT/data/tensorboard_logs" --host=0.0.0.0 --port=6006
    else
        echo "TensorBoard logs directory not found: $PROJECT_ROOT/data/tensorboard_logs"
        exit 1
    fi
    exit 0
elif [ "$1" = "carla" ]; then
    # CARLA server management
    if [ -z "$2" ]; then
        echo "CARLA command required. Use: start, stop, status, restart, logs"
        echo "   Example: $0 carla start"
        exit 1
    fi
    
    case "$2" in
        start)
            echo "Starting CARLA server in headless mode..."
            "$PROJECT_ROOT/scripts/carla_server_manager.sh" start --headless --background
            ;;
        stop)
            echo "Stopping CARLA server..."
            "$PROJECT_ROOT/scripts/carla_server_manager.sh" stop
            ;;
        status)
            echo "Checking CARLA server status..."
            "$PROJECT_ROOT/scripts/carla_server_manager.sh" status
            ;;
        restart)
            echo "Restarting CARLA server..."
            "$PROJECT_ROOT/scripts/carla_server_manager.sh" restart --headless --background
            ;;
        logs)
            echo "Showing CARLA server logs..."
            "$PROJECT_ROOT/scripts/carla_server_manager.sh" logs
            ;;
        *)
            echo "Unknown CARLA command: $2"
            echo "   Available commands: start, stop, status, restart, logs"
            exit 1
            ;;
    esac
    exit 0
fi

# For running commands, do the full environment check
echo "Checking dependencies..."
if ! python -c "import torch, numpy, gymnasium, pygame, yaml" 2>/dev/null; then
    echo "Some dependencies are missing. Installing from requirements.txt..."
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        python -m pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        echo "Error: requirements.txt not found at $PROJECT_ROOT/requirements.txt"
        exit 1
    fi
fi

# Check for CARLA installation
echo "Checking CARLA installation..."
if ! python -c "import carla" 2>/dev/null; then
    echo "CARLA Python API not found."
    echo "   Please install CARLA and the Python API:"
    echo "   1. Download CARLA from https://github.com/carla-simulator/carla/releases"
    echo "   2. Extract it to a directory (e.g., /opt/carla)"
    echo "   3. Install the Python API: pip install /path/to/carla/PythonAPI/carla/dist/carla-*.whl"
    echo "   4. Update app/config.py with the correct CARLA_ROOT path"
    echo ""
    read -q "REPLY?Continue anyway? (y/N): "
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create necessary directories
echo "Creating data directories..."
mkdir -p "$PROJECT_ROOT/data/logs"
mkdir -p "$PROJECT_ROOT/data/sensor_capture"
mkdir -p "$PROJECT_ROOT/data/tensorboard_logs"
mkdir -p "$PROJECT_ROOT/models/model_checkpoints"

# Check if CARLA server is running (and optionally start it)
echo "Checking for CARLA server..."
if "$PROJECT_ROOT/scripts/carla_server_manager.sh" status >/dev/null 2>&1; then
    echo "CARLA server is running"
else
    echo "CARLA server is not running"
    echo "   You can start it with: $0 carla start"
    echo "   Or manually start CARLA server in another terminal"
    echo ""
    read -q "REPLY?Start CARLA server automatically in headless mode? (y/N): "
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting CARLA server in background..."
        "$PROJECT_ROOT/scripts/carla_server_manager.sh" start --headless --background
        if [ $? -eq 0 ]; then
            echo "CARLA server started successfully"
        else
            echo "Failed to start CARLA server"
            exit 1
        fi
    else
        echo "Continuing without starting CARLA server"
        echo "   The training application may fail if CARLA is not available"
    fi
fi

# Change to project root
cd "$PROJECT_ROOT"

# Execute the main commands
if [ "$#" -eq 0 ]; then
    # Default: run the main application
    echo "Starting CARLA RL training application..."
    echo "   Command: python app/main.py"
    echo ""
    python app/main.py
elif [ "$1" = "viz" ]; then
    echo "Starting CARLA RL training with visualization..."
    echo "   Command: python app/main.py --enable-pygame-display"
    echo ""
    python app/main.py --enable-pygame-display
else
    echo "Unknown command: $1"
    echo "Use '$0 help' for available commands"
    exit 1
fi 