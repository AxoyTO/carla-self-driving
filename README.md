# CARLA RL Training Application

A containerized reinforcement learning training application for autonomous driving in the CARLA simulator with comprehensive CARLA server management.

## Project Structure

```
CARLA/
├── Docker Configuration
│   ├── Dockerfile              # Multi-stage Docker build with headless CARLA support
│   ├── docker-compose.yml      # Development environment with separated services
│   ├── .dockerignore           # Docker build exclusions
│   └── requirements.txt        # Python dependencies
│
├── Application Code
│   ├── app/                    # Core application (renamed from src/)
│   │   ├── main.py            # Entry point
│   │   ├── runner.py          # Training orchestration
│   │   ├── config.py          # Configuration constants
│   │   ├── environments/      # CARLA environment management
│   │   ├── training/          # RL training algorithms
│   │   ├── models/            # Neural network architectures
│   │   └── rl_agents/         # Agent implementations
│   │
│   └── utils/                  # Shared utilities (moved to root)
│       ├── pygame_visualizer.py
│       ├── open3d_visualizer.py
│       ├── data_logger.py
│       └── ...
│
├── DevOps & Deployment
│   ├── scripts/               # Deployment and utility scripts
│   │   ├── build.sh          # Docker image building
│   │   ├── deploy.sh         # Enhanced Docker Compose deployment
│   │   ├── carla_server_manager.sh  # Local CARLA server management
│   │   ├── run_local.sh      # Local development runner
│   │   └── kubernetes-deployment.yaml
│   │
│   └── configs/              # Configuration files
│
└── Data & Models
    ├── data/                 # Runtime data (mounted volumes)
    │   ├── logs/
    │   ├── sensor_capture/
    │   └── tensorboard_logs/
    │
    └── models/               # Model checkpoints (mounted volumes)
        └── model_checkpoints/
```

## Quick Start

### Docker Deployment (Recommended for Production)

#### Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- NVIDIA Docker (optional, for GPU acceleration)

#### 1. Build the Application

```bash
# Build Docker image
./scripts/build.sh

# Or manually:
docker build -t carla-rl:latest .
```

#### 2. Deploy with Docker Compose

```bash
# Start all services (CARLA server + training)
./scripts/deploy.sh up

# Or start services separately
./scripts/deploy.sh start-carla    # Start CARLA server first
./scripts/deploy.sh start-training # Start training after CARLA is ready
```

#### 3. Monitor Training

```bash
# View training logs
./scripts/deploy.sh logs

# View CARLA server logs
./scripts/deploy.sh logs-carla

# View TensorBoard (available at http://localhost:6007)
./scripts/deploy.sh status
```

### Local Development

For local development and testing, you can run the application directly with automatic CARLA server management:

#### Prerequisites

- Python 3.7+
- Conda environment with dependencies
- CARLA simulator installed (optional - can be auto-managed)

#### Setup

```bash
# Activate your conda environment
conda activate carla4

# Install dependencies if not already installed
pip install -r requirements.txt

# Install CARLA Python API (if not already installed)
pip install /path/to/carla/PythonAPI/carla/dist/carla-*.whl
```

#### Running Locally

```bash
# Basic training with automatic CARLA server management
./scripts/run_local.sh

# Training with visualization
./scripts/run_local.sh viz

# Manual CARLA server management
./scripts/run_local.sh carla start    # Start CARLA server in headless mode
./scripts/run_local.sh carla status   # Check CARLA server status
./scripts/run_local.sh carla stop     # Stop CARLA server

# See all available options
./scripts/run_local.sh help
```

#### CARLA Server Management

The application includes a comprehensive CARLA server manager for local development:

```bash
# Direct CARLA server management
./scripts/carla_server_manager.sh start --headless --background
./scripts/carla_server_manager.sh status
./scripts/carla_server_manager.sh logs
./scripts/carla_server_manager.sh stop

# Advanced options
./scripts/carla_server_manager.sh start --headless --port 2001 --quality Medium
./scripts/carla_server_manager.sh help  # See all options
```

## Docker Services

### carla-rl-training
- **Purpose**: Main RL training application
- **Ports**: 6006 (TensorBoard)
- **Volumes**: `./data`, `./models`, `./configs`
- **Dependencies**: Waits for carla-server to be healthy

### carla-server  
- **Purpose**: CARLA simulator server (headless with -RenderOffScreen)
- **Ports**: 2000-2002 (CARLA API)
- **Environment**: Runs with virtual display (Xvfb)
- **GPU**: Recommended for graphics rendering
- **Health Check**: Automatic CARLA client connectivity test

### tensorboard
- **Purpose**: Training monitoring and visualization
- **Port**: 6007 (changed to avoid conflicts)
- **Access**: http://localhost:6007

## Usage Commands

### Docker Commands

```bash
# Basic operations
./scripts/deploy.sh up              # Start all services
./scripts/deploy.sh down            # Stop all services  
./scripts/deploy.sh restart         # Restart all services
./scripts/deploy.sh status          # Check status

# Service-specific operations
./scripts/deploy.sh start-carla     # Start only CARLA server
./scripts/deploy.sh stop-carla      # Stop only CARLA server
./scripts/deploy.sh restart-carla   # Restart CARLA server
./scripts/deploy.sh start-training  # Start only training
./scripts/deploy.sh stop-training   # Stop only training

# Monitoring
./scripts/deploy.sh logs            # Training logs
./scripts/deploy.sh logs-carla      # CARLA server logs
./scripts/deploy.sh logs-tensorboard # TensorBoard logs
./scripts/deploy.sh logs-all        # All service logs

# Development
./scripts/deploy.sh shell           # Access training container
./scripts/deploy.sh shell-carla     # Access CARLA server container
./scripts/deploy.sh health          # Check health of all services
./scripts/deploy.sh cleanup         # Clean up containers and volumes
```

### Local Development Commands

```bash
# Basic training commands
./scripts/run_local.sh                        # Headless training with auto CARLA management
./scripts/run_local.sh viz                    # With visualization
python app/main.py --help                     # Show all options

# CARLA server management
./scripts/run_local.sh carla start            # Start CARLA server in background
./scripts/run_local.sh carla stop             # Stop CARLA server
./scripts/run_local.sh carla status           # Check CARLA status
./scripts/run_local.sh carla restart          # Restart CARLA server
./scripts/run_local.sh carla logs             # Show CARLA logs

# Configuration and monitoring
./scripts/run_local.sh config                 # Show configuration
./scripts/run_local.sh test                   # Test imports
./scripts/run_local.sh tensorboard            # Start TensorBoard

# Direct Python commands (if CARLA server is running)
python app/main.py --enable-pygame-display    # With visualization
python app/main.py --num-episodes 500         # Custom parameters
```

### CARLA Server Manager

The dedicated CARLA server manager provides fine-grained control:

```bash
# Basic operations
./scripts/carla_server_manager.sh start --headless --background
./scripts/carla_server_manager.sh stop
./scripts/carla_server_manager.sh status
./scripts/carla_server_manager.sh restart --headless --background

# Advanced options
./scripts/carla_server_manager.sh start \
    --headless \
    --port 2001 \
    --quality Medium \
    --resolution 1024x768 \
    --background

# Monitoring
./scripts/carla_server_manager.sh logs         # Show server logs
./scripts/carla_server_manager.sh kill         # Force kill all CARLA processes

# Help
./scripts/carla_server_manager.sh help         # Show all options
```

## Kubernetes Deployment

For production deployment on Kubernetes with separate CARLA server pods:

```bash
# Apply the deployment
kubectl apply -f scripts/kubernetes-deployment.yaml

# Check status
kubectl get pods -l app=carla-rl-training
kubectl get pods -l app=carla-server

# View logs
kubectl logs -f deployment/carla-rl-training
kubectl logs -f deployment/carla-server

# Access services
kubectl port-forward service/carla-rl-training-service 6006:6006  # TensorBoard
kubectl port-forward service/carla-server-service 2000:2000      # CARLA server
```

### Kubernetes Features

- **Separate Deployments**: CARLA server and training application in separate pods
- **Health Checks**: Automatic liveness and readiness probes for both services
- **Resource Management**: Configurable CPU, memory, and GPU limits
- **Pod Disruption Budgets**: High availability configuration
- **Persistent Storage**: Separate PVCs for data and models
- **Service Discovery**: CARLA server accessible via Kubernetes service

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CARLA_HOST` | `carla-server` | CARLA server hostname |
| `CARLA_PORT` | `2000` | CARLA server port |
| `CARLA_HEADLESS` | `false` | Enable headless mode with -RenderOffScreen |
| `START_CARLA_SERVER` | `false` | Start CARLA server in container |
| `PYTHONPATH` | `/app` | Python module search path |

### CARLA Server Configuration

The CARLA server supports various configuration options:

**Headless Mode (Recommended)**
- Uses `-RenderOffScreen` for server-side rendering
- Includes virtual display (Xvfb) for containerized environments
- Optimized for performance and resource usage

**Quality Levels**
- `Low`: Minimal graphics quality (default)
- `Medium`: Balanced quality and performance
- `High`: Maximum quality (requires more GPU)

**Container Arguments**
```bash
# Headless mode with custom quality
CARLA_ARGS="-carla-server -RenderOffScreen -quality-level=Medium -world-port=2000"

# Additional performance optimizations
CARLA_ARGS="$CARLA_ARGS -nullrhi"  # For pure headless operation
```

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Training data and logs |
| `./models` | `/app/models` | Model checkpoints |
| `./configs` | `/app/configs` | Configuration files |

### Configuration Files

The application uses YAML configuration files in the `configs/` directory:

- `training.yaml` - Training parameters and curriculum settings
- `environment.yaml` - CARLA environment and sensor configurations  
- `docker.yaml` - Container-specific settings

```python
# Load configurations in Python
from utils.config_loader import load_training_config, get_training_parameter

config = load_training_config()
learning_rate = get_training_parameter("learning_rate")
```

## Development Workflows

### Local Development Workflow

1. **Setup Environment**
   ```bash
   conda activate carla4
   ```

2. **Start CARLA Server (Optional - can be auto-managed)**
   ```bash
   ./scripts/run_local.sh carla start
   ```

3. **Quick Test**
   ```bash
   ./scripts/run_local.sh viz  # Runs with visualization and auto CARLA management
   ```

4. **Full Training**
   ```bash
   ./scripts/run_local.sh  # Headless training with auto CARLA management
   ```

5. **Monitor Progress**
   ```bash
   # In another terminal
   ./scripts/run_local.sh tensorboard
   # Open http://localhost:6006
   ```

### Container Development Workflow

1. **Build and Deploy**
   ```bash
   ./scripts/deploy.sh build
   ```

2. **Monitor Services**
   ```bash
   ./scripts/deploy.sh health
   ./scripts/deploy.sh logs-carla    # CARLA server status
   ./scripts/deploy.sh logs          # Training progress
   ```

3. **Debug**
   ```bash
   ./scripts/deploy.sh shell         # Access training container
   ./scripts/deploy.sh shell-carla   # Access CARLA server container
   ```

### CARLA Server Management Workflow

1. **Check CARLA Status**
   ```bash
   ./scripts/carla_server_manager.sh status
   ```

2. **Start CARLA Server**
   ```bash
   # Development mode
   ./scripts/carla_server_manager.sh start --headless --background
   
   # Production mode with custom settings
   ./scripts/carla_server_manager.sh start \
       --headless --background \
       --quality Medium \
       --port 2000
   ```

3. **Monitor CARLA**
   ```bash
   ./scripts/carla_server_manager.sh logs
   ```

4. **Stop CARLA**
   ```bash
   ./scripts/carla_server_manager.sh stop
   ```

## Monitoring & Debugging

### TensorBoard
- **URL**: http://localhost:6007 (Docker) or http://localhost:6006 (Local)
- **Metrics**: Training loss, episode rewards, success rates
- **Logs**: Real-time training progress

### CARLA Server Monitoring

```bash
# Docker environment
./scripts/deploy.sh logs-carla
./scripts/deploy.sh health

# Local environment
./scripts/carla_server_manager.sh status
./scripts/carla_server_manager.sh logs
./scripts/run_local.sh carla status
```

### Local Development Debugging

```bash
# Test imports
./scripts/run_local.sh test

# Check configuration
./scripts/run_local.sh config

# Test CARLA connection
python -c "import carla; client = carla.Client('localhost', 2000); print('CARLA version:', client.get_server_version())"

# CARLA server diagnostics
./scripts/carla_server_manager.sh status
```

### Application Logs
```bash
# Training application logs
./scripts/deploy.sh logs

# CARLA server logs  
./scripts/deploy.sh logs-carla

# All services
./scripts/deploy.sh logs-all

# Local CARLA server logs
./scripts/carla_server_manager.sh logs
```

### Health Checks
```bash
# Check service status
./scripts/deploy.sh health

# Test CARLA connection in container
./scripts/deploy.sh shell
python -c "import carla; print('CARLA OK')"

# Local CARLA health check
./scripts/carla_server_manager.sh status
```

## Troubleshooting

### Local Development Issues

**Import Errors**
```bash
# The app automatically adds project root to Python path
# If issues persist, check conda environment:
conda activate carla4
./scripts/run_local.sh test
```

**CARLA Server Issues**
```bash
# Check CARLA server status
./scripts/carla_server_manager.sh status

# Start CARLA server manually
./scripts/carla_server_manager.sh start --headless --background

# Check CARLA logs for errors
./scripts/carla_server_manager.sh logs

# Force kill and restart CARLA
./scripts/carla_server_manager.sh kill
./scripts/carla_server_manager.sh start --headless --background
```

**CARLA Connection Failed**
```bash
# Check if CARLA server is running
./scripts/run_local.sh carla status

# Start CARLA server if not running
./scripts/run_local.sh carla start

# Test connection
python -c "import carla; carla.Client('localhost', 2000).get_server_version()"
```

**Missing Dependencies**
```bash
# Activate correct environment and install dependencies
conda activate carla4
pip install -r requirements.txt
```

### Docker Issues

**CARLA Server Not Starting**
```bash
# Check CARLA server container logs
./scripts/deploy.sh logs-carla

# Check container status
./scripts/deploy.sh health

# Restart CARLA server
./scripts/deploy.sh restart-carla

# Access CARLA server container for debugging
./scripts/deploy.sh shell-carla
```

**Training Application Can't Connect to CARLA**
```bash
# Check if CARLA server is healthy
./scripts/deploy.sh health

# Check Docker network connectivity
./scripts/deploy.sh shell
ping carla-server

# Restart services in order
./scripts/deploy.sh stop-training
./scripts/deploy.sh start-carla
./scripts/deploy.sh start-training
```

**Out of Memory**
```bash
# Check resource usage
docker stats

# Increase Docker memory limits in docker-compose.yml
# Or reduce batch size in configs/
```

**GPU Not Available**
```bash
# Check GPU availability
nvidia-smi

# Install NVIDIA Docker runtime
# Uncomment GPU sections in docker-compose.yml
./scripts/deploy.sh restart
```

### CARLA Server Specific Issues

**CARLA Server Fails to Start**
```bash
# Check CARLA installation
./scripts/carla_server_manager.sh status

# Try different CARLA root path
./scripts/carla_server_manager.sh start --carla-root /path/to/carla --headless --background

# Check system resources
free -h  # Memory
df -h    # Disk space
```

**CARLA Server Unresponsive**
```bash
# Force kill and restart
./scripts/carla_server_manager.sh kill
./scripts/carla_server_manager.sh start --headless --background

# Check for zombie processes
ps aux | grep -i carla
```

**Port Conflicts**
```bash
# Use different port
./scripts/carla_server_manager.sh start --port 2001 --headless --background

# Check what's using the port
lsof -i :2000
```

## Performance Optimization

### GPU Acceleration
1. Install NVIDIA Docker runtime
2. Uncomment GPU sections in `docker-compose.yml`
3. Enable GPU for CARLA server: `nvidia.com/gpu: "1"`
4. Restart services: `./scripts/deploy.sh restart`

### CARLA Server Optimization

**Headless Mode Benefits**
- Reduces GPU memory usage
- Eliminates display rendering overhead
- Better performance in containerized environments
- Enables running multiple CARLA instances

**Quality vs Performance**
```bash
# Maximum performance (recommended for training)
./scripts/carla_server_manager.sh start --headless --quality Low

# Balanced performance
./scripts/carla_server_manager.sh start --headless --quality Medium

# Maximum quality (for evaluation/debugging)
./scripts/carla_server_manager.sh start --headless --quality High
```

### Resource Limits
Adjust in `docker-compose.yml`:
```yaml
carla-server:
  deploy:
    resources:
      limits:
        memory: 16Gi
        cpus: '8'
        nvidia.com/gpu: "1"
      reservations:
        memory: 4Gi
        cpus: '2'
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Test locally: `./scripts/run_local.sh test`
4. Test with Docker: `./scripts/deploy.sh build && ./scripts/deploy.sh health`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [CARLA Simulator](https://carla.org/) - Open-source autonomous driving simulator
- [OpenAI Gymnasium](https://gymnasium.farama.org/) - RL environment interface
- [PyTorch](https://pytorch.org/) - Deep learning framework