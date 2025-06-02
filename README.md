# CARLA Deep Reinforcement Learning Training Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![CARLA 0.9.15](https://img.shields.io/badge/CARLA-0.9.15-orange.svg)](https://carla.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)

A production-ready deep reinforcement learning platform for autonomous driving research in the CARLA simulator. Features advanced DQN algorithms, curriculum learning, and comprehensive deployment automation.

## 📋 Table of Contents

- [✨ Features](#-features)
- [📋 Prerequisites](#-prerequisites)
- [🚀 Installation](#-installation)
- [🏃 Quick Start](#-quick-start)
- [🔧 Usage](#-usage)
- [⚙️ Configuration](#️-configuration)
- [🎯 Training Features](#-training-features)
- [🐳 Deployment](#-deployment)
- [📊 Monitoring & Debugging](#-monitoring--debugging)
- [⚡ Performance Optimization](#-performance-optimization)
- [🔧 Troubleshooting](#-troubleshooting)
- [📚 API Reference](#-api-reference)
- [🤝 Contributing](#-contributing)
- [❓ FAQ](#-faq)
- [📄 License](#-license)

## ✨ Features

### 🧠 Advanced Deep Reinforcement Learning
- **Dueling DQN**: Separate value and advantage streams for superior value estimation
- **Double DQN**: Eliminates overestimation bias through decoupled action selection
- **Prioritized Experience Replay**: Importance sampling for accelerated learning
- **Noisy Networks**: Parameter-based exploration replacing epsilon-greedy strategies
- **N-step Learning**: Multi-step bootstrap targets for faster reward propagation
- **Mixed Precision Training**: FP16 optimization for enhanced memory efficiency

### 🎓 Curriculum Learning System
- **10-Phase Progressive Training**: From basic forward driving to complex urban navigation
- **Automated Phase Progression**: Intelligent evaluation-based curriculum advancement
- **Phase-Specific Configurations**: Tailored rewards, traffic, and evaluation criteria
- **Adaptive Difficulty Scaling**: Dynamic adjustment based on performance metrics

### 🚗 Autonomous Driving Capabilities
- **Multi-Sensor Integration**: Camera, LiDAR, Radar with intelligent sensor management
- **Traffic Simulation**: Dynamic traffic scenarios with pedestrians and vehicles
- **Traffic Light Compliance**: Advanced intersection navigation and rule following
- **Urban Driving**: Complex city scenarios with lane changes and parking maneuvers

### 🔧 Production-Ready Infrastructure
- **Docker Containerization**: Isolated, reproducible development environments
- **Kubernetes Support**: Scalable production deployment orchestration
- **Comprehensive Monitoring**: Real-time TensorBoard integration with detailed metrics
- **Automated CARLA Management**: Intelligent server lifecycle and health monitoring

## 📋 Prerequisites

### Software Dependencies

**Required:**
- [Docker Engine 20.10+](https://docs.docker.com/engine/install/)
- [Docker Compose 2.0+](https://docs.docker.com/compose/install/)
- [Python 3.7+](https://www.python.org/downloads/)

**GPU Support (Highly Recommended):**
- [NVIDIA Docker Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- CUDA-compatible GPU drivers

**For Local Development:**
- [CARLA Simulator 0.9.15](https://carla.org/download/)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or Python virtual environment

## 🚀 Installation

### Option 1: Docker Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/axoyto/carla-rl-training.git
cd carla-rl-training

# Build the Docker image
./scripts/build.sh

# Verify installation
./scripts/deploy.sh health
```

### Option 2: Local Python Installation

```bash
# Clone repository
git clone https://github.com/axoyto/carla-rl-training.git
cd carla-rl-training

# Create and activate virtual environment
conda create -n carla-rl python=3.7
conda activate carla-rl

# Install dependencies
pip install -r requirements.txt

# Verify installation
./scripts/run_local.sh test
```

### CARLA Installation (Local Development Only)

```bash
# Download and extract CARLA 0.9.15
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.15.tar.gz
tar -xzf CARLA_0.9.15.tar.gz

# Install CARLA Python API
pip install /path/to/CARLA_0.9.15/PythonAPI/carla/dist/carla-*.whl
```

## 🏃 Quick Start

### Docker Deployment (5-Minute Setup)

```bash
# 1. Start all services
./scripts/deploy.sh up

# 2. Monitor training progress
./scripts/deploy.sh logs

# 3. Access TensorBoard at http://localhost:6007
```

### Local Development

```bash
# 1. Start training with auto-managed CARLA server
./scripts/run_local.sh

# 2. Start training with visualization
./scripts/run_local.sh viz

# 3. Manual CARLA server control
./scripts/run_local.sh carla start
```

## 🔧 Usage

### Basic Training Commands

```bash
# Standard headless training
python app/main.py

# Training with visualization
python app/main.py --enable-pygame-display

# Load and continue training from checkpoint
python app/main.py --load-model-from ./models/best_model

# Custom training configuration
python app/main.py --num-episodes 2000 --log-level DEBUG
```

### Advanced Configuration

```bash
# High-performance training
python app/main.py --time-scale 2.0 --batch-size 64

# Debugging and data collection
python app/main.py --save-sensor-data --enable-pygame-display --log-level DEBUG

# Custom environment settings
python app/main.py --town Town05 --num-eval-episodes 10
```

### Available Command Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--log-level` | str | INFO | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `--num-episodes` | int | auto | Total training episodes (auto-determined by curriculum) |
| `--save-dir` | str | ./models/model_checkpoints | Model checkpoint save directory |
| `--load-model-from` | str | None | Path to existing model for continuation |
| `--enable-pygame-display` | flag | False | Enable real-time visualization |
| `--save-sensor-data` | flag | False | Save sensor data for analysis |
| `--time-scale` | float | 1.0 | CARLA simulation speed multiplier |
| `--batch-size` | int | 32 | Training batch size |

## ⚙️ Configuration

### Main Configuration File

The primary configuration is managed through `configs/config.yaml`:

```yaml
# Training Parameters
agent:
  learning_rate: 0.00005
  gamma: 0.99
  use_dueling_dqn: true
  use_prioritized_replay: true
  use_noisy_nets: true
  replay_buffer_capacity: 100000
  batch_size: 32

# Environment Settings
environment:
  town: "Town03"
  host: "localhost"
  port: 2000
  image_width: 84
  image_height: 84
  timestep: 0.10

# Curriculum Learning
curriculum:
  evaluation:
    enabled: true
    episodes_per_evaluation: 5
    completion_criteria:
      min_goal_completion_rate: 0.6
      min_collision_free_rate: 0.8
      min_sidewalk_free_rate: 0.9
```

### Environment-Specific Configuration

Configure CARLA environment settings in `configs/environment.yaml`:

```yaml
# Sensor Configuration
sensors:
  camera:
    width: 84
    height: 84
    fov: 90
  lidar:
    channels: 32
    range: 50.0
    points_per_second: 120000
  radar:
    range: 70.0
    horizontal_fov: 30.0

# Traffic Settings
traffic:
  default_vehicles: 20
  default_pedestrians: 10
  dynamic_spawning: true
```

## 🎯 Training Features

### Deep Q-Network Enhancements

#### Dueling DQN Architecture
```yaml
# Automatic implementation - configured via YAML
agent:
  use_dueling_dqn: true
  model_hidden_dims: [512, 256]
```

#### Prioritized Experience Replay
```yaml
# Enhanced learning from important experiences
agent:
  use_prioritized_replay: true
  per_alpha: 0.6
  per_beta_start: 0.4
```

### Curriculum Learning Phases

| Phase | Description | Episodes | Traffic | Key Skills |
|-------|-------------|----------|---------|------------|
| 1 | Forward Only (No Steering) | 50 | None | Basic throttle/brake control |
| 2 | Basic Control (Straight) | 100 | None | Extended straight driving |
| 3 | Simple Turns | 300 | None | Steering, basic navigation |
| 4 | Lane Following | 500 | None | Lane keeping, gentle curves |
| 5 | Light Static Traffic | 750 | 20V/10P Static | Urban navigation, obstacles |
| 6 | Reverse Maneuvers | 250 | None | Parking, tight spaces |
| 7 | Traffic Lights | 600 | 15V/10P Static | Rule compliance, intersections |
| 8 | Dynamic Traffic | 1000 | 25V/15P Dynamic | Complex interactions |
| 9 | Complex Urban | 1500 | 40V/30P Dynamic | Full urban driving |
| 10 | Dense Traffic | 2000 | 60V/40P Dynamic | High-density scenarios |

### Reward System Components

#### Safety Metrics (40% weight)
- **Collision Avoidance**: -1000 penalty for crashes
- **Sidewalk Avoidance**: -800 penalty for off-road driving
- **Proximity Awareness**: -15 penalty for close calls

#### Performance Metrics (30% weight)
- **Goal Achievement**: +300 reward for reaching destination
- **Progress Tracking**: Distance-based incremental rewards
- **Speed Optimization**: Rewards for maintaining target speeds

#### Compliance Metrics (15% weight)
- **Traffic Light Adherence**: +15 reward for proper stops
- **Lane Discipline**: Penalties for improper lane changes
- **Rule Following**: Comprehensive traffic rule compliance

## 🐳 Deployment

### Docker Services Architecture

#### carla-rl-training
```yaml
# Main training service
ports:
  - "6006:6006"  # TensorBoard
volumes:
  - ./data:/app/data
  - ./models:/app/models
  - ./configs:/app/configs
depends_on:
  - carla-server
```

#### carla-server
```yaml
# CARLA simulator service
ports:
  - "2000-2002:2000-2002"  # CARLA API
environment:
  - CARLA_HEADLESS=true
  - DISPLAY=:99
healthcheck:
  test: ["CMD", "python3", "-c", "import carla; carla.Client('localhost', 2000).get_server_version()"]
```

### Kubernetes Production Deployment

```bash
# Deploy to cluster
kubectl apply -f scripts/kubernetes-deployment.yaml

# Scale training pods
kubectl scale deployment carla-rl-training --replicas=3

# Monitor deployment
kubectl get pods -l app=carla-rl-training
kubectl logs -f deployment/carla-rl-training
```

### Production Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CARLA_HOST` | carla-server | CARLA server hostname |
| `CARLA_PORT` | 2000 | CARLA server port |
| `CARLA_HEADLESS` | true | Headless mode flag |
| `START_CARLA_SERVER` | false | Auto-start CARLA in container |
| `PYTHONPATH` | /app | Python module search path |

## 📊 Monitoring & Debugging

### TensorBoard Analytics

Access comprehensive training analytics at **http://localhost:6007**:

#### Training Metrics
- **Episode Rewards**: Raw scores and exponential moving averages
- **Loss Functions**: Q-network loss, policy loss, and gradient norms
- **Learning Progress**: Value function estimation and policy convergence

#### Performance Metrics
- **Success Rates**: Goal completion and collision avoidance rates
- **Efficiency Metrics**: Episode duration and step efficiency
- **Exploration**: Action distribution and exploration effectiveness

#### Curriculum Progress
- **Phase Advancement**: Automatic progression through curriculum stages
- **Evaluation Results**: Phase completion criteria and performance thresholds

### Log Analysis

```bash
# Real-time training monitoring
./scripts/deploy.sh logs -f

# Error detection and debugging
./scripts/deploy.sh logs | grep -E "(ERROR|CRITICAL|Exception)"

# Performance analysis
./scripts/deploy.sh logs | grep -E "(Episode|Phase|Evaluation)"

# CARLA server diagnostics
./scripts/deploy.sh logs-carla
```

### Model Checkpointing Strategy

#### Automatic Saves
- **Best Performance**: Highest-scoring models across all metrics
- **Phase Completion**: Models saved upon curriculum advancement
- **Regular Intervals**: Periodic saves every 50 episodes (configurable)
- **Training State**: Complete optimizer and replay buffer state

#### Manual Model Management
```bash
# Load specific checkpoint
python app/main.py --load-model-from ./models/model_checkpoints/episode_500

# List available checkpoints
ls -la ./models/model_checkpoints/
```

## ⚡ Performance Optimization

### GPU Acceleration Setup

```bash
# 1. Enable GPU support in docker-compose.yml
sed -i 's/# nvidia.com\/gpu: "1"/nvidia.com\/gpu: "1"/g' docker-compose.yml

# 2. Start with GPU acceleration
./scripts/deploy.sh up

# 3. Verify GPU usage
nvidia-smi
```

### Memory Optimization

```yaml
# Reduce memory usage in configs/config.yaml
agent:
  replay_buffer_capacity: 50000  # Reduce from 100000
  batch_size: 16                # Reduce from 32

environment:
  image_width: 64              # Reduce from 84
  image_height: 64             # Reduce from 84
```

### Training Speed Enhancement

```bash
# Faster simulation
python app/main.py --time-scale 2.0

# Optimized CARLA settings
./scripts/carla_server_manager.sh start --headless --quality Low

# Reduced logging overhead
python app/main.py --log-level WARNING
```

### Distributed Training (Advanced)

```bash
# Multi-GPU training (requires code modifications)
python app/main.py --device cuda:0 --parallel-envs 4

# Kubernetes horizontal scaling
kubectl scale deployment carla-rl-training --replicas=5
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### CARLA Connection Failures

**Problem**: `ConnectionRefusedError: [Errno 111] Connection refused`

**Solution**:
```bash
# Check CARLA server status
./scripts/carla_server_manager.sh status

# Restart CARLA server
./scripts/carla_server_manager.sh restart --headless --background

# Verify network connectivity
ping localhost
telnet localhost 2000
```

#### Memory Issues

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Check GPU memory usage
nvidia-smi

# Reduce batch size
python app/main.py --batch-size 16

# Enable gradient accumulation
# Edit configs/config.yaml: gradient_accumulation_steps: 2
```

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'carla'`

**Solution**:
```bash
# Verify Python environment
./scripts/run_local.sh test

# Check CARLA installation
python -c "import carla; print(carla.__version__)"

# Reinstall CARLA Python API
pip install /path/to/CARLA_0.9.15/PythonAPI/carla/dist/carla-*.whl --force-reinstall
```

### Performance Issues

#### Slow Training

**Potential Causes & Solutions**:

1. **CPU Bottleneck**:
   ```bash
   # Monitor CPU usage
   htop
   
   # Increase CARLA server performance
   ./scripts/carla_server_manager.sh start --headless --quality Low
   ```

2. **I/O Bottleneck**:
   ```bash
   # Use SSD storage for data directory
   # Reduce logging frequency in configs
   ```

3. **Network Latency** (Docker):
   ```bash
   # Use host networking for better performance
   docker run --network host carla-rl:latest
   ```

### Debug Mode Activation

```bash
# Enable comprehensive debugging
export CARLA_DEBUG=1
export PYTORCH_DEBUG=1

# Run with maximum verbosity
python app/main.py --log-level DEBUG --enable-pygame-display

# Profile training performance
python app/main.py --enable-profiler
```

## 📚 API Reference

### Core Classes

#### DQNAgent
```python
from app.rl_agents.dqn_agent import DQNAgent

agent = DQNAgent(
    state_size=512,
    action_size=6,
    learning_rate=0.00005,
    use_dueling=True,
    use_noisy_nets=True
)
```

#### CarlaEnvironment
```python
from app.environments.carla_env import CarlaEnvironment

env = CarlaEnvironment(
    host='localhost',
    port=2000,
    town='Town03',
    image_size=(84, 84)
)
```

#### Configuration Loading
```python
from utils.config_loader import load_config

config = load_config('configs/config.yaml')
training_params = config['agent']
```

### Training Script Integration

```python
#!/usr/bin/env python3
"""Custom training script example."""

from app.runner import TrainingRunner

def main():
    """Run custom training session."""
    runner = TrainingRunner()
    runner.setup()
    runner.run()
    runner.cleanup()

if __name__ == '__main__':
    main()
```

## ❓ FAQ

### General Questions

**Q: What CARLA version is supported?**
A: Currently supports CARLA 0.9.15. Compatibility with other versions is not guaranteed.

**Q: Can I run this without a GPU?**
A: Yes, but training will be significantly slower. GPU acceleration is highly recommended.

**Q: How long does training take?**
A: Complete curriculum training takes 24-48 hours on a modern GPU, depending on hardware.

### Technical Questions

**Q: How do I modify the reward function?**
A: Edit `app/environments/reward_calculator.py` and adjust weights in `configs/config.yaml`.

**Q: Can I add custom sensors?**
A: Yes, extend the `SensorManager` class in `app/environments/sensor_manager.py`.

**Q: How do I deploy on multiple GPUs?**
A: Currently requires manual code modifications. Multi-GPU support is planned for future releases.

### Troubleshooting

**Q: Training crashes with "Connection lost" error**
A: CARLA server likely crashed. Check logs with `./scripts/deploy.sh logs-carla` and restart server.

**Q: TensorBoard shows no data**
A: Ensure TensorBoard is pointing to correct log directory: `./data/tensorboard_logs/`

**Q: Models not saving**
A: Check disk space and permissions on `./models/model_checkpoints/` directory.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Fork and clone repository
git clone https://github.com/axoyto/carla-rl-training.git
cd carla-rl-training

# Create development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks (if available)
# pre-commit install
```

### Code Standards

- **Python Style**: Follow PEP 8 with `black` formatting
- **Type Hints**: All functions must have complete type annotations
- **Docstrings**: Google-style docstrings for all public functions
# - **Testing**: Maintain >90% test coverage with `pytest`

### Submission Process

1. **Fork Repository**: Create your feature branch
2. **Local Testing**: Run `./scripts/run_local.sh test`
3. **Docker Testing**: Verify with `./scripts/deploy.sh health`
4. **Code Standards**: Follow Python typing and docstring conventions
5. **Submit PR**: Include test results and performance metrics

### Reporting Issues

Please include the following when reporting issues:
- Environment details (OS, Python version, GPU)
- Reproducible steps
- Error logs and stack traces
- Expected vs actual behavior

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **CARLA Simulator**: [CARLA License](https://carla.org/license/)
- **PyTorch**: [BSD License](https://github.com/pytorch/pytorch/blob/master/LICENSE)
- **NumPy**: [BSD License](https://numpy.org/license.html)

### Citation

If you use this platform in your research, please cite:

```bibtex
@software{carla_rl_platform,
  title={Developing and Evaluating Self-Driving Car Models Using Synthetic Data from the CARLA Simulator},
  author={Tevfik Oguzhan Aksoy},
  year={2025},
  url={https://github.com/axoyto/carla-rl-training}
}
```

---

**🔗 Useful Links**
- [CARLA Documentation](https://carla.readthedocs.io/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Guide](https://kubernetes.io/docs/)

**📧 Support**
- Create an [Issue](https://github.com/axoyto/carla-rl-training/issues)
