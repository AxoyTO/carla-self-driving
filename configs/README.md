# Configuration Files

This directory contains YAML configuration files for the CARLA RL Training Application.

## 📁 Configuration Files

### `training.yaml`
Contains training-specific parameters:
- Learning rate, batch size, gamma
- Episode counts and training schedules  
- Curriculum learning phases
- Visualization settings

### `environment.yaml`
Contains CARLA environment settings:
- CARLA server connection details
- World and weather settings
- Vehicle and sensor configurations
- Traffic and pedestrian settings
- Action space definitions

### `docker.yaml`
Contains Docker-specific settings:
- Container resource limits
- Port mappings
- Volume configurations
- Environment variables
- Development vs production settings

## 🔧 Using Configuration Files

### Loading Configurations in Python

```python
from utils.config_loader import (
    load_training_config,
    load_environment_config,
    get_training_parameter,
    get_environment_parameter
)

# Load entire config files
training_config = load_training_config()
env_config = load_environment_config()

# Get specific parameters using dot notation
learning_rate = get_training_parameter("learning_rate")
carla_host = get_environment_parameter("carla_server.host")
lidar_range = get_environment_parameter("sensors.lidar.range")
```

### Merging Multiple Configs

```python
from utils.config_loader import config_loader

# Merge training and environment configs
merged_config = config_loader.merge_configs("training", "environment")

# Use merged config
total_episodes = merged_config["training"]["total_episodes"]
carla_port = merged_config["environment"]["carla_server"]["port"]
```

### Environment-Specific Configs

You can create environment-specific configuration files:

```bash
# Development environment
configs/training.dev.yaml
configs/environment.dev.yaml

# Production environment  
configs/training.prod.yaml
configs/environment.prod.yaml
```

Then load them based on environment:

```python
import os
from utils.config_loader import config_loader

env = os.getenv("ENV", "dev")
training_config = config_loader.load_config(f"training.{env}")
```

## 🐳 Docker Integration

Configuration files are automatically mounted into containers at `/app/configs/`.

### Using Configs with Docker Compose

The configs are mounted as volumes in `docker-compose.yml`:

```yaml
volumes:
  - ./configs:/app/configs
```

### Using Configs with Kubernetes

Configs can be loaded as ConfigMaps:

```bash
kubectl create configmap carla-configs --from-file=configs/
```

## 📝 Configuration Examples

### Override Training Parameters

Create a custom training config for experimentation:

```yaml
# configs/training.experiment.yaml
training:
  learning_rate: 0.0005  # Higher learning rate
  batch_size: 64         # Larger batch size
  total_episodes: 2000   # More episodes
  
  curriculum:
    enable: false        # Disable curriculum learning
```

### Custom Environment Settings

Create a config for a specific town:

```yaml
# configs/environment.town01.yaml
environment:
  world:
    town: "Town01"
    weather: "CloudyNoon"
    
  traffic:
    vehicles:
      number: 50  # More traffic
    pedestrians:
      number: 100
```

## 🔄 Configuration Validation

The config loader includes basic validation:

- Warns about missing files
- Provides default values for missing keys
- Logs successful config loads

### Error Handling

```python
from utils.config_loader import config_loader

# Safe config loading with defaults
learning_rate = config_loader.get_config_value(
    "training", 
    "training.learning_rate", 
    default=0.001
)
```

## 🛠️ Best Practices

1. **Use descriptive names**: `training.experiment1.yaml` not `config1.yaml`
2. **Keep defaults**: Always provide sensible defaults in code
3. **Document changes**: Comment significant parameter changes
4. **Validate ranges**: Ensure parameters are within valid ranges
5. **Environment separation**: Use separate configs for dev/staging/prod

## 📊 Configuration Hierarchy

Configurations are merged in this order (later overrides earlier):

1. Default values in code
2. Base configuration files (`training.yaml`, `environment.yaml`)
3. Environment-specific configs (`training.prod.yaml`)
4. Command-line arguments or environment variables

## 🔍 Debugging Configurations

To debug configuration loading:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Enable config loader logging
from utils.config_loader import config_loader
config_loader.load_all_configs()
```

This will show which configs are loaded and any warnings about missing files. 