"""
Centralized configuration management system for CARLA DQN training.
Provides type-safe configuration with validation and easy experiment setup.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
import yaml
import json
import os
import logging
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class AgentType(Enum):
    """Supported agent types."""
    STANDARD_DQN = "standard_dqn"
    ENHANCED_DQN = "enhanced_dqn"
    LIGHTWEIGHT_ENHANCED = "lightweight_enhanced"


class OptimizationMode(Enum):
    """Training optimization modes."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    RESEARCH = "research"


@dataclass
class AgentConfig:
    """DQN Agent configuration."""
    # Basic hyperparameters
    learning_rate: float = 0.0001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Training parameters
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 1000
    tau: float = 0.005
    
    # Enhanced features
    use_dueling: bool = False
    use_double_dqn: bool = True
    gradient_clip: float = 1.0
    use_prioritized_replay: bool = False
    
    # Agent type
    agent_type: AgentType = AgentType.LIGHTWEIGHT_ENHANCED
    
    def validate(self):
        """Validate agent configuration parameters."""
        assert 0 < self.learning_rate < 1, f"Learning rate must be in (0, 1), got {self.learning_rate}"
        assert 0 <= self.gamma <= 1, f"Gamma must be in [0, 1], got {self.gamma}"
        assert self.epsilon_start >= self.epsilon_end, f"Epsilon start ({self.epsilon_start}) must be >= epsilon end ({self.epsilon_end})"
        assert self.batch_size > 0, f"Batch size must be positive, got {self.batch_size}"
        assert self.memory_size >= self.batch_size, f"Memory size ({self.memory_size}) must be >= batch size ({self.batch_size})"
        assert self.target_update_freq > 0, f"Target update frequency must be positive, got {self.target_update_freq}"
        assert 0 < self.tau <= 1, f"Tau must be in (0, 1], got {self.tau}"
        assert self.gradient_clip >= 0, f"Gradient clip must be non-negative, got {self.gradient_clip}"


@dataclass
class EnvironmentConfig:
    """CARLA environment configuration."""
    # CARLA connection
    carla_host: str = "localhost"
    carla_port: int = 2000
    town: str = "Town03"
    
    # Simulation parameters
    timestep: float = 0.05
    time_scale: float = 1.0
    max_episode_steps: int = 1000
    
    # Sensor configuration
    enable_sensors: List[str] = field(default_factory=lambda: ["rgb_camera", "gnss", "imu"])
    image_size: Tuple[int, int] = (84, 84)
    
    # Visualization
    enable_pygame_display: bool = False
    pygame_width: int = 1280
    pygame_height: int = 720
    disable_sensor_views: bool = False
    
    # Data saving
    save_sensor_data: bool = False
    sensor_data_save_path: str = "./data/sensor_data"
    sensor_save_interval: int = 100
    
    def validate(self):
        """Validate environment configuration."""
        assert 1000 <= self.carla_port <= 65535, f"CARLA port must be in [1000, 65535], got {self.carla_port}"
        assert 0.001 <= self.timestep <= 1.0, f"Timestep must be in [0.001, 1.0], got {self.timestep}"
        assert 0.1 <= self.time_scale <= 10.0, f"Time scale must be in [0.1, 10.0], got {self.time_scale}"
        assert self.max_episode_steps > 0, f"Max episode steps must be positive, got {self.max_episode_steps}"
        assert len(self.image_size) == 2, f"Image size must be a tuple of 2 integers, got {self.image_size}"
        assert all(s > 0 for s in self.image_size), f"Image size dimensions must be positive, got {self.image_size}"


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    # Episode parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 1000
    
    # Evaluation
    eval_interval: int = 25
    num_eval_episodes: int = 5
    epsilon_eval: float = 0.0
    
    # Saving and logging
    save_interval: int = 100
    save_dir: str = "./models"
    log_level: str = "INFO"
    load_model_from: Optional[str] = None
    
    # Performance optimization
    optimization_mode: OptimizationMode = OptimizationMode.BALANCED
    enable_performance_monitoring: bool = True
    enable_observation_preprocessing: bool = True
    
    def validate(self):
        """Validate training configuration."""
        assert self.num_episodes > 0, f"Number of episodes must be positive, got {self.num_episodes}"
        assert self.max_steps_per_episode > 0, f"Max steps per episode must be positive, got {self.max_steps_per_episode}"
        assert self.eval_interval > 0, f"Evaluation interval must be positive, got {self.eval_interval}"
        assert self.num_eval_episodes > 0, f"Number of evaluation episodes must be positive, got {self.num_eval_episodes}"
        assert 0 <= self.epsilon_eval <= 1, f"Evaluation epsilon must be in [0, 1], got {self.epsilon_eval}"
        assert self.save_interval > 0, f"Save interval must be positive, got {self.save_interval}"
        assert self.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], f"Invalid log level: {self.log_level}"


@dataclass
class ObservationPreprocessingConfig:
    """Observation preprocessing configuration."""
    enable_compression: bool = True
    primary_camera_size: Tuple[int, int] = (64, 64)
    secondary_camera_size: Tuple[int, int] = (32, 32)
    use_grayscale_secondary: bool = True
    normalize_sensors: bool = True
    enable_caching: bool = True
    cache_size: int = 100
    
    def validate(self):
        """Validate preprocessing configuration."""
        assert len(self.primary_camera_size) == 2, "Primary camera size must be a tuple of 2 integers"
        assert len(self.secondary_camera_size) == 2, "Secondary camera size must be a tuple of 2 integers"
        assert all(s > 0 for s in self.primary_camera_size), "Primary camera size dimensions must be positive"
        assert all(s > 0 for s in self.secondary_camera_size), "Secondary camera size dimensions must be positive"
        assert self.cache_size > 0, f"Cache size must be positive, got {self.cache_size}"


@dataclass
class MonitoringConfig:
    """Performance monitoring configuration."""
    enable_plots: bool = True
    save_interval: int = 100
    plot_update_interval: int = 5
    metrics_window_size: int = 100
    log_dir: str = "./logs/monitoring"
    
    # Real-time dashboard
    enable_dashboard: bool = False
    dashboard_port: int = 8080
    
    def validate(self):
        """Validate monitoring configuration."""
        assert self.save_interval > 0, f"Save interval must be positive, got {self.save_interval}"
        assert self.plot_update_interval > 0, f"Plot update interval must be positive, got {self.plot_update_interval}"
        assert self.metrics_window_size > 0, f"Metrics window size must be positive, got {self.metrics_window_size}"
        assert 1000 <= self.dashboard_port <= 65535, f"Dashboard port must be in [1000, 65535], got {self.dashboard_port}"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Core configurations
    agent: AgentConfig = field(default_factory=AgentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: ObservationPreprocessingConfig = field(default_factory=ObservationPreprocessingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Experiment metadata
    experiment_name: str = "carla_dqn_experiment"
    description: str = "CARLA DQN training experiment"
    tags: List[str] = field(default_factory=list)
    
    # Device configuration
    device: DeviceType = DeviceType.AUTO
    
    def validate(self):
        """Validate all configuration components."""
        self.agent.validate()
        self.environment.validate()
        self.training.validate()
        self.preprocessing.validate()
        self.monitoring.validate()
        
        # Cross-component validations
        if self.training.optimization_mode == OptimizationMode.CONSERVATIVE:
            self._apply_conservative_settings()
        elif self.training.optimization_mode == OptimizationMode.AGGRESSIVE:
            self._apply_aggressive_settings()
    
    def _apply_conservative_settings(self):
        """Apply conservative settings for stability."""
        self.agent.batch_size = min(self.agent.batch_size, 16)
        self.agent.memory_size = min(self.agent.memory_size, 5000)
        self.agent.use_prioritized_replay = False
        self.agent.use_dueling = False
        self.preprocessing.enable_compression = True
        self.preprocessing.primary_camera_size = (32, 32)
        
    def _apply_aggressive_settings(self):
        """Apply aggressive settings for performance."""
        self.agent.use_prioritized_replay = True
        self.agent.use_dueling = True
        self.agent.use_double_dqn = True
        self.agent.learning_rate = min(self.agent.learning_rate * 1.5, 0.001)
        self.preprocessing.enable_compression = False  # Use full resolution


class ConfigManager:
    """Configuration manager for loading, saving, and validating configurations."""
    
    @staticmethod
    def load_from_yaml(config_path: str) -> ExperimentConfig:
        """Load configuration from YAML file with validation."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert to configuration objects
        config = ConfigManager._dict_to_config(config_dict)
        config.validate()
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    @staticmethod
    def load_from_json(config_path: str) -> ExperimentConfig:
        """Load configuration from JSON file with validation."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = ConfigManager._dict_to_config(config_dict)
        config.validate()
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    @staticmethod
    def save_to_yaml(config: ExperimentConfig, output_path: str):
        """Save configuration to YAML file."""
        config_dict = ConfigManager._config_to_dict(config)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved configuration to {output_path}")
    
    @staticmethod
    def save_to_json(config: ExperimentConfig, output_path: str):
        """Save configuration to JSON file."""
        config_dict = ConfigManager._config_to_dict(config)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved configuration to {output_path}")
    
    @staticmethod
    def create_preset_config(preset_name: str) -> ExperimentConfig:
        """Create a preset configuration."""
        if preset_name == "lightweight_enhanced":
            return ConfigManager._create_lightweight_enhanced_config()
        elif preset_name == "conservative":
            return ConfigManager._create_conservative_config()
        elif preset_name == "research":
            return ConfigManager._create_research_config()
        else:
            raise ValueError(f"Unknown preset: {preset_name}")
    
    @staticmethod
    def _create_lightweight_enhanced_config() -> ExperimentConfig:
        """Create lightweight enhanced configuration."""
        config = ExperimentConfig()
        config.experiment_name = "lightweight_enhanced_experiment"
        config.agent.agent_type = AgentType.LIGHTWEIGHT_ENHANCED
        config.agent.use_double_dqn = True
        config.agent.use_dueling = False
        config.agent.gradient_clip = 1.0
        config.training.optimization_mode = OptimizationMode.BALANCED
        config.preprocessing.enable_compression = True
        return config
    
    @staticmethod
    def _create_conservative_config() -> ExperimentConfig:
        """Create conservative configuration for stability."""
        config = ExperimentConfig()
        config.experiment_name = "conservative_experiment"
        config.agent.agent_type = AgentType.STANDARD_DQN
        config.agent.batch_size = 16
        config.agent.memory_size = 5000
        config.training.optimization_mode = OptimizationMode.CONSERVATIVE
        config.preprocessing.primary_camera_size = (32, 32)
        config.preprocessing.secondary_camera_size = (24, 24)
        return config
    
    @staticmethod
    def _create_research_config() -> ExperimentConfig:
        """Create research configuration with all features enabled."""
        config = ExperimentConfig()
        config.experiment_name = "research_experiment"
        config.agent.agent_type = AgentType.ENHANCED_DQN
        config.agent.use_prioritized_replay = True
        config.agent.use_dueling = True
        config.agent.use_double_dqn = True
        config.training.optimization_mode = OptimizationMode.RESEARCH
        config.preprocessing.enable_compression = False
        config.monitoring.enable_dashboard = True
        return config
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to configuration object."""
        # Convert enums
        for section in ['agent', 'training']:
            if section in config_dict:
                section_dict = config_dict[section]
                if 'agent_type' in section_dict:
                    section_dict['agent_type'] = AgentType(section_dict['agent_type'])
                if 'optimization_mode' in section_dict:
                    section_dict['optimization_mode'] = OptimizationMode(section_dict['optimization_mode'])
        
        if 'device' in config_dict:
            config_dict['device'] = DeviceType(config_dict['device'])
        
        # Create configuration objects
        agent_config = AgentConfig(**config_dict.get('agent', {}))
        env_config = EnvironmentConfig(**config_dict.get('environment', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        preprocessing_config = ObservationPreprocessingConfig(**config_dict.get('preprocessing', {}))
        monitoring_config = MonitoringConfig(**config_dict.get('monitoring', {}))
        
        # Create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['agent', 'environment', 'training', 'preprocessing', 'monitoring']}
        
        return ExperimentConfig(
            agent=agent_config,
            environment=env_config,
            training=training_config,
            preprocessing=preprocessing_config,
            monitoring=monitoring_config,
            **main_config
        )
    
    @staticmethod
    def _config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary."""
        from dataclasses import asdict
        
        config_dict = asdict(config)
        
        # Convert enums to strings
        if 'agent' in config_dict and 'agent_type' in config_dict['agent']:
            config_dict['agent']['agent_type'] = config_dict['agent']['agent_type'].value
        
        if 'training' in config_dict and 'optimization_mode' in config_dict['training']:
            config_dict['training']['optimization_mode'] = config_dict['training']['optimization_mode'].value
            
        if 'device' in config_dict:
            config_dict['device'] = config_dict['device'].value
        
        return config_dict 