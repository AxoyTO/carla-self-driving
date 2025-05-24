"""
Configuration loader utility for CARLA RL Training Application
Provides functions to load and merge YAML configuration files.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Utility class for loading and managing configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.loaded_configs = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load a specific configuration file.
        
        Args:
            config_name: Name of the config file (without .yaml extension)
            
        Returns:
            Dictionary containing the configuration
        """
        config_path = self.config_dir / f"{config_name}.yaml"
        
        if not config_path.exists():
            logger.warning(f"Config file {config_path} not found")
            return {}
            
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                self.loaded_configs[config_name] = config
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}")
            return {}
    
    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load all YAML configuration files from the config directory.
        
        Returns:
            Dictionary with config names as keys and config data as values
        """
        configs = {}
        
        if not self.config_dir.exists():
            logger.warning(f"Config directory {self.config_dir} not found")
            return configs
            
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            config_data = self.load_config(config_name)
            if config_data:
                configs[config_name] = config_data
                
        return configs
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configuration files.
        Later configs override earlier ones.
        
        Args:
            config_names: Names of configs to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config_name in config_names:
            config = self.load_config(config_name)
            merged = self._deep_merge(merged, config)
            
        return merged
    
    def get_config_value(self, config_name: str, key_path: str, default: Any = None) -> Any:
        """
        Get a specific value from a config using dot notation.
        
        Args:
            config_name: Name of the configuration
            key_path: Dot-separated path to the value (e.g., 'training.learning_rate')
            default: Default value if key not found
            
        Returns:
            The configuration value or default
        """
        if config_name not in self.loaded_configs:
            config = self.load_config(config_name)
        else:
            config = self.loaded_configs[config_name]
            
        return self._get_nested_value(config, key_path, default)
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _get_nested_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """Get a nested value using dot notation."""
        keys = key_path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
                
        return current

# Global config loader instance
config_loader = ConfigLoader()

def load_training_config() -> Dict[str, Any]:
    """Load the training configuration."""
    return config_loader.load_config("training")

def load_environment_config() -> Dict[str, Any]:
    """Load the environment configuration."""
    return config_loader.load_config("environment")

def load_docker_config() -> Dict[str, Any]:
    """Load the Docker configuration."""
    return config_loader.load_config("docker")

def get_training_parameter(key_path: str, default: Any = None) -> Any:
    """Get a training parameter using dot notation."""
    return config_loader.get_config_value("training", f"training.{key_path}", default)

def get_environment_parameter(key_path: str, default: Any = None) -> Any:
    """Get an environment parameter using dot notation."""
    return config_loader.get_config_value("environment", f"environment.{key_path}", default) 