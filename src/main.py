# src/main.py - Main entry point for the CARLA RL training application.
import config
from utils.carla_loader import add_carla_to_sys_path

add_carla_to_sys_path(config.CARLA_ROOT) 

from runner import start_training_session

if __name__ == '__main__':
    """Script entry point."""
    start_training_session() 