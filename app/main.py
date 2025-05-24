# app/main.py - Main entry point for the CARLA RL training application.
import os
import sys

# Add project root to Python path for local development
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from app/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import app.config as config
from utils.carla_loader import add_carla_to_sys_path

add_carla_to_sys_path(config.CARLA_ROOT) 

from app.runner import start_training_session

if __name__ == '__main__':
    """Script entry point."""
    start_training_session() 