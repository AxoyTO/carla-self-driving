# src/main.py - Main entry point for the CARLA RL training application.

# To run: python -m src.main --arguments...
# Ensure PYTHONPATH includes the root directory of the project (e.g., /home/toaksoy/CARLA)
# or run from the root directory.

from runner import start_training_session

if __name__ == '__main__':
    """Script entry point."""
    start_training_session() 