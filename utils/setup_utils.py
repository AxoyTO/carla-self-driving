import argparse
import logging
import os
import sys

# Add project root to Python path for local development
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from utils/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import app.config as config

def parse_arguments():
    """Parses command-line arguments using defaults from config.py."""
    parser = argparse.ArgumentParser(
        description="Train a Deep Q-Network (DQN) agent to drive a car in the CARLA simulator. "
                    "The agent learns from camera images to navigate to random waypoints.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- CARLA Installation Path ---
    parser.add_argument("--carla-root", type=str, default=config.CARLA_ROOT,
                        help=f"Path to the CARLA simulator root directory (default: {config.CARLA_ROOT})")

    # --- General Training Arguments ---
    parser.add_argument("--log-level", type=str, default=config.LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help=f"Set the logging level (default: {config.LOG_LEVEL})")
    parser.add_argument("--num-episodes", type=int, default=config.NUM_EPISODES,
                        help=f"Total training episodes (default: {config.NUM_EPISODES}).")
    parser.add_argument("--max-steps-per-episode", type=int, default=config.MAX_STEPS_PER_EPISODE,
                        help=f"Max steps per training episode (default: {config.MAX_STEPS_PER_EPISODE}).")

    # --- Model Saving/Loading ---
    parser.add_argument("--save-dir", type=str, default=config.SAVE_DIR,
                        help=f"Base directory to save models and logs (default: {config.SAVE_DIR}).")
    parser.add_argument("--load-model-from", type=str, default=config.LOAD_MODEL_FROM,
                        help="Directory to load a pre-trained model from (e.g., ./models/model_checkpoints/run_ID). Overrides starting fresh.")
    parser.add_argument("--save-interval", type=int, default=config.SAVE_INTERVAL,
                        help=f"Save a checkpoint every N episodes (default: {config.SAVE_INTERVAL}).")

    # --- Evaluation ---
    parser.add_argument("--eval-interval", type=int, default=config.EVAL_INTERVAL,
                        help=f"Evaluate model every N training episodes (default: {config.EVAL_INTERVAL}).")
    parser.add_argument("--num-eval-episodes", type=int, default=config.NUM_EVAL_EPISODES,
                        help=f"Number of episodes to run for each evaluation (default: {config.NUM_EVAL_EPISODES}).")
    parser.add_argument("--epsilon-eval", type=float, default=config.EPSILON_EVAL,
                        help=f"Epsilon for evaluation phase (default: {config.EPSILON_EVAL}).")

    # --- Pygame Display (CARLA Environment) ---
    parser.add_argument("--enable-pygame-display", action="store_true", default=None, 
                        help="Enable the Pygame display window for CARLA. Overrides config if present.")
    parser.add_argument("--disable-sensor-views", action="store_true", default=False,
                        help="Disable sensor views (RGB, Depth, Semantic, LIDAR, RADAR) in Pygame display for performance. Spectator view remains.")
    parser.add_argument("--pygame-width", type=int, default=config.PYGAME_WIDTH,
                        help=f"Pygame display width (default: {config.PYGAME_WIDTH}).")
    parser.add_argument("--pygame-height", type=int, default=config.PYGAME_HEIGHT,
                        help=f"Pygame display height (default: {config.PYGAME_HEIGHT}).")

    # --- Simulation Speed ---
    parser.add_argument("--time-scale", type=float, default=config.ENV_TIME_SCALE,
                        help=f"Time scale factor for simulation speed (default: {config.ENV_TIME_SCALE}). "
                             f"Values > 1.0 speed up simulation, < 1.0 slow it down.")

    # --- Sensor Data Saving (CARLA Environment) ---
    parser.add_argument("--save-sensor-data", action="store_true", default=None,
                        help="Enable saving of sensor data. Overrides config if present.")
    parser.add_argument("--disable-save-sensor-data", action="store_false", dest="save_sensor_data", default=None,
                        help="Disable saving of sensor data. Overrides config if present.")
    parser.add_argument("--sensor-data-save-path", type=str, default=config.SENSOR_DATA_SAVE_PATH,
                        help=f"Base directory for captured sensor data (default: {config.SENSOR_DATA_SAVE_PATH}).")
    parser.add_argument("--sensor-save-interval", type=int, default=config.SENSOR_SAVE_INTERVAL,
                        help=f"Save sensor data every N steps (default: {config.SENSOR_SAVE_INTERVAL}).")
                        
    # --- DQN Agent Hyperparameters ---
    parser.add_argument("--learning-rate", "--lr", type=float, default=config.LEARNING_RATE,
                        help=f"Learning rate for the optimizer (default: {config.LEARNING_RATE}).")
    parser.add_argument("--gamma", type=float, default=config.GAMMA,
                        help=f"Discount factor for future rewards (default: {config.GAMMA}).")
    parser.add_argument("--tau", type=float, default=config.TAU,
                        help=f"Soft update parameter for target network (default: {config.TAU}).")
    parser.add_argument("--replay-buffer-capacity", type=int, default=config.REPLAY_BUFFER_CAPACITY,
                        help=f"Capacity of the replay buffer (default: {config.REPLAY_BUFFER_CAPACITY}).")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help=f"Batch size for training the model (default: {config.BATCH_SIZE}).")
    parser.add_argument("--update-every", type=int, default=config.UPDATE_EVERY,
                        help=f"Update the model every N steps (default: {config.UPDATE_EVERY}).")
    parser.add_argument("--epsilon-start", type=float, default=config.EPSILON_START,
                        help=f"Initial epsilon for exploration (default: {config.EPSILON_START}).")
    parser.add_argument("--epsilon-end", type=float, default=config.EPSILON_END,
                        help=f"Final epsilon value after decay (default: {config.EPSILON_END}).")
    parser.add_argument("--epsilon-decay", type=float, default=config.EPSILON_DECAY,
                        help=f"Epsilon decay rate per episode (default: {config.EPSILON_DECAY}).")

    # --- Profiling --- 
    parser.add_argument("--enable-profiler", action="store_true", default=False, 
                        help="Enable cProfile for the training session.")

    args = parser.parse_args()

    # Post-processing for boolean flags to correctly override config
    # If action="store_true" with default=None, args.val will be None if flag not used, True if used.
    # If action="store_false" with default=None, args.val will be None if flag not used, False if used.
    if args.enable_pygame_display is None: # --enable-pygame-display was not used
        args.enable_pygame_display = config.ENABLE_PYGAME_DISPLAY # Default to config value (False)
    # If --enable-pygame-display was used, args.enable_pygame_display is True.
    # This correctly reflects user intent to override config if they explicitly enable it.

    if args.save_sensor_data is None: # Neither --save nor --disable was used
        args.save_sensor_data = config.SAVE_SENSOR_DATA
    # Similar logic as above for save_sensor_data.
    
    return args

def setup_logging(log_level_str):
    """Configures basic logging for the application.
    
    Args:
        log_level_str (str): The desired logging level as a string (e.g., "INFO", "DEBUG").
    """
    numeric_log_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_log_level, int):
        # Fallback to INFO if invalid level string is provided, and log a warning.
        logging.warning(f"Invalid log level string: '{log_level_str}'. Defaulting to INFO.")
        numeric_log_level = logging.INFO
        log_level_str = "INFO" # Update for the confirmation message
        
    logging.basicConfig(level=numeric_log_level,
                        format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Get the root logger and confirm the level.
    # Using a logger specific to this setup function to announce.
    setup_logger = logging.getLogger(__name__) # Logger for setup_utils module
    setup_logger.info(f"Logging configured to level: {log_level_str.upper()}")
    return numeric_log_level # Return the numeric level for convenience 