import torch
import os
import sys
from typing import Dict, Any, Optional

# Add project root to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from app/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config_loader import ConfigLoader

# Initialize config loader with correct path to configs directory
# The configs directory is at the project root level, not relative to app/
config_dir = os.path.join(project_root, "configs")
config_loader = ConfigLoader(config_dir)

# Load main configuration
_main_config = config_loader.load_config("config")

# --- Dynamic/Computed Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration Access Functions ---
def get_config_value(key_path: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation."""
    return config_loader._get_nested_value(_main_config, key_path, default)

def get_general_config(key: str, default: Any = None) -> Any:
    """Get a general configuration value."""
    return get_config_value(f"general.{key}", default)

def get_default_config(key: str, default: Any = None) -> Any:
    """Get a default configuration value."""
    return get_config_value(f"defaults.{key}", default)

def get_environment_config(key: str, default: Any = None) -> Any:
    """Get an environment configuration value."""
    return get_config_value(f"environment.{key}", default)

def get_agent_config(key: str, default: Any = None) -> Any:
    """Get an agent configuration value."""
    return get_config_value(f"agent.{key}", default)

def get_reward_config(key: str, default: Any = None) -> Any:
    """Get a reward calculator configuration value."""
    return get_config_value(f"reward_calculator.{key}", default)

def get_action_config(key: str, default: Any = None) -> Any:
    """Get an action configuration value."""
    return get_config_value(f"actions.{key}", default)

def get_curriculum_config(key: str, default: Any = None) -> Any:
    """Get a curriculum configuration value."""
    return get_config_value(f"curriculum.{key}", default)

# --- Exposed Configuration Values (for backward compatibility) ---
# General Settings
EXPERIMENT_NAME = get_general_config("experiment_name", "dqn_carla_agent")
CARLA_ROOT = get_general_config("carla_root", "/opt/carla-simulator")

# Argument Parser Default Values
LOG_LEVEL = get_default_config("log_level", "INFO")
ENABLE_PYGAME_DISPLAY = get_default_config("enable_pygame_display", False)
PYGAME_WIDTH = get_default_config("pygame_width", 1920)
PYGAME_HEIGHT = get_default_config("pygame_height", 1080)
DISABLE_SENSOR_VIEWS = get_default_config("disable_sensor_views", False)

# Fix the SAVE_DIR to use absolute path based on project root
_save_dir_relative = get_default_config("save_dir", "./models/model_checkpoints")
if _save_dir_relative.startswith('./'):
    # Convert relative path to absolute path based on project root
    SAVE_DIR = os.path.join(project_root, _save_dir_relative[2:])  # Remove './' prefix
else:
    SAVE_DIR = _save_dir_relative

LOAD_MODEL_FROM = get_default_config("load_model_from", None)
SAVE_INTERVAL = get_default_config("save_interval", 50)
EVAL_INTERVAL = get_default_config("eval_interval", 25)
NUM_EVAL_EPISODES = get_default_config("num_eval_episodes", 5)

# Fix sensor data save path similarly
_sensor_data_path_relative = get_default_config("sensor_data_save_path", "./data/sensor_capture")
if _sensor_data_path_relative.startswith('./'):
    SENSOR_DATA_SAVE_PATH = os.path.join(project_root, _sensor_data_path_relative[2:])
else:
    SENSOR_DATA_SAVE_PATH = _sensor_data_path_relative

SAVE_SENSOR_DATA = get_default_config("save_sensor_data", False)
SENSOR_SAVE_INTERVAL = get_default_config("sensor_save_interval", 100)
NUM_EPISODES = get_default_config("num_episodes", 1000)
MAX_STEPS_PER_EPISODE = get_default_config("max_steps_per_episode", 1000)
EPSILON_START = get_default_config("epsilon_start", 1.0)
EPSILON_END = get_default_config("epsilon_end", 0.01)
EPSILON_DECAY = get_default_config("epsilon_decay", 0.995)
EPSILON_EVAL = get_default_config("epsilon_eval", 0.01)

# Environment Parameters
ENV_HOST = get_environment_config("host", "localhost")
ENV_PORT = get_environment_config("port", 2000)
ENV_TOWN = get_environment_config("town", "Town03")
ENV_TIMESTEP = get_environment_config("timestep", 0.10)
ENV_TIME_SCALE = get_environment_config("time_scale", 1.0)
IMAGE_WIDTH = get_environment_config("image_width", 84)
IMAGE_HEIGHT = get_environment_config("image_height", 84)

# Agent Parameters
LEARNING_RATE = get_agent_config("learning_rate", 1e-4)
GAMMA = get_agent_config("gamma", 0.99)
TAU = get_agent_config("tau", 1e-3)
REPLAY_BUFFER_CAPACITY = get_agent_config("replay_buffer_capacity", 100000)
BATCH_SIZE = get_agent_config("batch_size", 64)
UPDATE_EVERY = get_agent_config("update_every", 4)

# Sensor Default Parameters
CARLA_DEFAULT_IMAGE_WIDTH = get_environment_config("camera.default_width", 84)
CARLA_DEFAULT_IMAGE_HEIGHT = get_environment_config("camera.default_height", 84)

CARLA_DEFAULT_LIDAR_CHANNELS = get_environment_config("lidar.default_channels", 32)
CARLA_DEFAULT_LIDAR_RANGE = get_environment_config("lidar.default_range", 50.0)
CARLA_DEFAULT_LIDAR_POINTS_PER_SECOND = get_environment_config("lidar.default_points_per_second", 120000)
CARLA_DEFAULT_LIDAR_ROTATION_FREQUENCY = get_environment_config("lidar.default_rotation_frequency", 10.0)
CARLA_DEFAULT_LIDAR_UPPER_FOV = get_environment_config("lidar.default_upper_fov", 15.0)
CARLA_DEFAULT_LIDAR_LOWER_FOV = get_environment_config("lidar.default_lower_fov", -25.0)
CARLA_PROCESSED_LIDAR_NUM_POINTS = get_environment_config("lidar.processed_num_points", 720)

CARLA_DEFAULT_RADAR_RANGE = get_environment_config("radar.default_range", 70.0)
CARLA_DEFAULT_RADAR_HORIZONTAL_FOV = get_environment_config("radar.default_horizontal_fov", 30.0)
CARLA_DEFAULT_RADAR_VERTICAL_FOV = get_environment_config("radar.default_vertical_fov", 10.0)
CARLA_DEFAULT_RADAR_POINTS_PER_SECOND = get_environment_config("radar.default_points_per_second", 1500)
CARLA_PROCESSED_RADAR_MAX_DETECTIONS = get_environment_config("radar.processed_max_detections", 20)

# Reward Calculator Parameters
REWARD_CALC_PENALTY_COLLISION = get_reward_config("penalty_collision", -1000.0)
REWARD_CALC_REWARD_GOAL_REACHED = get_reward_config("reward_goal_reached", 300.0)
REWARD_CALC_PENALTY_PER_STEP = get_reward_config("penalty_per_step", -0.1)
REWARD_CALC_REWARD_DISTANCE_FACTOR = get_reward_config("reward_distance_factor", 1.0)
REWARD_CALC_WAYPOINT_REACHED_THRESHOLD = get_reward_config("waypoint_reached_threshold", 5.0)
REWARD_CALC_TARGET_SPEED_KMH_DEFAULT = get_reward_config("target_speed_kmh_default", 40.0)

REWARD_CALC_TARGET_SPEED_REWARD_FACTOR = get_reward_config("target_speed_reward_factor", 0.5)
REWARD_CALC_TARGET_SPEED_STD_DEV_KMH = get_reward_config("target_speed_std_dev_kmh", 10.0)
REWARD_CALC_MIN_FORWARD_SPEED_THRESHOLD = get_reward_config("min_forward_speed_threshold", 0.1)
REWARD_CALC_PENALTY_STUCK_OR_REVERSING_BASE = get_reward_config("penalty_stuck_or_reversing_base", -0.5)

REWARD_CALC_LANE_CENTERING_REWARD_FACTOR = get_reward_config("lane_centering_reward_factor", 0.2)
REWARD_CALC_LANE_ORIENTATION_PENALTY_FACTOR = get_reward_config("lane_orientation_penalty_factor", 0.1)
REWARD_CALC_PENALTY_OFFROAD = get_reward_config("penalty_offroad", -50.0)

REWARD_CALC_PENALTY_TRAFFIC_LIGHT_RED_MOVING = get_reward_config("penalty_traffic_light_red_moving", -75.0)
REWARD_CALC_REWARD_TRAFFIC_LIGHT_GREEN_PROCEED = get_reward_config("reward_traffic_light_green_proceed", 5.0)
REWARD_CALC_REWARD_TRAFFIC_LIGHT_STOPPED_AT_RED = get_reward_config("reward_traffic_light_stopped_at_red", 15.0)
REWARD_CALC_VEHICLE_STOPPED_SPEED_THRESHOLD = get_reward_config("vehicle_stopped_speed_threshold", 0.1)

REWARD_CALC_PROXIMITY_THRESHOLD_VEHICLE = get_reward_config("proximity_threshold_vehicle", 4.0)
REWARD_CALC_PENALTY_PROXIMITY_VEHICLE_FRONT = get_reward_config("penalty_proximity_vehicle_front", -15.0)

REWARD_CALC_PENALTY_EXCESSIVE_STEER_BASE = get_reward_config("penalty_excessive_steer_base", -0.5)
REWARD_CALC_STEER_THRESHOLD_STRAIGHT = get_reward_config("steer_threshold_straight", 0.1)
REWARD_CALC_MIN_SPEED_FOR_STEER_PENALTY_KMH = get_reward_config("min_speed_for_steer_penalty_kmh", 10.0)

REWARD_CALC_PENALTY_SOLID_LANE_CROSS = get_reward_config("penalty_solid_lane_cross", -40.0)
REWARD_CALC_PENALTY_SIDEWALK = get_reward_config("penalty_sidewalk", -800.0)
STOP_AT_GOAL_SPEED_THRESHOLD = get_reward_config("stop_at_goal_speed_threshold", 0.2)

# Phase 0 Specific Adjustments
REWARD_CALC_PHASE0_PENALTY_PER_STEP = get_reward_config("phase0.penalty_per_step", -0.01)
REWARD_CALC_PHASE0_DISTANCE_FACTOR_MULTIPLIER = get_reward_config("phase0.distance_factor_multiplier", 2.5)
REWARD_CALC_PHASE0_GOAL_REWARD_MULTIPLIER = get_reward_config("phase0.goal_reward_multiplier", 1.5)
REWARD_CALC_PHASE0_STUCK_PENALTY_BASE = get_reward_config("phase0.stuck_penalty_base", -0.1)
REWARD_CALC_PHASE0_STUCK_MULTIPLIER_STUCK = get_reward_config("phase0.stuck_multiplier_stuck", 1.0)
REWARD_CALC_PHASE0_STUCK_MULTIPLIER_REVERSING = get_reward_config("phase0.stuck_multiplier_reversing", 2.0)
REWARD_CALC_PHASE0_OFFROAD_PENALTY = get_reward_config("phase0.offroad_penalty", -10.0)
REWARD_CALC_PHASE0_OFFROAD_NO_WAYPOINT_MULTIPLIER = get_reward_config("phase0.offroad_no_waypoint_multiplier", 1.5)

# Action Mapping
DISCRETE_ACTION_MAP = get_action_config("discrete_action_map", {})
DISCRETE_ACTION_MAP_NO_REVERSE = get_action_config("discrete_action_map_no_reverse", {})
NUM_DISCRETE_ACTIONS = len(DISCRETE_ACTION_MAP)

# Curriculum Phases
CARLA_DEFAULT_CURRICULUM_PHASES = get_curriculum_config("default_phases", [])

# --- Helper function to update config from args ---
def update_config_from_args(config_module, args):
    """
    Updates the configuration module's attributes with values from argparse.Namespace.
    Only updates if the attribute exists in the config_module and the args value is not None.
    """
    for key, value in vars(args).items():
        # Convert arg names (e.g., log_level) to config names (e.g., LOG_LEVEL)
        config_key = key.upper()
        if hasattr(config_module, config_key):
            if value is not None:
                setattr(config_module, config_key, value)

    # Special handling for derived paths if SAVE_DIR is updated by args
    if hasattr(args, 'save_dir') and args.save_dir is not None:
        save_dir_value = args.save_dir
        # Convert relative path to absolute path based on project root if needed
        if save_dir_value.startswith('./'):
            save_dir_value = os.path.join(project_root, save_dir_value[2:])
        config_module.SAVE_DIR = save_dir_value

    # Update device based on availability
    config_module.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_config_dict(config_module):
    """Returns a dictionary of all uppercase attributes from the config module."""
    return {key: getattr(config_module, key) for key in dir(config_module) if key.isupper()} 