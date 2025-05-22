import torch
import os

# --- General Settings ---
EXPERIMENT_NAME = "dqn_carla_agent"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Argument Parser Default Values ---
# These can be overridden by command-line arguments
LOG_LEVEL = "INFO"
ENABLE_PYGAME_DISPLAY = False
PYGAME_WIDTH = 1920
PYGAME_HEIGHT = 1080
SAVE_DIR = "./model_checkpoints" # Base directory for saving all runs
LOAD_MODEL_FROM = None # e.g., "./model_checkpoints/run_xxxxxxxx_xxxxxx"
SAVE_INTERVAL = 50 # episodes
EVAL_INTERVAL = 25 # training episodes
NUM_EVAL_EPISODES = 5
SAVE_SENSOR_DATA = False
SENSOR_DATA_SAVE_PATH = "./sensor_capture"
SENSOR_SAVE_INTERVAL = 100 # steps
NUM_EPISODES = 1000 # Total training episodes
MAX_STEPS_PER_EPISODE = 1000 # Max steps per training episode
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
EPSILON_EVAL = 0.01 # Epsilon for evaluation phase

# --- Environment Parameters (CarlaEnv) ---
ENV_HOST = 'localhost'
ENV_PORT = 2000
ENV_TOWN = 'Town03'
ENV_TIMESTEP = 0.10  # seconds, for synchronous mode
ENV_TIME_SCALE = 1.0  # Time scale factor (1.0 = normal speed, 2.0 = 2x speed, etc.)
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
# NUM_DISCRETE_ACTIONS is derived by CarlaEnv based on discrete_actions mapping

# --- Agent Parameters (DQNAgent & DQNModel) ---
# observation_space and action_space are determined by the environment
LEARNING_RATE = 1e-4
GAMMA = 0.99  # Discount factor
TAU = 1e-3  # For soft update of target network
REPLAY_BUFFER_CAPACITY = 100000
BATCH_SIZE = 64
UPDATE_EVERY = 4  # Steps

# --- Trainer Parameters (DQNTrainer) ---
# Most trainer parameters are passed via args or derived

# --- Derived/Runtime Configurations (Managed by main.py or trainer.py) ---
# These are not "configs" in the sense of being user-settable defaults here,
# but rather variables that will be determined at runtime.
# RUN_SPECIFIC_SAVE_DIR: Set in main.py, e.g., os.path.join(SAVE_DIR, f"run_{timestamp}")
# CURRENT_BEST_MODEL_DIR: Set in main.py, e.g., os.path.join(RUN_SPECIFIC_SAVE_DIR, "best_model")


# --- Helper function to update config from args ---
def update_config_from_args(config_module, args):
    """
    Updates the configuration module\'s attributes with values from argparse.Namespace.
    Only updates if the attribute exists in the config_module and the args value is not None
    (for arguments that might not be set).
    """
    for key, value in vars(args).items():
        # Convert arg names (e.g., log_level) to config names (e.g., LOG_LEVEL)
        config_key = key.upper()
        if hasattr(config_module, config_key):
            # We only update if the command-line arg was actually provided (or has a default different from None)
            # or if we want to allow overriding with None (which we generally don't for these types of configs)
            if value is not None: # Ensure we don't overwrite a config default with a None arg if parser allows it
                setattr(config_module, config_key, value)
                # print(f"Config updated: {config_key} = {value}") # For debugging
            # else:
                # print(f"Config not updated for {config_key} as arg value is None") # For debugging
        # else:
            # print(f"Config key {config_key} not found in config module") # For debugging

    # Special handling for derived paths if SAVE_DIR is updated by args
    if hasattr(args, 'save_dir') and args.save_dir is not None:
        config_module.SAVE_DIR = args.save_dir
        # Note: RUN_SPECIFIC_SAVE_DIR and CURRENT_BEST_MODEL_DIR will be
        # constructed in main.py using the potentially updated SAVE_DIR.

    # Update device based on availability, can be part of initial config or updated if needed
    config_module.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Reward Calculator Parameters ---
# Standard Rewards/Penalties
REWARD_CALC_PENALTY_COLLISION = -1000.0
REWARD_CALC_REWARD_GOAL_REACHED = 300.0
REWARD_CALC_PENALTY_PER_STEP = -0.1
REWARD_CALC_REWARD_DISTANCE_FACTOR = 1.0
REWARD_CALC_WAYPOINT_REACHED_THRESHOLD = 5.0  # meters
REWARD_CALC_TARGET_SPEED_KMH_DEFAULT = 40.0 # Default, can be overridden by RewardCalculator init arg

# Speed related
REWARD_CALC_TARGET_SPEED_REWARD_FACTOR = 0.5
REWARD_CALC_TARGET_SPEED_STD_DEV_KMH = 10.0
REWARD_CALC_MIN_FORWARD_SPEED_THRESHOLD = 0.1  # m/s
REWARD_CALC_PENALTY_STUCK_OR_REVERSING_BASE = -0.5

# Lane keeping
REWARD_CALC_LANE_CENTERING_REWARD_FACTOR = 0.2
REWARD_CALC_LANE_ORIENTATION_PENALTY_FACTOR = 0.1
REWARD_CALC_PENALTY_OFFROAD = -50.0

# Traffic lights
REWARD_CALC_PENALTY_TRAFFIC_LIGHT_RED_MOVING = -75.0
REWARD_CALC_REWARD_TRAFFIC_LIGHT_GREEN_PROCEED = 5.0
REWARD_CALC_REWARD_TRAFFIC_LIGHT_STOPPED_AT_RED = 15.0
REWARD_CALC_VEHICLE_STOPPED_SPEED_THRESHOLD = 0.1  # m/s

# Proximity
REWARD_CALC_PROXIMITY_THRESHOLD_VEHICLE = 4.0  # meters
REWARD_CALC_PENALTY_PROXIMITY_VEHICLE_FRONT = -15.0

# New: Penalty for crossing a solid lane marking
REWARD_CALC_PENALTY_SOLID_LANE_CROSS = -40.0 # Adjusted penalty for crossing solid lines

# New: Penalty for driving on the sidewalk
REWARD_CALC_PENALTY_SIDEWALK = -800.0

# New: Speed threshold (m/s) considered "stopped" when reaching goal
STOP_AT_GOAL_SPEED_THRESHOLD = 0.2

# Phase 0 Specific Adjustments (can be overridden by curriculum config)
REWARD_CALC_PHASE0_PENALTY_PER_STEP = -0.01
REWARD_CALC_PHASE0_DISTANCE_FACTOR_MULTIPLIER = 2.5
REWARD_CALC_PHASE0_GOAL_REWARD_MULTIPLIER = 1.5
REWARD_CALC_PHASE0_STUCK_PENALTY_BASE = -0.1
REWARD_CALC_PHASE0_STUCK_MULTIPLIER_STUCK = 1.0
REWARD_CALC_PHASE0_STUCK_MULTIPLIER_REVERSING = 2.0
REWARD_CALC_PHASE0_OFFROAD_PENALTY = -10.0
REWARD_CALC_PHASE0_OFFROAD_NO_WAYPOINT_MULTIPLIER = 1.5

# --- Sensor Default Parameters (CarlaEnv) ---
CARLA_DEFAULT_IMAGE_WIDTH = 84
CARLA_DEFAULT_IMAGE_HEIGHT = 84

CARLA_DEFAULT_LIDAR_CHANNELS = 32
CARLA_DEFAULT_LIDAR_RANGE = 50.0  # meters
CARLA_DEFAULT_LIDAR_POINTS_PER_SECOND = 120000
CARLA_DEFAULT_LIDAR_ROTATION_FREQUENCY = 10.0  # Hz
CARLA_DEFAULT_LIDAR_UPPER_FOV = 15.0
CARLA_DEFAULT_LIDAR_LOWER_FOV = -25.0
CARLA_PROCESSED_LIDAR_NUM_POINTS = 720 # Example: 360 azimuth steps * 2 beams (simplified)

CARLA_DEFAULT_RADAR_RANGE = 70.0  # meters
CARLA_DEFAULT_RADAR_HORIZONTAL_FOV = 30.0  # degrees
CARLA_DEFAULT_RADAR_VERTICAL_FOV = 10.0  # degrees
CARLA_DEFAULT_RADAR_POINTS_PER_SECOND = 1500
CARLA_PROCESSED_RADAR_MAX_DETECTIONS = 20

# --- CarlaEnv Default Curriculum Phases ---
CARLA_DEFAULT_CURRICULUM_PHASES = [
    {"name": "Phase0_BasicControl_Straight", "episodes": 30, 
     "reward_config": "phase0", "spawn_config": "fixed_straight",
     "traffic_config": {"num_vehicles": 0, "num_walkers": 0, "type": "none"}},

    {"name": "Phase0_BasicControl_SimpleTurns", "episodes": 50, 
     "reward_config": "phase0", "spawn_config": "fixed_simple_turns", 
     "traffic_config": {"num_vehicles": 0, "num_walkers": 0, "type": "none"},
     "require_stop_at_goal": True},

    {"name": "Phase1_LaneFollowing", "episodes": 150, 
     "reward_config": "standard", "spawn_config": "random_gentle_curves", 
     "traffic_config": {"num_vehicles": 0, "num_walkers": 0, "type": "none"},
     "require_stop_at_goal": True},

    {"name": "Phase2_NavigateStaticObstacles", "episodes": 300, 
     "reward_config": "standard", "spawn_config": "random_urban", 
     "traffic_config": {"num_vehicles": 10, "num_walkers": 0, "type": "static"},
     "require_stop_at_goal": True},

    {"name": "Phase3_TrafficLights", "episodes": 500, 
     "reward_config": "standard", "spawn_config": "random_urban_with_traffic_lights", 
     "traffic_config": {"num_vehicles": 0, "num_walkers": 0, "type": "none"},
     "require_stop_at_goal": True},

    {"name": "Phase4_LightDynamicTraffic", "episodes": 1000, 
     "reward_config": "standard", "spawn_config": "random_urban_full", 
     "traffic_config": {"num_vehicles": 15, "num_walkers": 0, "type": "dynamic"},
     "require_stop_at_goal": True},

    {"name": "Phase5_ComplexDriving", "episodes": 3000, 
     "reward_config": "standard", "spawn_config": "random_urban_full", 
     "traffic_config": {"num_vehicles": 30, "num_walkers": 20, "type": "dynamic"},
     "require_stop_at_goal": True}
]

def get_config_dict(config_module):
    """Returns a dictionary of all uppercase attributes from the config module."""
    return {key: getattr(config_module, key) for key in dir(config_module) if key.isupper()} 