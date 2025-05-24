import torch
import os

# --- General Settings ---
EXPERIMENT_NAME = "dqn_carla_agent"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CARLA_ROOT = "/opt/carla-simulator" # Default CARLA installation path

# --- Argument Parser Default Values ---
# These can be overridden by command-line arguments
LOG_LEVEL = "INFO"
ENABLE_PYGAME_DISPLAY = False
PYGAME_WIDTH = 1920
PYGAME_HEIGHT = 1080
DISABLE_SENSOR_VIEWS = False # Default to showing all views
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

# Steering Behavior
REWARD_CALC_PENALTY_EXCESSIVE_STEER_BASE = -0.5 # Base penalty factor for high steering on straights
REWARD_CALC_STEER_THRESHOLD_STRAIGHT = 0.1  # Steer magnitude above which penalty applies on straights
REWARD_CALC_MIN_SPEED_FOR_STEER_PENALTY_KMH = 10.0 # Only apply steer penalty above this speed

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

# --- Action Mapping for Discrete Actions --- 
# Defines the control outputs for each discrete action index.
DISCRETE_ACTION_MAP = {
    0: {"throttle": 0.75, "steer": 0.0,  "brake": 0.0, "reverse": False, "name": "Fwd-Fast"},
    1: {"throttle": 0.5,  "steer": -0.5, "brake": 0.0, "reverse": False, "name": "Fwd-Left"},
    2: {"throttle": 0.5,  "steer": 0.5,  "brake": 0.0, "reverse": False, "name": "Fwd-Right"},
    3: {"throttle": 0.0,  "steer": 0.0,  "brake": 1.0, "reverse": False, "name": "Brake"},
    4: {"throttle": 0.3,  "steer": 0.0,  "brake": 0.0, "reverse": False, "name": "Coast"}, # Gentle forward
    5: {"throttle": 0.3,  "steer": 0.0,  "brake": 0.0, "reverse": True,  "name": "Reverse"}
}

# Action mapping to be used when reverse is disallowed by the current curriculum phase.
# Action 5 (Reverse) is remapped to a moderate brake.
DISCRETE_ACTION_MAP_NO_REVERSE = {
    0: DISCRETE_ACTION_MAP[0],
    1: DISCRETE_ACTION_MAP[1],
    2: DISCRETE_ACTION_MAP[2],
    3: DISCRETE_ACTION_MAP[3],
    4: DISCRETE_ACTION_MAP[4],
    5: {"throttle": 0.0,  "steer": 0.0,  "brake": 0.5, "reverse": False, "name": "Brake"} # Remapped Reverse
}

NUM_DISCRETE_ACTIONS = len(DISCRETE_ACTION_MAP) # Should be 6

# --- CarlaEnv Default Curriculum Phases ---
CARLA_DEFAULT_CURRICULUM_PHASES = [
    {"name": "Phase0_BasicControl_Straight", "episodes": 200, 
     "reward_config": "phase0", "spawn_config": "fixed_straight",
     "traffic_config": {"num_vehicles": 0, "num_walkers": 0, "type": "none"}, # No traffic initially
     "allow_reverse": False, "max_steps": 300, # Shorter episodes for basic task
     "phase0_target_distance_m": 75.0, # Slightly longer straight
     "phase0_spawn_point_idx": 41 # Example: A known good straight spawn point in Town03
     },

    {"name": "Phase0_BasicControl_SimpleTurns", "episodes": 300, 
     "reward_config": "phase0", "spawn_config": "fixed_simple_turns", 
     "traffic_config": {"num_vehicles": 0, "num_walkers": 0, "type": "none"},
     "require_stop_at_goal": True,
     "allow_reverse": False, "max_steps": 500 
    },

    {"name": "Phase1_LaneFollowing_NoTraffic", "episodes": 500, 
     "reward_config": "standard", "spawn_config": "random_gentle_curves", 
     "traffic_config": {"num_vehicles": 0, "num_walkers": 0, "type": "none"},
     "require_stop_at_goal": False, # Focus on continuous driving
     "allow_reverse": False, "max_steps": 1000
    },

    {"name": "Phase2_LaneFollowing_LightStaticTraffic", "episodes": 750, 
     "reward_config": "standard", "spawn_config": "random_urban", # More varied routes
     "traffic_config": {"num_vehicles": 20, "num_walkers": 10, "type": "static"}, # Introduce static obstacles
     "require_stop_at_goal": False,
     "allow_reverse": False, "max_steps": 1200
    },
    
    {"name": "Phase2_5_ReverseManeuvers", "episodes": 250, 
     "reward_config": "standard", "spawn_config": "random_short_segment_for_reverse", # Custom spawn for reversing practice
     "traffic_config": {"num_vehicles": 0, "num_walkers": 0, "type": "none"},
     "require_stop_at_goal": True, # Goal is to park or similar
     "allow_reverse": True, "max_steps": 400 # Shorter, focused episodes
    },

    {"name": "Phase3_TrafficLights_NoDynamicTraffic", "episodes": 600, 
     "reward_config": "standard", "spawn_config": "random_urban_with_traffic_lights", 
     "traffic_config": {"num_vehicles": 15, "num_walkers": 10, "type": "static"}, # Static obstacles for complexity
     "require_stop_at_goal": True, 
     "allow_reverse": False, "max_steps": 1500
    },

    {"name": "Phase4_LightDynamicTraffic_Intersections", "episodes": 1000, 
     "reward_config": "standard", "spawn_config": "random_urban_with_traffic_lights", # Ensures intersections
     "traffic_config": {"num_vehicles": 25, "num_walkers": 15, "type": "dynamic"},
     "require_stop_at_goal": False,
     "allow_reverse": True, "max_steps": 2000
    },

    {"name": "Phase5_ComplexUrbanDriving", "episodes": 1500, 
     "reward_config": "standard", "spawn_config": "random_urban_full", 
     "traffic_config": {"num_vehicles": 40, "num_walkers": 30, "type": "dynamic"},
     "require_stop_at_goal": False,
     "allow_reverse": True, "max_steps": 2500
    },

    {"name": "Phase6_DenseTrafficAndPedestrians", "episodes": 2000, # Longest phase for mastery
     "reward_config": "standard", 
     "spawn_config": "random_urban_full", # Most challenging spawns
     "traffic_config": {
         "num_vehicles": 60, 
         "num_walkers": 40,  
         "type": "dynamic_traffic_light_aware", # Ensure TM uses TL info if possible
     },
     "require_stop_at_goal": False,
     "allow_reverse": True, 
     "max_steps": 3000
    }
]

def get_config_dict(config_module):
    """Returns a dictionary of all uppercase attributes from the config module."""
    return {key: getattr(config_module, key) for key in dir(config_module) if key.isupper()} 