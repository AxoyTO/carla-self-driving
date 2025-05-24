import logging
import gymnasium as gym
import os
import sys

# Add project root to Python path for local development
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from utils/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Environment
from app.environments.carla_env import CarlaEnv

# Agent and components
from app.rl_agents.dqn_agent import DQNAgent
from app.models.dqn_model import DQNModel
from app.replay_buffers.uniform_replay_buffer import UniformReplayBuffer

# Typing
from typing import Tuple, Dict, Any # For type hints

# Forward declaration for type hinting config module if needed, or just pass as 'Any'
# ConfigType = Any 

def initialize_training_components(args: Any, config: Any, numeric_log_level: int, logger: logging.Logger) -> Tuple[Any, Any, Any, Any]:
    """
    Initializes the core components for training: Environment, Replay Buffer, Q-Network Model, and Agent.

    Args:
        args: Parsed command-line arguments (argparse.Namespace).
        config: The configuration module.
        numeric_log_level: The numeric logging level (e.g., logging.INFO).
        logger: The main logger instance for logging messages.

    Returns:
        A tuple containing the initialized (env, replay_buffer, q_network_model, agent).
        Returns (None, None, None, None) if environment initialization fails.
    """
    logger.info("Initializing training components...")

    # 1. Environment
    try:
        logger.info("Initializing CARLA environment...")
        env = CarlaEnv(
            host=config.ENV_HOST, 
            port=config.ENV_PORT, 
            town=config.ENV_TOWN,
            timestep=config.ENV_TIMESTEP,
            time_scale=args.time_scale,
            image_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT),
            discrete_actions=True,  # Assuming discrete actions for DQN
            log_level=numeric_log_level,
            enable_pygame_display=args.enable_pygame_display,
            pygame_window_width=args.pygame_width,
            pygame_window_height=args.pygame_height,
            save_sensor_data=args.save_sensor_data,
            sensor_save_base_path=args.sensor_data_save_path,
            sensor_save_interval=args.sensor_save_interval
        )
        logger.info("CARLA environment initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize CarlaEnv: {e}", exc_info=True)
        logger.error("Please ensure a CARLA server is running.")
        return None, None, None, None

    # 2. Replay Buffer
    logger.info("Initializing replay buffer...")
    replay_buffer = UniformReplayBuffer(capacity=args.replay_buffer_capacity)
    logger.info(f"Replay buffer initialized with capacity: {args.replay_buffer_capacity}")

    # 3. DQN Model (Q-Network)
    logger.info("Initializing DQN model...")
    observation_space = env.observation_space
    n_actions = env.action_space.n
    logger.info(f"Environment observation space type: {type(observation_space)}")
    if isinstance(observation_space, gym.spaces.Dict):
        logger.info(f"Observation space keys: {list(observation_space.spaces.keys())}")
        for key, space_item in observation_space.spaces.items():
            logger.info(f"  {key}: shape={space_item.shape}, dtype={space_item.dtype}")
    logger.info(f"Number of actions: {n_actions}")

    q_network_model = DQNModel(observation_space=observation_space, n_actions=n_actions)
    logger.info("DQN model initialized successfully.")

    # 4. Agent
    logger.info("Initializing DQNAgent...")
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        model=q_network_model,
        replay_buffer=replay_buffer,
        lr=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        update_every=args.update_every,
        device=config.DEVICE  # DEVICE comes from the global config
    )
    logger.info("DQNAgent initialized successfully.")

    logger.info("All training components initialized.")
    return env, replay_buffer, q_network_model, agent 