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
from app.models.dueling_dqn_model import DuelingDQNModel
from app.replay_buffers.uniform_replay_buffer import UniformReplayBuffer
from app.replay_buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from app.replay_buffers.n_step_replay_buffer import NStepReplayBuffer

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
            sensor_save_interval=args.sensor_save_interval,
            start_from_phase=getattr(args, 'start_from_phase', None)
        )
        logger.info("CARLA environment initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize CarlaEnv: {e}", exc_info=True)
        logger.error("Please ensure a CARLA server is running.")
        return None, None, None, None

    # 2. Replay Buffer
    logger.info("Initializing replay buffer...")
    use_per = getattr(args, 'use_prioritized_replay', False)
    use_n_step = getattr(args, 'use_n_step_replay', False)
    n_step_value = getattr(args, 'n_step_value', 3)
    agent_gamma = getattr(args, 'gamma', 0.99) # Get gamma from agent args for NStepBuffer

    if use_n_step:
        replay_buffer = NStepReplayBuffer(
            capacity=args.replay_buffer_capacity,
            n_step=n_step_value,
            gamma=agent_gamma
        )
        logger.info(f"N-Step Replay Buffer initialized with N={n_step_value}, capacity: {args.replay_buffer_capacity}")
    elif use_per:
        replay_buffer = PrioritizedReplayBuffer(
            capacity=args.replay_buffer_capacity,
            alpha=getattr(args, 'per_alpha', 0.6),
            beta_start=getattr(args, 'per_beta_start', 0.4),
            beta_frames=getattr(args, 'per_beta_frames', 100000)
        )
        logger.info(f"Prioritized Replay Buffer initialized with capacity: {args.replay_buffer_capacity}")
    else:
        replay_buffer = UniformReplayBuffer(capacity=args.replay_buffer_capacity)
        logger.info(f"Uniform Replay Buffer initialized with capacity: {args.replay_buffer_capacity}")

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

    use_dueling_dqn = getattr(args, 'dueling_dqn', False)
    use_noisy_nets_default = getattr(args, 'use_noisy_nets', True)
    use_distributional_rl_default = getattr(args, 'use_distributional_rl', False)

    if use_dueling_dqn:
        q_network_model = DuelingDQNModel(
            observation_space=observation_space, 
            n_actions=n_actions,
            hidden_dims=getattr(args, 'model_hidden_dims', (512, 256, 128)),
            use_attention=getattr(args, 'model_use_attention', True),
            use_batch_norm=getattr(args, 'model_use_batch_norm', True),
            dropout_rate=getattr(args, 'model_dropout_rate', 0.1),
            use_noisy_nets=use_noisy_nets_default,
            noisy_sigma0=getattr(args, 'noisy_sigma0', 0.5),
            use_distributional_rl=use_distributional_rl_default,
            n_atoms=getattr(args, 'n_atoms', 51),
            v_min=getattr(args, 'v_min', -10.0),
            v_max=getattr(args, 'v_max', 10.0),
            use_lstm=getattr(args, 'use_lstm', False),
            lstm_hidden_size=getattr(args, 'lstm_hidden_size', 256),
            lstm_num_layers=getattr(args, 'lstm_num_layers', 1)
        )
        logger.info(f"Dueling DQN model initialized. NoisyNets: {use_noisy_nets_default}, Distributional: {use_distributional_rl_default}")
    else:
        q_network_model = DQNModel(
            observation_space=observation_space, 
            n_actions=n_actions,
            hidden_dims=getattr(args, 'model_hidden_dims_std', (512, 256)),
            use_batch_norm=getattr(args, 'model_use_batch_norm', True),
            dropout_rate=getattr(args, 'model_dropout_rate', 0.1),
            use_noisy_nets=use_noisy_nets_default,
            noisy_sigma0=getattr(args, 'noisy_sigma0', 0.5),
            use_distributional_rl=use_distributional_rl_default,
            n_atoms=getattr(args, 'n_atoms', 51),
            v_min=getattr(args, 'v_min', -10.0),
            v_max=getattr(args, 'v_max', 10.0),
            use_lstm=getattr(args, 'use_lstm', False),
            lstm_hidden_size=getattr(args, 'lstm_hidden_size', 256),
            lstm_num_layers=getattr(args, 'lstm_num_layers', 1)
        )
        logger.info(f"Standard DQN model initialized. NoisyNets: {use_noisy_nets_default}, Distributional: {use_distributional_rl_default}")

    # 4. Agent
    logger.info("Initializing DQNAgent...")
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        model=q_network_model,
        replay_buffer=replay_buffer,
        lr=args.learning_rate,
        gamma=agent_gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        update_every=args.update_every,
        device=config.DEVICE,  # DEVICE comes from the global config
        double_dqn=getattr(args, 'double_dqn', True),
        dueling_dqn=use_dueling_dqn, # Pass the flag to the agent
        use_mixed_precision=getattr(args, 'use_mixed_precision', True),
        gradient_accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
        max_grad_norm=getattr(args, 'max_grad_norm', 1.0),
        use_n_step=use_n_step, # Pass flag to agent
        use_noisy_nets=use_noisy_nets_default, # Pass the potentially defaulted noisy_nets flag
        use_distributional_rl=use_distributional_rl_default, # Pass the potentially defaulted distributional_rl flag
        n_atoms=getattr(args, 'n_atoms', 51),
        v_min=getattr(args, 'v_min', -10.0),
        v_max=getattr(args, 'v_max', 10.0),
        use_lstm=getattr(args, 'use_lstm', False) # Pass to agent
    )
    logger.info("DQNAgent initialized successfully.")

    logger.info("All training components initialized.")
    return env, replay_buffer, q_network_model, agent 