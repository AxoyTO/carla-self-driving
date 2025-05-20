import torch
import numpy as np
from collections import deque
import argparse # Import argparse
import logging  # Import logging

# Environment
from environments.carla_env import CarlaEnv
# from environments.mock_env import MockEnv # For testing without CARLA

# Agent and components
from agents.dqn_agent import DQNAgent
from models.dqn_model import DQNModel # We need to create this file and class
from replay_buffers.uniform_replay_buffer import UniformReplayBuffer # We need to create/fill this

# Utilities (optional, for later)
# from utils.logger import Logger
# from utils.config import load_config
from utils.logger import Logger # Import the new Logger

# For action/observation space definitions (if using Gymnasium)
import gymnasium as gym
from gymnasium import spaces

def train():
    """Main training loop for the RL agent."""

    # --- Argument Parsing for Log Level ---
    parser = argparse.ArgumentParser(
        description="Train a Deep Q-Network (DQN) agent to drive a car in the CARLA simulator. "
                    "The agent learns from camera images to navigate to random waypoints.",
        formatter_class=argparse.RawTextHelpFormatter # Allows for newlines in description and epilog
    )
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    args = parser.parse_args()

    # --- Basic Logging Configuration ---
    # Get the numeric logging level from the string argument
    numeric_log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError(f'Invalid log level: {args.log_level}')
    
    logging.basicConfig(level=numeric_log_level, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Get a logger for the main script (optional, if you want to log from main.py itself)
    main_logger = logging.getLogger("MainTrain")
    main_logger.info(f"Starting training with log level: {args.log_level.upper()}")

    # --- Hyperparameters & Configuration ---
    # TODO: Load from a config file or set them here
    # Environment params
    env_host = 'localhost'
    env_port = 2000
    env_town = 'Town03'
    env_timestep = 0.05 # seconds, for synchronous mode

    # Agent params
    # obs_shape will depend on actual sensor output from CarlaEnv
    # For now, let's assume a placeholder like (channels, height, width) e.g. (3, 84, 84)
    # This needs to match what _get_observation() in CarlaEnv returns and DQNModel expects
    # dummy_obs_shape_for_model = (3, 84, 84) # Example: (C, H, W) for PyTorch Conv2D
    # n_actions will depend on how we define actions in CarlaEnv
    # dummy_n_actions_for_model = 3 # Example: 0: straight, 1: left, 2: right
    image_width = 84
    image_height = 84
    num_discrete_actions = 5 # Matches default in CarlaEnv

    lr = 1e-4               # Learning rate
    gamma = 0.99            # Discount factor
    tau = 1e-3              # Soft update factor for target network
    buffer_capacity = 10000 # Replay buffer size
    batch_size = 64         # Minibatch size for learning
    update_every = 4        # How often to update the Q-network
    
    # Training loop params
    num_episodes = 1000
    max_steps_per_episode = 1000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_logger.info(f"Using device: {device}")

    # --- Initialization ---
    # 0. Logger (TensorBoard Logger)
    tb_logger = Logger(experiment_name="dqn_carla_agent") # Renamed to tb_logger to avoid confusion with python logging

    # 1. Environment
    # For now, CarlaEnv's action_space and observation_space are not fully defined.
    # We will pass dummy/placeholder spaces for now or define them properly soon.
    try:
        env = CarlaEnv(host=env_host, port=env_port, town=env_town, timestep=env_timestep,
                       image_size=(image_width, image_height), # Pass configured image size
                       num_actions=num_discrete_actions, discrete_actions=True,
                       log_level=numeric_log_level) # Pass numeric log level to CarlaEnv
        
        # The action and observation spaces are now defined within CarlaEnv constructor
        # So, we can directly use them.
        # if env._action_space is None:
        #     env._action_space = spaces.Discrete(dummy_n_actions_for_model) # Match dummy_n_actions_for_model
        # if env._observation_space is None:
        #     # This needs to match what _get_observation returns: (H, W, C) np.uint8
        #     # The DQNModel will expect (C, H, W) typically.
        #     env._observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

    except Exception as e:
        main_logger.error(f"Failed to initialize CarlaEnv: {e}", exc_info=True) # Changed from print
        main_logger.error("Please ensure a CARLA server is running.") # Changed from print
        return

    # 2. Replay Buffer
    # Assuming UniformReplayBuffer takes capacity
    replay_buffer = UniformReplayBuffer(capacity=buffer_capacity)

    # 3. DQN Model (Q-Network)
    # The model needs to know the shape of observations and number of actions.
    # We need to create dqn_model.py with a DQNModel class.
    # For PyTorch CNNs, input shape is typically (Batch, Channels, Height, Width)
    # Our current dummy obs from CarlaEnv is (H, W, C) = (84,84,3). Permute if needed.
    # Let's assume DQNModel will handle permutation or expect (C,H,W)
    # q_network_model = DQNModel(state_shape=dummy_obs_shape_for_model, n_actions=dummy_n_actions_for_model)
    
    # Get state shape and number of actions from the environment spaces
    # CarlaEnv observation_space.shape is (C, H, W) <-- This comment is outdated
    # state_shape = env.observation_space.shape # <-- This would fail for Dict space
    # n_actions = env.action_space.n
    # main_logger.info(f"Environment state shape: {state_shape}, Number of actions: {n_actions}")

    # q_network_model = DQNModel(state_shape=state_shape, n_actions=n_actions) # <-- Old initialization

    # New initialization for DQNModel that accepts the observation_space dictionary
    observation_space = env.observation_space # This is a gym.spaces.Dict
    n_actions = env.action_space.n
    main_logger.info(f"Environment observation space type: {type(observation_space)}")
    if isinstance(observation_space, gym.spaces.Dict):
        main_logger.info(f"Observation space keys: {list(observation_space.spaces.keys())}")
        # Optionally log shapes of individual spaces within the Dict
        for key, space in observation_space.spaces.items():
            main_logger.info(f"  {key}: shape={space.shape}, dtype={space.dtype}")
    main_logger.info(f"Number of actions: {n_actions}")

    q_network_model = DQNModel(observation_space=observation_space, n_actions=n_actions)
    main_logger.info("DQNModel initialized successfully with dictionary observation space.")
    
    # 4. Agent
    agent = DQNAgent(observation_space=env.observation_space, 
                     action_space=env.action_space, 
                     model=q_network_model, 
                     replay_buffer=replay_buffer,
                     lr=lr, gamma=gamma, tau=tau, 
                     batch_size=batch_size, update_every=update_every, 
                     device=device)

    # --- Training Loop ---
    scores_deque = deque(maxlen=100) # For tracking average scores
    scores = []
    epsilon = epsilon_start
    total_training_steps = 0

    for i_episode in range(1, num_episodes + 1):
        try:
            state, _ = env.reset() # CarlaEnv returns obs, info
            episode_score = 0
            episode_losses = [] # To store losses for this episode
            termination_reason = "Unknown"

            for t_step in range(max_steps_per_episode):
                total_training_steps += 1
                action = agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                replay_buffer.add(state, action, reward, next_state, done)
                
                current_loss = agent.step_experience_and_learn()
                if current_loss is not None:
                    episode_losses.append(current_loss)

                state = next_state
                episode_score += reward
                if done:
                    if terminated:
                        termination_reason = info.get("termination_reason", "terminated_by_env")
                        # info from env.step might contain more details
                        # e.g. CarlaEnv could set info={'termination_reason': 'collision'} in _check_done
                    elif truncated:
                        termination_reason = f"max_steps_reached ({max_steps_per_episode})"
                    break
            
            # If loop finished without done=True, it means max_steps_per_episode was reached
            if not (terminated or truncated):
                termination_reason = f"max_steps_reached ({max_steps_per_episode})"

            scores_deque.append(episode_score)
            scores.append(episode_score)
            avg_score = np.mean(scores_deque)
            avg_episode_loss = np.mean(episode_losses) if episode_losses else None
            epsilon = max(epsilon_end, epsilon_decay * epsilon) # Decay epsilon

            # Console logging for episode summary
            main_logger.info(f"Episode {i_episode}/{num_episodes}\tSteps: {t_step+1}")
            main_logger.info(f"  Score: {episode_score:.2f}\tAvg Score (100 ep): {avg_score:.2f}")
            main_logger.info(f"  Avg Loss: {avg_episode_loss if avg_episode_loss is not None else 'N/A' :.4f}\tEpsilon: {epsilon:.3f}")
            main_logger.info(f"  Termination: {termination_reason}")
            
            # TensorBoard Logging
            tb_logger.log_episode_stats(
                episode=i_episode, 
                score=episode_score, 
                avg_score=avg_score, 
                epsilon=epsilon, 
                total_steps=t_step+1,
                avg_loss=avg_episode_loss
            )

            # if i_episode % 100 == 0:
            #     print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque):.2f}')
                # Add model saving here if needed
                # agent.save("./checkpoints/dqn_agent")

        except RuntimeError as e:
            main_logger.error(f"Runtime error during episode {i_episode}: {e}", exc_info=True) # Changed from print
            # Potentially try to reconnect or cleanup and restart environment
            # For now, we'll break the training loop
            break
        except KeyboardInterrupt:
            main_logger.info("Training interrupted by user.") # Changed from print
            break
        finally:
            # Any per-episode cleanup if necessary (CarlaEnv.reset already handles actor destruction)
            pass 
            
    main_logger.info("Training finished.") # Changed from print
    # --- Cleanup ---
    env.close()
    tb_logger.close() # Close the TensorBoard logger

if __name__ == '__main__':
    # Before running, ensure you have created and filled:
    # - RL/models/dqn_model.py (with a DQNModel class that has get_constructor_args())
    # - RL/replay_buffers/uniform_replay_buffer.py (with UniformReplayBuffer class)
    # And that your CARLA server is running.
    train() 