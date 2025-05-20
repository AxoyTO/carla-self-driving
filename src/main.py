import torch
import numpy as np
from collections import deque
import argparse # Import argparse
import logging  # Import logging
import os # For cleaning up best_model_dir if needed

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
    parser.add_argument("--enable-pygame-display", action="store_true", 
                        help="Enable the Pygame display window for the CARLA environment view (default: disabled).")
    parser.add_argument("--pygame-width", type=int, default=1920,
                        help="Set the width of the Pygame display window (default: 1920).")
    parser.add_argument("--pygame-height", type=int, default=1080,
                        help="Set the height of the Pygame display window (default: 1080).")
    parser.add_argument("--save-dir", type=str, default="./model_checkpoints",
                        help="Directory to save models (default: ./model_checkpoints).")
    parser.add_argument("--load-model-from", type=str, default=None,
                        help="Directory to load a pre-trained model from (e.g., ./model_checkpoints/experiment_run_1).")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save a checkpoint every N episodes (default: 50).")
    parser.add_argument("--eval-interval", type=int, default=25,
                        help="Evaluate model every N training episodes (default: 25).")
    parser.add_argument("--num-eval-episodes", type=int, default=5,
                        help="Number of episodes to run for each evaluation (default: 5).")
    parser.add_argument("--save-sensor-data", action="store_true",
                        help="Enable saving of sensor data to disk (default: disabled).")
    parser.add_argument("--sensor-data-save-path", type=str, default="./sensor_capture",
                        help="Base directory to save captured sensor data (default: ./sensor_capture).")
    parser.add_argument("--sensor-save-interval", type=int, default=100,
                        help="Save sensor data every N steps (default: 100).")
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
    main_logger.info(f"Models will be saved in: {os.path.abspath(args.save_dir)}")
    if args.load_model_from:
        main_logger.info(f"Attempting to load model from: {os.path.abspath(args.load_model_from)}")

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
    num_discrete_actions = 6 # Updated to 6 to include Reverse action

    lr = 1e-4               
    gamma = 0.99            
    tau = 1e-3              
    buffer_capacity = 100000 # Increased from 10000
    batch_size = 64         
    update_every = 4        
    
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

    # Create unique subdirectory for this training run's checkpoints if not loading
    run_specific_save_dir = args.load_model_from # If loading, save to same dir
    if not run_specific_save_dir:
        run_timestamp = tb_logger.get_timestamp() # Assuming Logger has a method to get its timestamp string
        run_specific_save_dir = os.path.join(args.save_dir, f"run_{run_timestamp}")
    
    best_model_dir = os.path.join(run_specific_save_dir, "best_model")
    # if not args.load_model_from and os.path.exists(best_model_dir): # Clean up previous best if fresh run
    #     shutil.rmtree(best_model_dir)
    # os.makedirs(best_model_dir, exist_ok=True) # Ensure best_model_dir exists for current run

    # 1. Environment
    # Pygame is now disabled by default, enabled if flag is present
    enable_pygame = args.enable_pygame_display 
    pygame_win_width = args.pygame_width
    pygame_win_height = args.pygame_height
    try:
        env = CarlaEnv(host=env_host, port=env_port, town=env_town, timestep=env_timestep,
                       image_size=(image_width, image_height), 
                       num_actions=num_discrete_actions, discrete_actions=True,
                       log_level=numeric_log_level,
                       enable_pygame_display=enable_pygame,
                       pygame_window_width=pygame_win_width, 
                       pygame_window_height=pygame_win_height,
                       initial_curriculum_episodes=50, # Example, make this an arg if needed
                       save_sensor_data=args.save_sensor_data,
                       sensor_save_base_path=args.sensor_data_save_path,
                       sensor_save_interval=args.sensor_save_interval)
        
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

    start_episode = 1
    best_eval_score = -float('inf')

    if args.load_model_from:
        if agent.load(args.load_model_from, map_location=device):
            main_logger.info(f"Successfully loaded model and optimizer from {args.load_model_from}")
            # Optionally, try to load training progress (e.g., episode number, epsilon)
            # This would require saving/loading these values as well.
            # For now, epsilon will restart, and episode count too.
            # If a best_score.txt exists in the loaded dir, load it to continue tracking best model.
            try:
                with open(os.path.join(args.load_model_from, "best_score.txt"), "r") as f:
                    best_eval_score = float(f.read())
                    main_logger.info(f"Loaded best_eval_score: {best_eval_score}")
            except FileNotFoundError:
                main_logger.info("No best_score.txt found in loaded model directory. Starting fresh best score.")
            except ValueError:
                main_logger.warning("Could not parse best_score.txt. Starting fresh best score.")
        else:
            main_logger.error(f"Failed to load model from {args.load_model_from}. Starting from scratch.")
            run_specific_save_dir = os.path.join(args.save_dir, f"run_{tb_logger.get_timestamp()}") # New dir for fresh run
    
    # Ensure current run's save directories exist
    os.makedirs(run_specific_save_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True) # This should use run_specific_save_dir
    # Correction: best_model_dir should be based on the potentially new run_specific_save_dir
    current_best_model_dir = os.path.join(run_specific_save_dir, "best_model")
    os.makedirs(current_best_model_dir, exist_ok=True)

    # --- Training Loop --- (with evaluation and saving)
    scores_deque = deque(maxlen=100) # For tracking average scores
    scores = []
    epsilon = epsilon_start
    total_training_steps = 0

    for i_episode in range(start_episode, num_episodes + 1):
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

            # --- Evaluation Phase ---
            if i_episode % args.eval_interval == 0:
                main_logger.info(f"--- Starting Evaluation after Episode {i_episode} ---")
                avg_eval_score, goal_rate = evaluate_agent(env, agent, args.num_eval_episodes, device)
                main_logger.info(f"Evaluation Avg Score: {avg_eval_score:.2f}, Goal Rate: {goal_rate*100:.1f}%")
                tb_logger.log_scalar("evaluation/avg_score", avg_eval_score, i_episode)
                tb_logger.log_scalar("evaluation/goal_reached_rate", goal_rate, i_episode)

                if avg_eval_score > best_eval_score:
                    main_logger.info(f"New best evaluation score: {avg_eval_score:.2f} (previously: {best_eval_score:.2f}). Saving best model.")
                    best_eval_score = avg_eval_score
                    agent.save(current_best_model_dir, model_name="best_dqn_agent")
                    # Save the best score to a file to resume tracking
                    with open(os.path.join(run_specific_save_dir, "best_score.txt"), "w") as f:
                        f.write(str(best_eval_score))
                main_logger.info(f"--- Finished Evaluation ---")

            # --- Periodic Saving --- 
            if i_episode % args.save_interval == 0:
                checkpoint_dir = os.path.join(run_specific_save_dir, f"episode_{i_episode}")
                agent.save(checkpoint_dir)
                main_logger.info(f"Saved checkpoint at episode {i_episode} to {checkpoint_dir}")

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
    # Save final model
    final_model_dir = os.path.join(run_specific_save_dir, "final_model")
    agent.save(final_model_dir)
    main_logger.info(f"Saved final model to {final_model_dir}")
    
    env.close()
    tb_logger.close() # Close the TensorBoard logger

def evaluate_agent(env, agent, num_episodes, device, epsilon_eval=0.01):
    """Evaluates the agent's performance over a number of episodes."""
    total_score = 0.0
    goals_reached = 0
    for i in range(num_episodes):
        state, _ = env.reset()
        episode_score = 0
        done = False
        # Limit max steps during evaluation to avoid infinite loops if agent is bad
        for eval_step in range(env.spec.max_episode_steps if env.spec else 1000): 
            action = agent.select_action(state, epsilon_eval) # Use low epsilon for exploitation
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_score += reward
            if done:
                if info.get("termination_reason") == "goal_reached":
                    goals_reached += 1
                break
        total_score += episode_score
        logger.info(f"  Eval Episode {i+1}/{num_episodes} Score: {episode_score:.2f}, Termination: {info.get('termination_reason', 'unknown')}")
    avg_score = total_score / num_episodes
    goal_reached_rate = goals_reached / num_episodes
    return avg_score, goal_reached_rate

if __name__ == '__main__':
    # Before running, ensure you have created and filled:
    # - RL/models/dqn_model.py (with a DQNModel class that has get_constructor_args())
    # - RL/replay_buffers/uniform_replay_buffer.py (with UniformReplayBuffer class)
    # And that your CARLA server is running.
    train() 