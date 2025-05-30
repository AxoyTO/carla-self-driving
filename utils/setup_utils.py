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
    """Parses command-line arguments using defaults from config or new additions."""
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
    # Handle case where NUM_EPISODES might be None (curriculum-controlled)
    default_episodes = config.NUM_EPISODES if config.NUM_EPISODES is not None else 1000
    parser.add_argument("--num-episodes", type=int, default=default_episodes,
                        help=f"Total training episodes (default: {default_episodes}). Ignored when curriculum learning is used.")
    parser.add_argument("--max-steps-per-episode", type=int, default=config.MAX_STEPS_PER_EPISODE,
                        help=f"Max steps per training episode (default: {config.MAX_STEPS_PER_EPISODE}).")
    parser.add_argument("--start-from-phase", type=int, default=None,
                        help="Start training from a specific phase number (1-based indexing). Skips all previous phases.")

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

    # Comprehensive Evaluation Arguments
    parser.add_argument("--comprehensive-eval-interval", type=int, default=config.COMPREHENSIVE_EVAL_INTERVAL,
                        help=f"Run comprehensive evaluation every N episodes (default: {config.COMPREHENSIVE_EVAL_INTERVAL}).")
    parser.add_argument("--comprehensive-eval-min-phase", type=int, default=config.COMPREHENSIVE_EVAL_MIN_PHASE,
                        help=f"Start comprehensive evaluation after phase N (default: {config.COMPREHENSIVE_EVAL_MIN_PHASE}).")
    parser.add_argument("--use-comprehensive-for-best-model", action="store_true", 
                        default=config.USE_COMPREHENSIVE_FOR_BEST_MODEL,
                        help="Use comprehensive evaluation for best model selection (recommended).")
    parser.add_argument("--no-comprehensive-for-best-model", dest="use_comprehensive_for_best_model", 
                        action="store_false",
                        help="Use single-phase evaluation for best model selection (legacy mode).")

    # --- Pygame Display (CARLA Environment) ---
    pg_group = parser.add_argument_group('Pygame Display')
    pg_group.add_argument("--enable-pygame-display", dest='enable_pygame_display', action="store_true", default=config.ENABLE_PYGAME_DISPLAY,
                        help=f"Enable the Pygame display window for CARLA. {'(Enabled by default)' if config.ENABLE_PYGAME_DISPLAY else '(Disabled by default)'}")
    pg_group.add_argument("--no-enable-pygame-display", dest='enable_pygame_display', action="store_false",
                        help="Disable the Pygame display window for CARLA.")
    pg_group.add_argument("--disable-sensor-views", dest='disable_sensor_views', action="store_true", default=config.DISABLE_SENSOR_VIEWS, # Assuming config.DISABLE_SENSOR_VIEWS exists, default False
                        help="Disable sensor views (RGB, Depth, etc.) in Pygame for performance. Spectator view remains. (Off by default)")
    pg_group.add_argument("--no-disable-sensor-views", dest='disable_sensor_views', action="store_false",
                        help="Enable sensor views in Pygame display.")
    pg_group.add_argument("--pygame-width", type=int, default=config.PYGAME_WIDTH,
                        help=f"Pygame display width (default: {config.PYGAME_WIDTH}).")
    pg_group.add_argument("--pygame-height", type=int, default=config.PYGAME_HEIGHT,
                        help=f"Pygame display height (default: {config.PYGAME_HEIGHT}).")

    # --- Simulation Speed ---
    parser.add_argument("--time-scale", type=float, default=config.ENV_TIME_SCALE,
                        help=f"Time scale factor for simulation speed (default: {config.ENV_TIME_SCALE}). "
                             f"Values > 1.0 speed up simulation, < 1.0 slow it down.")

    # --- Sensor Data Saving (CARLA Environment) ---
    sd_group = parser.add_argument_group('Sensor Data Saving')
    sd_group.add_argument("--save-sensor-data", dest='save_sensor_data', action="store_true", default=config.SAVE_SENSOR_DATA,
                        help=f"Enable saving of sensor data. {'(Enabled by default)' if config.SAVE_SENSOR_DATA else '(Disabled by default)'}")
    sd_group.add_argument("--no-save-sensor-data", dest='save_sensor_data', action="store_false",
                        help="Disable saving of sensor data.")
    sd_group.add_argument("--sensor-data-save-path", type=str, default=config.SENSOR_DATA_SAVE_PATH,
                        help=f"Base directory for captured sensor data (default: {config.SENSOR_DATA_SAVE_PATH}).")
    sd_group.add_argument("--sensor-save-interval", type=int, default=config.SENSOR_SAVE_INTERVAL,
                        help=f"Save sensor data every N steps (default: {config.SENSOR_SAVE_INTERVAL}).")
                        
    # --- DQN Agent Hyperparameters ---
    dqn_hp_group = parser.add_argument_group('DQN Hyperparameters')
    dqn_hp_group.add_argument("--learning-rate", "--lr", type=float, default=config.LEARNING_RATE,
                        help=f"Learning rate for the optimizer (default: {config.LEARNING_RATE}).")
    dqn_hp_group.add_argument("--gamma", type=float, default=config.GAMMA,
                        help=f"Discount factor for future rewards (default: {config.GAMMA}).")
    dqn_hp_group.add_argument("--tau", type=float, default=config.TAU,
                        help=f"Soft update parameter for target network (default: {config.TAU}).")
    dqn_hp_group.add_argument("--replay-buffer-capacity", type=int, default=config.REPLAY_BUFFER_CAPACITY,
                        help=f"Capacity of the replay buffer (default: {config.REPLAY_BUFFER_CAPACITY}).")
    dqn_hp_group.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help=f"Batch size for training the model (default: {config.BATCH_SIZE}).")
    dqn_hp_group.add_argument("--update-every", type=int, default=config.UPDATE_EVERY,
                        help=f"Update the model every N steps (default: {config.UPDATE_EVERY}).")
    dqn_hp_group.add_argument("--epsilon-start", type=float, default=config.EPSILON_START,
                        help=f"Initial epsilon for exploration (default: {config.EPSILON_START}).")
    dqn_hp_group.add_argument("--epsilon-end", type=float, default=config.EPSILON_END,
                        help=f"Final epsilon value after decay (default: {config.EPSILON_END}).")
    dqn_hp_group.add_argument("--epsilon-decay", type=float, default=config.EPSILON_DECAY,
                        help=f"Epsilon decay rate per episode (default: {config.EPSILON_DECAY}).")

    # --- Profiling --- 
    prof_group = parser.add_argument_group('Profiling')
    prof_group.add_argument("--enable-profiler", dest='enable_profiler', action="store_true", default=False, # Assuming no config for this, off by default
                        help="Enable cProfile for the training session. (Disabled by default)")
    prof_group.add_argument("--no-enable-profiler", dest='enable_profiler', action="store_false",
                        help="Disable cProfile for the training session.")

    # --- DQN Enhancements --- 
    enh_group = parser.add_argument_group('DQN Enhancements')
    enh_group.add_argument("--disable-all-enhancements", action="store_true", default=False,
                        help="Disable all advanced DQN enhancements below. If set, all enhancements are turned off unless individually re-enabled by other flags.")

    # Double DQN
    enh_group.add_argument("--double-dqn", dest='use_double_dqn', action="store_true", default=config.USE_DOUBLE_DQN,
                        help=f"Enable Double DQN. {'(Enabled by default)' if config.USE_DOUBLE_DQN else '(Disabled by default)'}")
    enh_group.add_argument("--no-double-dqn", dest='use_double_dqn', action="store_false", help="Disable Double DQN.")

    # Dueling DQN
    enh_group.add_argument("--dueling-dqn", dest='use_dueling_dqn', action="store_true", default=config.USE_DUELING_DQN,
                        help=f"Enable Dueling DQN. {'(Enabled by default)' if config.USE_DUELING_DQN else '(Disabled by default)'}")
    enh_group.add_argument("--no-dueling-dqn", dest='use_dueling_dqn', action="store_false", help="Disable Dueling DQN.")

    # Prioritized Experience Replay (PER)
    enh_group.add_argument("--prioritized-replay", "--per", dest='use_prioritized_replay', action="store_true", default=config.USE_PRIORITIZED_REPLAY,
                        help=f"Enable Prioritized Experience Replay (PER). {'(Enabled by default)' if config.USE_PRIORITIZED_REPLAY else '(Disabled by default)'}")
    enh_group.add_argument("--no-prioritized-replay", "--no-per", dest='use_prioritized_replay', action="store_false", help="Disable PER.")
    enh_group.add_argument("--per-alpha", type=float, default=config.PER_ALPHA, help=f"PER alpha (priority exponent) (default: {config.PER_ALPHA})")
    enh_group.add_argument("--per-beta-start", type=float, default=config.PER_BETA_START, help=f"PER beta initial value (importance sampling) (default: {config.PER_BETA_START})")
    enh_group.add_argument("--per-beta-frames", type=int, default=config.PER_BETA_FRAMES, help=f"PER beta annealing frames (default: {config.PER_BETA_FRAMES})")

    # N-Step Returns
    enh_group.add_argument("--n-step-replay", "--n-step", dest='use_n_step_replay', action="store_true", default=config.USE_N_STEP_REPLAY,
                        help=f"Enable N-Step Replay. {'(Enabled by default)' if config.USE_N_STEP_REPLAY else '(Disabled by default)'}")
    enh_group.add_argument("--no-n-step-replay", "--no-n-step", dest='use_n_step_replay', action="store_false", help="Disable N-Step Replay.")
    enh_group.add_argument("--n-step-value", type=int, default=config.N_STEP_VALUE, help=f"N-step value for N-Step Replay (default: {config.N_STEP_VALUE})")

    # Noisy Nets
    enh_group.add_argument("--noisy-nets", dest='use_noisy_nets', action="store_true", default=config.USE_NOISY_NETS,
                        help=f"Enable Noisy Nets for exploration. {'(Enabled by default)' if config.USE_NOISY_NETS else '(Disabled by default)'}")
    enh_group.add_argument("--no-noisy-nets", dest='use_noisy_nets', action="store_false", help="Disable Noisy Nets.")
    enh_group.add_argument("--noisy-sigma0", type=float, default=config.NOISY_SIGMA0, help=f"Noisy Nets sigma0 (initial noise stdev) (default: {config.NOISY_SIGMA0})")

    # Distributional RL (C51)
    enh_group.add_argument("--distributional-rl", "--dist-rl", dest='use_distributional_rl', action="store_true", default=config.USE_DISTRIBUTIONAL_RL,
                        help=f"Enable Distributional RL (C51). {'(Enabled by default)' if config.USE_DISTRIBUTIONAL_RL else '(Disabled by default)'}")
    enh_group.add_argument("--no-distributional-rl", "--no-dist-rl", dest='use_distributional_rl', action="store_false", help="Disable Distributional RL (C51).")
    enh_group.add_argument("--n-atoms", type=int, default=config.DIST_N_ATOMS, help=f"Number of atoms for C51 (default: {config.DIST_N_ATOMS})")
    enh_group.add_argument("--v-min", type=float, default=config.DIST_V_MIN, help=f"Min value for C51 value distribution (default: {config.DIST_V_MIN})")
    enh_group.add_argument("--v-max", type=float, default=config.DIST_V_MAX, help=f"Max value for C51 value distribution (default: {config.DIST_V_MAX})")

    # LSTM
    enh_group.add_argument("--lstm", dest='use_lstm', action="store_true", default=config.USE_LSTM,
                        help=f"Enable LSTM layer in the model. {'(Enabled by default)' if config.USE_LSTM else '(Disabled by default)'}")
    enh_group.add_argument("--no-lstm", dest='use_lstm', action="store_false", help="Disable LSTM layer.")
    enh_group.add_argument("--lstm-hidden-size", type=int, default=config.LSTM_HIDDEN_SIZE, help=f"LSTM hidden size (default: {config.LSTM_HIDDEN_SIZE})")
    enh_group.add_argument("--lstm-num-layers", type=int, default=config.LSTM_NUM_LAYERS, help=f"Number of LSTM layers (default: {config.LSTM_NUM_LAYERS})")

    # --- Model architecture specific ---
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument("--model-hidden-dims", type=int, nargs='+', default=config.MODEL_HIDDEN_DIMS, help=f"Hidden layer dimensions (default: {config.MODEL_HIDDEN_DIMS})")
    model_group.add_argument("--model-use-attention", dest='model_use_attention', action="store_true", default=config.MODEL_USE_ATTENTION,
                        help=f"Use attention mechanism in model. {'(Enabled by default)' if config.MODEL_USE_ATTENTION else '(Disabled by default)'}")
    model_group.add_argument("--no-model-use-attention", dest='model_use_attention', action="store_false", help="Do not use attention mechanism.")
    model_group.add_argument("--model-use-batch-norm", dest='model_use_batch_norm', action="store_true", default=config.MODEL_USE_BATCH_NORM,
                        help=f"Use batch normalization in model. {'(Enabled by default)' if config.MODEL_USE_BATCH_NORM else '(Disabled by default)'}")
    model_group.add_argument("--no-model-use-batch-norm", dest='model_use_batch_norm', action="store_false", help="Do not use batch normalization.")
    model_group.add_argument("--model-dropout-rate", type=float, default=config.MODEL_DROPOUT_RATE, help=f"Dropout rate in model (default: {config.MODEL_DROPOUT_RATE})")

    # --- Advanced Training Options (from DQNTrainer) ---
    adv_train_group = parser.add_argument_group('Advanced Training Options')
    adv_train_group.add_argument("--adaptive-epsilon", dest='adaptive_epsilon', action="store_true", default=config.ADAPTIVE_EPSILON,
                        help=f"Enable adaptive epsilon. {'(Enabled by default)' if config.ADAPTIVE_EPSILON else '(Disabled by default)'}")
    adv_train_group.add_argument("--no-adaptive-epsilon", dest='adaptive_epsilon', action="store_false", help="Disable adaptive epsilon.")
    adv_train_group.add_argument("--early-stopping", dest='early_stopping', action="store_true", default=config.EARLY_STOPPING,
                        help=f"Enable early stopping. {'(Enabled by default)' if config.EARLY_STOPPING else '(Disabled by default)'}")
    adv_train_group.add_argument("--no-early-stopping", dest='early_stopping', action="store_false", help="Disable early stopping.")
    adv_train_group.add_argument("--early-stopping-patience", type=int, default=config.EARLY_STOPPING_PATIENCE, help=f"Patience for early stopping (default: {config.EARLY_STOPPING_PATIENCE})")
    adv_train_group.add_argument("--min-improvement-threshold", type=float, default=config.MIN_IMPROVEMENT_THRESHOLD, help=f"Min improvement for early stopping (default: {config.MIN_IMPROVEMENT_THRESHOLD})")

    # --- Agent Training Enhancements (from DQNAgent) ---
    agent_enh_group = parser.add_argument_group('Agent Training Enhancements')
    agent_enh_group.add_argument("--mixed-precision", dest='use_mixed_precision', action="store_true", default=config.USE_MIXED_PRECISION,
                        help=f"Enable mixed precision training. {'(Enabled by default)' if config.USE_MIXED_PRECISION else '(Disabled by default)'}")
    agent_enh_group.add_argument("--no-mixed-precision", dest='use_mixed_precision', action="store_false", help="Disable mixed precision training.")
    agent_enh_group.add_argument("--gradient-accumulation-steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS, help=f"Gradient accumulation steps (default: {config.GRADIENT_ACCUMULATION_STEPS})")
    agent_enh_group.add_argument("--max-grad-norm", type=float, default=config.MAX_GRAD_NORM, help=f"Max gradient norm for clipping (default: {config.MAX_GRAD_NORM})")

    args = parser.parse_args()
    
    # Apply --disable-all-enhancements if set
    if args.disable_all_enhancements:
        print("INFO: --disable-all-enhancements is active. Disabling all major DQN improvements.")
        args.use_double_dqn = False
        args.use_dueling_dqn = False
        args.use_prioritized_replay = False
        args.use_n_step_replay = False
        args.use_noisy_nets = False
        args.use_distributional_rl = False
        args.use_lstm = False
        # Note: Other flags like adaptive_epsilon, early_stopping, model_use_attention etc. are not part of this bundle.
        # User would need to disable them individually if desired.
        
    # The boolean flags (e.g. args.use_double_dqn, args.enable_pygame_display) 
    # are now directly set by argparse based on their defaults from config and any CLI flags.
    # No further post-processing is needed for them here, beyond the 'disable_all_enhancements' override.

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