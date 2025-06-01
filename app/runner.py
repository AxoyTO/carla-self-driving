import logging
import os
import sys
import cProfile
import pstats
import glob
from typing import Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import app.config as config

from utils.logger import Logger as TensorBoardLogger
from utils.setup_utils import parse_arguments, setup_logging
from utils.component_initializers import initialize_training_components
from app.training.dqn_trainer import DQNTrainer

TENSORBOARD_BASE_LOG_DIR = os.path.join(project_root, "data", "tensorboard_logs")

def _find_best_model_to_load(base_model_save_dir: str, logger_instance: logging.Logger) -> Optional[str]:
    """
    Checks for best_model/best_score.txt directly in base_model_save_dir
    and returns the path to the best_model directory if found and valid.
    """
    logger_instance.info(f"Attempting to find best model in: {base_model_save_dir}")
    best_score_so_far = -float('inf')
    path_to_best_model_dir = None

    potential_best_model_dir = os.path.join(base_model_save_dir, "best_model")
    score_file_path = os.path.join(potential_best_model_dir, "best_score.txt")
    
    if os.path.exists(score_file_path) and os.path.isdir(potential_best_model_dir):
        try:
            with open(score_file_path, "r") as f:
                score = float(f.read().strip())
            
            score_type = "driving score" if score <= 100 else "legacy raw score"
            logger_instance.debug(f"Found {score_type} {score:.2f} in {score_file_path}")
            
            if score > best_score_so_far: # Technically, only one 'best_model' dir, so this check is simple
                best_score_so_far = score
                path_to_best_model_dir = potential_best_model_dir
        except (ValueError, IOError) as e:
            logger_instance.warning(f"Could not read or parse score file at {score_file_path}: {e}")
    
    if path_to_best_model_dir:
        score_type = "driving score" if best_score_so_far <= 100 else "legacy raw score"
        logger_instance.info(f"Determined best model to load: {path_to_best_model_dir} with {score_type} {best_score_so_far:.2f}")
        if best_score_so_far > 100:
            logger_instance.info("Note: Legacy model detected. Training will adapt to new driving score system.")
    else:
        logger_instance.info(f"No best_model directory with a valid best_score.txt found directly in {base_model_save_dir}.")
        
    return path_to_best_model_dir

class TrainingRunner:
    """Orchestrates the entire training session lifecycle."""
    def __init__(self):
        self.args = None
        # self.config = config # config is a module, can be imported directly where needed
        self.numeric_log_level = None
        self.main_logger = None # Logger for the runner itself
        
        self.tb_logger = None
        self.env = None
        self.replay_buffer = None
        self.q_network_model = None
        self.agent = None
        self.trainer = None

    def setup(self):
        """Parses arguments, sets up logging, and initializes all components."""
        self.args = parse_arguments()
        config.update_config_from_args(config, self.args) # Update global config state
        self.numeric_log_level = setup_logging(self.args.log_level)
        
        self.main_logger = logging.getLogger(__name__) 

        logging.getLogger("utils.open3d_visualizer").setLevel(logging.WARNING)

        if self.args.load_model_from is None:
            self.main_logger.info(f"No specific model to load. Attempting to find best model in {self.args.save_dir}...")
            best_model_path = _find_best_model_to_load(self.args.save_dir, self.main_logger)
            if best_model_path:
                self.args.load_model_from = best_model_path
                self.main_logger.info(f"Autoloading best model from: {os.path.abspath(self.args.load_model_from)}")
            else:
                self.main_logger.info("No best model found to autoload. Starting fresh.")

        self.main_logger.info(f"Training session setup initiated by TrainingRunner.")
        self.main_logger.info(f"Log level set to: {self.args.log_level.upper()}")
        self.main_logger.info(f"Using device: {config.DEVICE}")
        self.main_logger.info(f"Model checkpoints will be saved in: {os.path.abspath(self.args.save_dir)}")
        self.main_logger.info(f"TensorBoard logs will be saved in: {os.path.abspath(TENSORBOARD_BASE_LOG_DIR)}")
        if self.args.load_model_from: # This will now reflect autoloaded path if one was found
            self.main_logger.info(f"Attempting to load model from: {os.path.abspath(self.args.load_model_from)}")
        else:
            self.main_logger.info("No model specified to load, starting fresh.")

        os.makedirs(TENSORBOARD_BASE_LOG_DIR, exist_ok=True)
        self.tb_logger = TensorBoardLogger(log_dir=TENSORBOARD_BASE_LOG_DIR, experiment_name=config.EXPERIMENT_NAME)
        self.main_logger.info(f"TensorBoard logger initialized. Logs in: {os.path.abspath(self.tb_logger.run_dir)}")

        self.main_logger.info("Initializing core training components...")
        self.env, self.replay_buffer, self.q_network_model, self.agent = initialize_training_components(
            self.args, config, self.numeric_log_level, self.main_logger # Pass config module
        )

        if self.env is None:
            self.main_logger.error("Failed to initialize training components. Exiting setup.")
            if self.tb_logger: self.tb_logger.close()
            sys.exit(1)
        self.main_logger.info("Core training components initialized successfully.")

        self.main_logger.info("Initializing DQNTrainer...")
        self.trainer = DQNTrainer(agent=self.agent, env=self.env, replay_buffer=self.replay_buffer,
                                  tb_logger=self.tb_logger, args=self.args, device=config.DEVICE)
        self.main_logger.info("DQNTrainer initialized.")
        self.main_logger.info("TrainingRunner setup complete.")

    def run(self):
        """Runs the main training loop."""
        if not self.trainer:
            self.main_logger.error("Trainer not initialized. Cannot run. Ensure setup() was called successfully.")
            return
        
        self.main_logger.info("Starting training loop via DQNTrainer...")
        try:
            self.trainer.train_loop()
            self.main_logger.info("Training loop completed.")
        except KeyboardInterrupt:
            self.main_logger.info("Training process interrupted by user.")
        except Exception as e:
            self.main_logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)

    def cleanup(self):
        """Cleans up all resources (environment, loggers)."""
        if self.main_logger:
            self.main_logger.info("Initiating cleanup of resources...")
        else:
            print("Initiating cleanup of resources (main_logger not available).")

        if self.env is not None:
            if self.main_logger: self.main_logger.info("Closing environment...")
            else: print("Closing environment...")
            self.env.close()
            self.env = None 
        else:
            if self.main_logger: self.main_logger.info("Environment was not initialized or already cleaned up.")

        if self.tb_logger is not None:
            if self.main_logger: self.main_logger.info("Closing TensorBoard logger...")
            else: print("Closing TensorBoard logger...")
            self.tb_logger.close()
            self.tb_logger = None
        else:
            if self.main_logger: self.main_logger.info("TensorBoard logger was not initialized or already cleaned up.")

        if self.main_logger:
            self.main_logger.info("TrainingRunner cleanup finished.")
        else:
            print("TrainingRunner cleanup finished (main_logger not available).")

def start_training_session():
    """Entry point function to set up, run, and cleanup a training session."""
    runner = TrainingRunner()
    main_logger = logging.getLogger("TrainingSession")
    should_cleanup = True
    profiler = None

    try:
        runner.setup() # Call setup first to parse arguments

        if hasattr(runner.args, 'enable_profiler') and runner.args.enable_profiler:
            main_logger.info("Profiler enabled.")
            profiler = cProfile.Profile()
            profiler.enable()
        
        runner.run()

    except SystemExit as e:
        if e.code == 0:
            should_cleanup = False 
        else:
            main_logger.error(f"SystemExit with code {e.code} during setup/run.", exc_info=True)
    except Exception as e:
        main_logger.error(f"Critical error in training session: {e}", exc_info=True)
    finally:
        if profiler:
            profiler.disable()
            main_logger.info("Profiler disabled. Printing stats:")
            stats = pstats.Stats(profiler).sort_stats('cumulative') # Sort by cumulative time
            stats.print_stats(30) # Print top 30 functions
            save_dir_for_profile = "."
            if runner and hasattr(runner, 'args') and runner.args and hasattr(runner.args, 'save_dir') and runner.args.save_dir:
                save_dir_for_profile = runner.args.save_dir
                os.makedirs(save_dir_for_profile, exist_ok=True) # Ensure dir exists
            profile_file_path = os.path.join(save_dir_for_profile, "profile_output.prof")
            try:
                stats.dump_stats(profile_file_path)
                main_logger.info(f"Profiler stats saved to {profile_file_path}")
            except Exception as dump_exc:
                main_logger.error(f"Could not dump profiler stats to {profile_file_path}: {dump_exc}")

        if should_cleanup:
            if runner:
                runner.cleanup()
            main_logger.info("Training session concluded.") 