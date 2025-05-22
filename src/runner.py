import logging
import os
import sys
import cProfile
import pstats

# Configuration
import config # Assuming src is in PYTHONPATH or current dir

# Utilities and Components
from utils.logger import Logger as TensorBoardLogger
from utils.setup_utils import parse_arguments, setup_logging
from utils.component_initializers import initialize_training_components
from training.dqn_trainer import DQNTrainer

class TrainingRunner:
    """Orchestrates the entire training session lifecycle."""
    def __init__(self):
        self.args = None
        # self.config = config # config is a module, can be imported directly where needed
        self.numeric_log_level = None
        self.main_logger = None # Logger for the runner itself
        
        # Components to be initialized and managed
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
        
        # Get a logger for the runner module (or a specific name)
        self.main_logger = logging.getLogger(__name__) 

        self.main_logger.info(f"Training session setup initiated by TrainingRunner.")
        self.main_logger.info(f"Log level set to: {self.args.log_level.upper()}")
        self.main_logger.info(f"Using device: {config.DEVICE}")
        self.main_logger.info(f"Models will be saved in: {os.path.abspath(self.args.save_dir)}")
        if self.args.load_model_from:
            self.main_logger.info(f"Attempting to load model from: {os.path.abspath(self.args.load_model_from)}")

        self.main_logger.info("Initializing TensorBoard logger...")
        self.tb_logger = TensorBoardLogger(log_dir=self.args.save_dir, experiment_name=config.EXPERIMENT_NAME)

        self.main_logger.info("Initializing core training components...")
        self.env, self.replay_buffer, self.q_network_model, self.agent = initialize_training_components(
            self.args, config, self.numeric_log_level, self.main_logger # Pass config module
        )

        if self.env is None: # Check if environment initialization failed
            self.main_logger.error("Failed to initialize training components. Exiting setup.")
            # self.cleanup() # Call cleanup, but tb_logger might not exist if env failed early.
            if self.tb_logger: self.tb_logger.close() # Ensure tb_logger is closed if created
            sys.exit(1) # Exit if components could not be initialized
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
        # Cleanup is called separately by the top-level function

    def cleanup(self):
        """Cleans up all resources (environment, loggers)."""
        if self.main_logger:
            self.main_logger.info("Initiating cleanup of resources...")
        else:
            # If main_logger isn't set up, use a basic print or a global logger if available
            print("Initiating cleanup of resources (main_logger not available).")

        if self.env is not None:
            if self.main_logger: self.main_logger.info("Closing environment...")
            else: print("Closing environment...")
            self.env.close()
            self.env = None 
        else:
            if self.main_logger: self.main_logger.info("Environment was not initialized or already cleaned up.")
            # else: print("Environment was not initialized or already cleaned up.") # Less critical to print this one

        if self.tb_logger is not None:
            if self.main_logger: self.main_logger.info("Closing TensorBoard logger...")
            else: print("Closing TensorBoard logger...")
            self.tb_logger.close()
            self.tb_logger = None
        else:
            if self.main_logger: self.main_logger.info("TensorBoard logger was not initialized or already cleaned up.")

        # Other components (agent, buffer, model, trainer) typically don't have explicit close methods
        # unless they manage external resources not tied to env or tb_logger.
        if self.main_logger:
            self.main_logger.info("TrainingRunner cleanup finished.")
        else:
            print("TrainingRunner cleanup finished (main_logger not available).")

def start_training_session():
    """Entry point function to set up, run, and cleanup a training session."""
    runner = TrainingRunner()
    main_logger = logging.getLogger("TrainingSession") # General logger before runner's logger is up
    should_cleanup = True
    profiler = None # Initialize profiler to None

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
            # You can also dump stats to a file for more detailed analysis with a profile viewer:
            # stats.dump_stats(os.path.join(runner.args.save_dir if runner.args else ".", "profile_output.prof"))
            # Ensure runner.args and runner.args.save_dir exist if dumping stats, or provide a default path.
            # For simplicity, let's ensure save_dir is accessible or use a default like current dir.
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