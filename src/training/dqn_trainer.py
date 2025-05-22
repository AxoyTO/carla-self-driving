import torch
import numpy as np
from collections import deque
import os
import logging
from typing import Tuple

# Assuming your project structure allows these imports from src/
# If run with `python -m src.main`, then these should work.
# from ..agents.dqn_agent import DQNAgent # Would be used if passed as type hint
# from ..environments.carla_env import CarlaEnv
# from ..utils.logger import Logger

logger = logging.getLogger(__name__) # Logger for this module

class DQNTrainer:
    def __init__(self, agent, env, replay_buffer, tb_logger, args, device):
        """
        Initializes the DQNTrainer.
        Args:
            agent: The DQN agent instance.
            env: The CARLA environment instance.
            replay_buffer: The replay buffer instance.
            tb_logger: The TensorBoard logger instance.
            args: Parsed command-line arguments (or a config object).
            device: PyTorch device ('cpu' or 'cuda').
        """
        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.tb_logger = tb_logger
        self.args = args
        self.device = device

        # Training loop parameters from args
        self.num_episodes = getattr(args, 'num_episodes', 1000)
        self.max_steps_per_episode = getattr(args, 'max_steps_per_episode', 1000)
        self.epsilon_start = getattr(args, 'epsilon_start', 1.0)
        self.epsilon_end = getattr(args, 'epsilon_end', 0.01)
        self.epsilon_decay = getattr(args, 'epsilon_decay', 0.995)
        
        # Evaluation parameters from args
        self.eval_interval = getattr(args, 'eval_interval', 25)
        self.num_eval_episodes = getattr(args, 'num_eval_episodes', 5)
        self.epsilon_eval = getattr(args, 'epsilon_eval', 0.01)

        # Saving parameters from args
        self.save_interval = getattr(args, 'save_interval', 50)
        self.run_specific_save_dir = getattr(args, 'run_specific_save_dir', './model_checkpoints/default_run') # Ensure this is passed correctly or set
        self.current_best_model_dir = getattr(args, 'current_best_model_dir', os.path.join(self.run_specific_save_dir, 'best_model'))

        self.start_episode = 1
        self.best_eval_score = -float('inf')
        self.last_eval_score = -float('inf')  # Track the most recent evaluation score for checkpoint saving
        
        if args.load_model_from:
            # Attempt to load best_eval_score if continuing a run
            # Try multiple possible locations for the best_score.txt file
            score_loaded = False
            
            # First, try to load from the model directory itself
            score_path = os.path.join(args.load_model_from, "best_score.txt")
            if os.path.exists(score_path):
                try:
                    with open(score_path, "r") as f:
                        self.best_eval_score = float(f.read().strip())
                        logger.info(f"Loaded best_eval_score from model directory: {self.best_eval_score}")
                        score_loaded = True
                except (ValueError, IOError) as e:
                    logger.warning(f"Error reading score file at {score_path}: {e}")
            
            # If not found in model directory, try parent directory
            if not score_loaded and os.path.dirname(args.load_model_from):
                parent_score_path = os.path.join(os.path.dirname(args.load_model_from), "best_score.txt")
                if os.path.exists(parent_score_path):
                    try:
                        with open(parent_score_path, "r") as f:
                            self.best_eval_score = float(f.read().strip())
                            logger.info(f"Loaded best_eval_score from parent directory: {self.best_eval_score}")
                            score_loaded = True
                    except (ValueError, IOError) as e:
                        logger.warning(f"Error reading score file at {parent_score_path}: {e}")
            
            if not score_loaded:
                logger.info("No valid best_score.txt found. Starting with fresh best score.")

    def _run_episode(self, i_episode: int, current_epsilon: float) -> Tuple[float, float, int, str, bool]:
        """Runs a single training episode.
        Returns:
            episode_score (float): Total score for the episode.
            avg_episode_loss (float): Average training loss for the episode (or None).
            steps_taken (int): Number of steps taken in the episode.
            termination_reason (str): Reason for episode termination.
            runtime_error_occurred (bool): Flag indicating if a runtime error happened.
        """
        try:
            state, _ = self.env.reset() # Reset is called here, at the start of each episode run
            episode_score = 0.0
            episode_losses = []
            termination_reason = "max_steps_reached"
            runtime_error_occurred = False

            for t_step in range(self.max_steps_per_episode):
                action = self.agent.select_action(state, current_epsilon)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                self.replay_buffer.add(state, action, reward, next_state, done)
                
                current_loss = self.agent.step_experience_and_learn()
                if current_loss is not None:
                    episode_losses.append(current_loss)

                state = next_state
                episode_score += reward
                if done:
                    if terminated:
                        termination_reason = info.get("termination_reason", "terminated_by_env")
                    # If truncated, termination_reason remains "max_steps_reached" (default or set if loop finishes)
                    break 
            # Ensure steps_taken is accurate, t_step is 0-indexed
            steps_taken = t_step + 1
            avg_episode_loss = np.mean(episode_losses) if episode_losses else None
            return episode_score, avg_episode_loss, steps_taken, termination_reason, runtime_error_occurred
        
        except RuntimeError as e:
            logger.error(f"Runtime error during episode {i_episode}: {e}", exc_info=True)
            # Return dummy values indicating failure within this episode run
            return 0.0, None, 0, "runtime_error", True

    def _log_episode_summary(self, i_episode: int, total_episodes: int, 
                             episode_score: float, avg_score_deque: float, 
                             current_epsilon: float, steps_taken: int, 
                             avg_episode_loss: float, termination_reason: str):
        """Logs the summary of a completed training episode."""
        logger.info(f"Episode {i_episode}/{total_episodes}\tSteps: {steps_taken}")
        logger.info(f"  Score: {episode_score:.2f}\tAvg Score (100 ep): {avg_score_deque:.2f}")
        avg_loss_str = f"{avg_episode_loss:.4f}" if avg_episode_loss is not None else "N/A"
        logger.info(f"  Avg Loss: {avg_loss_str}\tEpsilon (applied): {current_epsilon:.3f}") # Clarified this is epsilon applied in episode
        logger.info(f"  Termination: {termination_reason}")
        
        self.tb_logger.log_episode_stats(
            episode=i_episode, 
            score=episode_score, 
            avg_score=avg_score_deque, 
            epsilon=current_epsilon, # Epsilon used for this episode
            total_steps=steps_taken, 
            avg_loss=avg_episode_loss
        )

    def _handle_evaluation_and_saving(self, i_episode: int):
        """Handles model evaluation and saving checkpoints or best models."""
        # Evaluation
        if i_episode % self.eval_interval == 0:
            logger.info(f"--- Starting Evaluation after Training Episode {i_episode} ---")
            eval_avg_score, eval_goal_rate = self.evaluate_agent()
            self.last_eval_score = eval_avg_score  # Store the most recent eval score
            
            logger.info(f"Evaluation Avg Score: {eval_avg_score:.2f}, Goal Rate: {eval_goal_rate*100:.1f}%")
            self.tb_logger.log_scalar("evaluation/avg_score", eval_avg_score, i_episode)
            self.tb_logger.log_scalar("evaluation/goal_reached_rate", eval_goal_rate, i_episode)

            if eval_avg_score > self.best_eval_score:
                logger.info(f"New best evaluation score: {eval_avg_score:.2f} (previously: {self.best_eval_score:.2f}). Saving best model.")
                self.best_eval_score = eval_avg_score
                
                # Construct path for the best model (within the run-specific directory)
                # self.current_best_model_dir is already defined in __init__ as os.path.join(self.run_specific_save_dir, "best_model")
                os.makedirs(self.current_best_model_dir, exist_ok=True) # Ensure dir exists
                self.agent.save(self.current_best_model_dir, model_name="best_dqn_agent") # Pass directory
                
                # Save the best score to a file INSIDE the best model directory
                with open(os.path.join(self.current_best_model_dir, "best_score.txt"), "w") as f:
                    f.write(str(self.best_eval_score))
                    
                # For backward compatibility, also save in the parent directory
                with open(os.path.join(self.run_specific_save_dir, "best_score.txt"), "w") as f:
                    f.write(str(self.best_eval_score))
            logger.info(f"--- Finished Evaluation ---")

        # Periodic Checkpoint Saving
        if i_episode % self.save_interval == 0:
            checkpoint_dir = os.path.join(self.run_specific_save_dir, f"episode_{i_episode}")
            os.makedirs(checkpoint_dir, exist_ok=True) # Ensure dir exists
            self.agent.save(checkpoint_dir) # Pass directory for checkpoint
            
            # Save the most recent evaluation score with this checkpoint
            with open(os.path.join(checkpoint_dir, "score.txt"), "w") as f:
                f.write(str(self.last_eval_score))
                f.write(f"\nEval Goal Rate: {eval_goal_rate*100:.1f}%")
                
            logger.info(f"Saved checkpoint at episode {i_episode} to {checkpoint_dir} with score: {self.last_eval_score:.2f}")

    def train_loop(self):
        logger.info(f"Starting training loop for {self.num_episodes} episodes from episode {self.start_episode}.")
        scores_deque = deque(maxlen=100) # For rolling average score
        current_epsilon_for_episode = self.epsilon_start # Epsilon to be USED for the upcoming episode

        for i_episode in range(self.start_episode, self.num_episodes + 1):
            try:
                # Epsilon passed to _run_episode is the one for THIS episode
                episode_score, avg_episode_loss, steps_taken, termination_reason, runtime_error = \
                    self._run_episode(i_episode, current_epsilon_for_episode)

                if runtime_error:
                    logger.warning(f"Episode {i_episode} ended due to a runtime error. Stopping training loop.")
                    break 

                scores_deque.append(episode_score)
                avg_score_deque = np.mean(scores_deque)
                
                # Log summary for the completed episode (using epsilon that was applied)
                self._log_episode_summary(
                    i_episode, self.num_episodes, episode_score, avg_score_deque,
                    current_epsilon_for_episode, steps_taken, avg_episode_loss, termination_reason
                )
                
                # Decay epsilon for the NEXT episode
                current_epsilon_for_episode = max(self.epsilon_end, self.epsilon_decay * current_epsilon_for_episode)

                self._handle_evaluation_and_saving(i_episode)

            except KeyboardInterrupt:
                logger.info(f"Training loop interrupted by user at episode {i_episode}.") 
                break # Exit the main training loop
            finally:
                # Any cleanup specific to the end of an episode iteration (even if interrupted) could go here.
                pass 
            
        logger.info("Training loop finished.")
        final_model_dir = os.path.join(self.run_specific_save_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True) # Ensure dir exists
        self.agent.save(final_model_dir)
        
        # Save final scores
        with open(os.path.join(final_model_dir, "score.txt"), "w") as f:
            f.write(str(self.last_eval_score))
            if hasattr(self, 'last_goal_rate'):
                f.write(f"\nEval Goal Rate: {self.last_goal_rate*100:.1f}%")
            
        logger.info(f"Saved final model to {final_model_dir} with score: {self.last_eval_score:.2f}")

    def evaluate_agent(self):
        """Evaluates the agent's performance over a number of episodes."""
        logger.info(f"Running evaluation for {self.num_eval_episodes} episodes.")
        total_score = 0.0
        goals_reached = 0
        original_pygame_state = self.env.enable_pygame_display
        self.env.enable_pygame_display = False # Optionally disable rendering during eval for speed

        for i in range(self.num_eval_episodes):
            state, _ = self.env.reset()
            episode_score = 0
            done = False
            max_eval_steps = getattr(self.env, 'spec', {}).get('max_episode_steps', 1000) if hasattr(self.env, 'spec') else 1000
            
            for eval_step in range(max_eval_steps): 
                action = self.agent.select_action(state, self.epsilon_eval)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                state = next_state
                episode_score += reward
                if done:
                    if info.get("termination_reason") == "goal_reached":
                        goals_reached += 1
                    break
            total_score += episode_score
            logger.info(f"  Eval Episode {i+1}/{self.num_eval_episodes} Score: {episode_score:.2f}, Termination: {info.get('termination_reason', 'unknown')}")
        
        self.env.enable_pygame_display = original_pygame_state # Restore pygame state
        avg_score = total_score / self.num_eval_episodes if self.num_eval_episodes > 0 else 0
        goal_reached_rate = goals_reached / self.num_eval_episodes if self.num_eval_episodes > 0 else 0
        return avg_score, goal_reached_rate 