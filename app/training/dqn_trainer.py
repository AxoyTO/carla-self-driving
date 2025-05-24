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
        self.model_base_save_dir = args.save_dir 
        self.current_best_model_dir = os.path.join(self.model_base_save_dir, "best_model")

        self.start_episode = 1
        self.best_eval_score = -float('inf')
        self.last_eval_score = -float('inf')
        
        if args.load_model_from:
            # Try to load best_eval_score from the *parent* of the loaded model dir if it's a 'best_model' dir,
            # or from the model dir itself if it's an episode checkpoint.
            # The primary source for best_eval_score should now be the best_score.txt directly in self.model_base_save_dir
            score_file_in_base_dir = os.path.join(self.model_base_save_dir, "best_score.txt")
            score_file_in_loaded_model_dir = os.path.join(args.load_model_from, "best_score.txt") # If loading from best_model dir

            score_to_load = None
            if os.path.exists(score_file_in_base_dir):
                score_to_load = score_file_in_base_dir
            elif os.path.exists(score_file_in_loaded_model_dir) and "best_model" in args.load_model_from.lower():
                score_to_load = score_file_in_loaded_model_dir
            # Add other potential legacy paths if needed here, but new saves will put it in model_base_save_dir

            if score_to_load:
                try:
                    with open(score_to_load, "r") as f:
                        self.best_eval_score = float(f.read().strip())
                        logger.info(f"Loaded best_eval_score from {score_to_load}: {self.best_eval_score}")
                except (ValueError, IOError) as e:
                    logger.warning(f"Error reading score file at {score_to_load}: {e}")
            else:
                logger.info("No valid best_score.txt found in expected locations. Starting with fresh best score if not resuming a checkpoint score.")

    def _process_step(self, state, action) -> Tuple[dict, float, bool, bool, dict]:
        """Processes a single step in the environment.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Tuple containing:
            - next_state: New state after action
            - reward: Reward received
            - terminated: Whether episode terminated
            - truncated: Whether episode was truncated
            - info: Additional info from environment
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state, reward, terminated, truncated, info

    def _update_replay_buffer(self, state, action, reward, next_state, done):
        """Updates the replay buffer with the latest experience."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def _run_episode(self, i_episode: int, current_epsilon: float) -> Tuple[float, float, int, str, bool]:
        """Runs a single training episode.
        
        Args:
            i_episode: Current episode number
            current_epsilon: Current epsilon value for exploration
            
        Returns:
            Tuple containing:
            - episode_score: Total score for the episode
            - avg_episode_loss: Average training loss for the episode
            - steps_taken: Number of steps taken
            - termination_reason: Why the episode ended
            - runtime_error_occurred: Whether a runtime error happened
        """
        try:
            state, _ = self.env.reset()
            episode_score = 0.0
            episode_losses = []
            termination_reason = "max_steps_reached"
            runtime_error_occurred = False

            for t_step in range(self.max_steps_per_episode):
                # Select and execute action
                action = self.agent.select_action(state, current_epsilon)
                next_state, reward, terminated, truncated, info = self._process_step(state, action)
                done = terminated or truncated

                # Update replay buffer and learn
                self._update_replay_buffer(state, action, reward, next_state, done)
                current_loss = self.agent.step_experience_and_learn()
                if current_loss is not None:
                    episode_losses.append(current_loss)

                # Update state and score
                state = next_state
                episode_score += reward

                if done:
                    if terminated:
                        termination_reason = info.get("termination_reason", "terminated_by_env")
                    break

            # Calculate final metrics
            steps_taken = t_step + 1
            avg_episode_loss = np.mean(episode_losses) if episode_losses else None
            return episode_score, avg_episode_loss, steps_taken, termination_reason, runtime_error_occurred

        except RuntimeError as e:
            logger.error(f"Runtime error during episode {i_episode}: {e}", exc_info=True)
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
            self.last_eval_score = eval_avg_score
            
            logger.info(f"Evaluation Avg Score: {eval_avg_score:.2f}, Goal Rate: {eval_goal_rate*100:.1f}%")
            self.tb_logger.log_scalar("evaluation/avg_score", eval_avg_score, i_episode)
            self.tb_logger.log_scalar("evaluation/goal_reached_rate", eval_goal_rate, i_episode)

            if eval_avg_score > self.best_eval_score:
                logger.info(f"New best: {eval_avg_score:.2f} (prev: {self.best_eval_score:.2f}). Saving to {self.current_best_model_dir}")
                self.best_eval_score = eval_avg_score
                os.makedirs(self.current_best_model_dir, exist_ok=True)
                self.agent.save(self.current_best_model_dir, model_name="best_dqn_agent")
                
                # Save the best score to a file directly in the model_base_save_dir
                with open(os.path.join(self.model_base_save_dir, "best_score.txt"), "w") as f:
                    f.write(str(self.best_eval_score))
                # Also save it inside the best_model directory for redundancy/clarity
                with open(os.path.join(self.current_best_model_dir, "best_score.txt"), "w") as f:
                    f.write(str(self.best_eval_score))
            logger.info(f"--- Finished Evaluation ---")

        # Periodic Checkpoint Saving
        if i_episode % self.save_interval == 0:
            # Checkpoints are saved directly under model_base_save_dir now
            checkpoint_dir = os.path.join(self.model_base_save_dir, f"episode_{i_episode}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.agent.save(checkpoint_dir)
            
            with open(os.path.join(checkpoint_dir, "score.txt"), "w") as f:
                f.write(str(self.last_eval_score))
                f.write(f"\nEval Goal Rate: {eval_goal_rate*100:.1f}%")
                
            logger.info(f"Saved checkpoint at ep {i_episode} to {checkpoint_dir} with score: {self.last_eval_score:.2f}")

    def _run_evaluation_episode(self, episode_num: int) -> Tuple[float, int, str]:
        """Runs a single evaluation episode.
        
        Args:
            episode_num: The episode number being evaluated
            
        Returns:
            Tuple containing:
            - episode_score: Total score for the episode
            - steps_taken: Number of steps taken
            - termination_reason: Why the episode ended
        """
        state, _ = self.env.reset()
        episode_score = 0
        max_eval_steps = getattr(self.env, 'spec', {}).get('max_episode_steps', 1000) if hasattr(self.env, 'spec') else 1000
        
        for eval_step in range(max_eval_steps):
            action = self.agent.select_action(state, self.epsilon_eval)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            state = next_state
            episode_score += reward
            
            if done:
                termination_reason = info.get("termination_reason", "unknown")
                logger.info(f"  Eval Episode {episode_num}/{self.num_eval_episodes} "
                          f"Score: {episode_score:.2f}, "
                          f"Steps: {eval_step + 1}, "
                          f"Termination: {termination_reason}")
                return episode_score, eval_step + 1, termination_reason
                
        # If we get here, we hit max steps
        return episode_score, max_eval_steps, "max_steps_reached"

    def _log_evaluation_summary(self, total_score: float, goals_reached: int, 
                              termination_reasons: dict, num_episodes: int):
        """Logs a summary of the evaluation results.
        
        Args:
            total_score: Sum of all episode scores
            goals_reached: Number of episodes where goal was reached
            termination_reasons: Dict mapping reasons to counts
            num_episodes: Total number of evaluation episodes
        """
        avg_score = total_score / num_episodes if num_episodes > 0 else 0
        goal_reached_rate = goals_reached / num_episodes if num_episodes > 0 else 0
        
        logger.info("\nEvaluation Summary:")
        logger.info(f"  Average Score: {avg_score:.2f}")
        logger.info(f"  Goal Reached Rate: {goal_reached_rate*100:.1f}%")
        logger.info("  Termination Reasons:")
        for reason, count in termination_reasons.items():
            percentage = (count / num_episodes) * 100
            logger.info(f"    - {reason}: {count} episodes ({percentage:.1f}%)")

    def evaluate_agent(self):
        """Evaluates the agent's performance over a number of episodes.
        
        Returns:
            Tuple containing:
            - avg_score: Average score across all evaluation episodes
            - goal_reached_rate: Percentage of episodes where goal was reached
        """
        logger.info(f"Running evaluation for {self.num_eval_episodes} episodes.")
        total_score = 0.0
        goals_reached = 0
        termination_reasons = {}
        
        # Store original pygame state and disable during evaluation
        original_pygame_state = self.env.enable_pygame_display
        self.env.enable_pygame_display = False

        try:
            for i in range(self.num_eval_episodes):
                episode_score, steps_taken, termination_reason = self._run_evaluation_episode(i + 1)
                
                # Update statistics
                total_score += episode_score
                termination_reasons[termination_reason] = termination_reasons.get(termination_reason, 0) + 1
                
                if termination_reason in ["goal_reached", "goal_reached_and_stopped"]:
                    goals_reached += 1
            
            # Log summary of all episodes
            self._log_evaluation_summary(total_score, goals_reached, 
                                       termination_reasons, self.num_eval_episodes)
            
            # Calculate final metrics
            avg_score = total_score / self.num_eval_episodes if self.num_eval_episodes > 0 else 0
            goal_reached_rate = goals_reached / self.num_eval_episodes if self.num_eval_episodes > 0 else 0
            
            return avg_score, goal_reached_rate
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            return 0.0, 0.0
        finally:
            self.env.enable_pygame_display = original_pygame_state  # Restore pygame state

    def _update_epsilon(self, current_epsilon: float) -> float:
        """Updates epsilon for the next episode using decay.
        
        Args:
            current_epsilon: Current epsilon value
            
        Returns:
            New epsilon value after decay
        """
        return max(self.epsilon_end, self.epsilon_decay * current_epsilon)

    def _save_final_model(self):
        """Saves the final model and its performance metrics."""
        # Final model also saved directly under model_base_save_dir
        final_model_dir = os.path.join(self.model_base_save_dir, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        self.agent.save(final_model_dir)
        
        with open(os.path.join(final_model_dir, "score.txt"), "w") as f:
            f.write(str(self.last_eval_score))
            if hasattr(self, 'last_goal_rate'): # Check if last_goal_rate was set
                f.write(f"\nEval Goal Rate: {self.last_goal_rate*100:.1f}%")
            
        logger.info(f"Saved final model to {final_model_dir} with score: {self.last_eval_score:.2f}")

    def train_loop(self):
        """Main training loop that runs for the specified number of episodes."""
        logger.info(f"Starting training loop for {self.num_episodes} episodes from episode {self.start_episode}.")
        scores_deque = deque(maxlen=100)  # For rolling average score
        current_epsilon = self.epsilon_start  # Epsilon for the upcoming episode

        try:
            for i_episode in range(self.start_episode, self.num_episodes + 1):
                # Run episode with current epsilon
                episode_score, avg_episode_loss, steps_taken, termination_reason, runtime_error = \
                    self._run_episode(i_episode, current_epsilon)

                if runtime_error:
                    logger.warning(f"Episode {i_episode} ended due to a runtime error. Stopping training loop.")
                    break 

                # Update statistics
                scores_deque.append(episode_score)
                avg_score_deque = np.mean(scores_deque)
                
                # Log episode summary
                self._log_episode_summary(
                    i_episode, self.num_episodes, episode_score, avg_score_deque,
                    current_epsilon, steps_taken, avg_episode_loss, termination_reason
                )
                
                # Update epsilon for next episode
                current_epsilon = self._update_epsilon(current_epsilon)

                # Handle evaluation and model saving
                self._handle_evaluation_and_saving(i_episode)

        except KeyboardInterrupt:
            logger.info(f"Training loop interrupted by user at episode {i_episode}.")
        finally:
            self._save_final_model()
            logger.info("Training loop finished.") 