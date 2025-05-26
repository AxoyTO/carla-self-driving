import torch
import numpy as np
from collections import deque
import os
import logging
from typing import Tuple, Optional
import time
from datetime import datetime

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
        
        # Create training run reports directory
        self.training_start_time = datetime.now()
        self.run_id = f"training_run_{self.training_start_time.strftime('%Y%m%d_%H%M%S')}"
        self.reports_dir = os.path.join("reports", "training_runs", self.run_id)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs("reports/best_model_reports", exist_ok=True)
        
        # Log training run info
        run_info_path = os.path.join(self.reports_dir, "training_info.txt")
        with open(run_info_path, "w") as f:
            f.write(f"Training Run: {self.run_id}\n")
            f.write(f"Started: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Episodes: {self.num_episodes}\n")
            f.write(f"Model Save Dir: {self.model_base_save_dir}\n")
            f.write("="*50 + "\n\n")
        
        logger.info(f"Training reports will be saved to: {self.reports_dir}")
        
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

        # Enhanced curriculum learning tracking
        self.curriculum_progress_tracker = {
            'phase_completion_history': [],
            'difficulty_adjustments': [],
            'performance_trends': deque(maxlen=20),  # Track last 20 evaluations
            'adaptive_threshold_adjustments': 0
        }
        
        # Adaptive training parameters
        self.adaptive_eval_interval = self.eval_interval
        self.min_eval_interval = 10
        self.max_eval_interval = 100
        
        # Performance-based epsilon adjustment
        self.adaptive_epsilon_enabled = getattr(args, 'adaptive_epsilon', True)
        self.base_epsilon_decay = self.epsilon_decay
        self.performance_buffer = deque(maxlen=10)
        
        # Early stopping and training stability
        self.early_stopping_enabled = getattr(args, 'early_stopping', True)
        self.patience = getattr(args, 'early_stopping_patience', 50)
        self.min_improvement = getattr(args, 'min_improvement_threshold', 0.01)
        self.best_score_history = deque(maxlen=self.patience)
        self.no_improvement_count = 0
        
        # Training stability monitoring
        self.stability_metrics = {
            'loss_variance_window': deque(maxlen=50),
            'reward_variance_window': deque(maxlen=100),
            'gradient_explosion_count': 0,
            'performance_plateau_count': 0,
            'curriculum_regression_count': 0
        }
        
        # Advanced performance tracking
        self.performance_analytics = {
            'episode_efficiency': [],  # Steps to goal ratio
            'learning_rate_history': [],
            'phase_transition_scores': [],
            'skill_retention_scores': {},  # Track performance across different skills
            'exploration_efficiency': []  # How well epsilon is being used
        }

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
            # Reset LSTM hidden state at the beginning of the episode if agent uses LSTM
            if hasattr(self.agent, 'use_lstm') and self.agent.use_lstm:
                self.agent.reset_lstm_hidden_state(batch_size=1) # Batch size 1 for action selection
                
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
        # Check if curriculum phase evaluation should be triggered
        should_eval_phase = (hasattr(self.env, 'curriculum_manager') and 
                            self.env.curriculum_manager and 
                            self.env.curriculum_manager.should_evaluate_phase())
        
        # Regular evaluation interval or phase evaluation
        should_eval_regular = i_episode % self.eval_interval == 0
        
        if should_eval_regular or should_eval_phase:
            if should_eval_phase:
                logger.info(f"--- Starting Automatic Phase Evaluation after Episode {i_episode} ---")
                # Use more episodes for phase evaluation
                eval_episodes = self.env.curriculum_manager.evaluation_episodes
                eval_type = "phase"
            else:
                logger.info(f"--- Starting Regular Evaluation after Episode {i_episode} ---")
                eval_episodes = self.num_eval_episodes
                eval_type = "regular"
            
            # Temporarily override num_eval_episodes for this evaluation
            original_eval_episodes = self.num_eval_episodes
            self.num_eval_episodes = eval_episodes
            
            try:
                avg_score, goal_rate = self.evaluate_agent()
                self.last_eval_score = avg_score
                
                # Get comprehensive metrics from the last evaluation
                performance_metrics = {}
                if hasattr(self, '_last_performance_report') and self._last_performance_report:
                    report = self._last_performance_report
                    performance_metrics = {
                        'goal_completion_rate': report.get('goal_success_rate', 0.0) / 100.0,
                        'collision_free_rate': report.get('collision_free_rate', 0.0) / 100.0,
                        'sidewalk_free_rate': report.get('sidewalk_free_rate', 0.0) / 100.0,
                        'violations_per_episode': report.get('violations_per_episode', 0.0),
                        'driving_score': report.get('overall_driving_score', 0.0)
                    }
                
                # Save detailed evaluation report to file
                if hasattr(self, '_last_performance_report') and self._last_performance_report:
                    self._save_evaluation_report(i_episode, eval_type, self._last_performance_report)
                
                # Minimal console output
                report = self._last_performance_report if hasattr(self, '_last_performance_report') else {}
                driving_score = report.get('overall_driving_score', 0.0)
                grade = report.get('performance_grade', 'N/A')
                goal_rate_pct = report.get('goal_success_rate', 0.0)
                collision_free_pct = report.get('collision_free_rate', 0.0)
                rule_compliance_pct = report.get('rule_compliance_rate', 0.0)
                
                report_filename = f"episode_{i_episode:03d}_{eval_type}_evaluation.txt"
                report_path = os.path.join(self.reports_dir, report_filename)
                
                logger.info(f"Episode {i_episode}: Evaluation completed - Driving Score: {driving_score:.1f}/100 ({grade})")
                logger.info(f"  Goal Rate: {goal_rate_pct:.1f}% | Collision-Free: {collision_free_pct:.1f}% | Rule Compliance: {rule_compliance_pct:.1f}%")
                logger.info(f"  Detailed report saved to: {report_path}")
                
                # Handle phase evaluation if needed
                if should_eval_phase:
                    phase_passed, evaluation_summary = self.env.curriculum_manager.evaluate_phase_completion(performance_metrics)
                    should_repeat_phase = self.env.curriculum_manager.handle_phase_evaluation_result(phase_passed, evaluation_summary)
                    
                    # Save phase evaluation results to file
                    phase_report_path = os.path.join(self.reports_dir, f"episode_{i_episode:03d}_phase_evaluation.txt")
                    with open(phase_report_path, "w") as f:
                        f.write(f"PHASE EVALUATION RESULTS - Episode {i_episode}\n")
                        f.write("="*60 + "\n\n")
                        f.write(evaluation_summary)
                        f.write(f"\n\nResult: {'PASSED' if phase_passed else 'FAILED'}\n")
                        f.write(f"Action: {'Advance to next phase' if not should_repeat_phase else 'Repeat current phase'}\n")
                    
                    if should_repeat_phase:
                        logger.info("Phase evaluation: FAILED - Repeating phase")
                    else:
                        logger.info("Phase evaluation: PASSED - Advancing to next phase")
                        self.env.curriculum_manager._advance_to_next_phase()
                
                # Regular evaluation logging and model saving
                if should_eval_regular:
                    # Log traditional metrics for backward compatibility
                    self.tb_logger.log_scalar("evaluation/avg_score", avg_score, i_episode)
                    self.tb_logger.log_scalar("evaluation/goal_reached_rate", goal_rate, i_episode)
                    
                    # Log comprehensive metrics if available (from the last evaluation)
                    if hasattr(self, '_last_performance_report') and self._last_performance_report:
                        report = self._last_performance_report
                        
                        # Overall Performance
                        self.tb_logger.log_scalar("performance/overall_driving_score", report['overall_driving_score'], i_episode)
                        self.tb_logger.log_scalar("performance/raw_avg_score", report['raw_avg_score'], i_episode)
                        
                        # Success Metrics
                        self.tb_logger.log_scalar("success/goal_completion_rate", report['goal_success_rate'], i_episode)
                        self.tb_logger.log_scalar("success/path_efficiency", report['avg_path_efficiency'], i_episode)
                        self.tb_logger.log_scalar("success/avg_time_to_goal", report['avg_time_to_goal_seconds'], i_episode)
                        
                        # Safety Metrics
                        self.tb_logger.log_scalar("safety/collision_free_rate", report['collision_free_rate'], i_episode)
                        self.tb_logger.log_scalar("safety/sidewalk_free_rate", report['sidewalk_free_rate'], i_episode)
                        self.tb_logger.log_scalar("safety/rule_compliance_rate", report['rule_compliance_rate'], i_episode)
                        self.tb_logger.log_scalar("safety/violations_per_episode", report['violations_per_episode'], i_episode)
                        
                        # Driving Quality
                        self.tb_logger.log_scalar("quality/avg_speed_kmh", report['avg_speed_kmh'], i_episode)
                        self.tb_logger.log_scalar("quality/max_speed_kmh", report['max_speed_achieved_kmh'], i_episode)
                        self.tb_logger.log_scalar("quality/smoothness_score", report['avg_smoothness_score'], i_episode)
                        
                        # Log termination breakdown as individual metrics
                        for reason, count in report['termination_breakdown'].items():
                            percentage = (count / report['num_episodes']) * 100
                            self.tb_logger.log_scalar(f"termination/{reason}_percentage", percentage, i_episode)

                    if avg_score > self.best_eval_score:
                        logger.info(f"New best score: {avg_score:.2f} (previous: {self.best_eval_score:.2f})")
                        self.best_eval_score = avg_score
                os.makedirs(self.current_best_model_dir, exist_ok=True)
                self.agent.save(self.current_best_model_dir, model_name="best_dqn_agent")
                
                # Save the best score to a file directly in the model_base_save_dir
                with open(os.path.join(self.model_base_save_dir, "best_score.txt"), "w") as f:
                    f.write(str(self.best_eval_score))
                # Also save it inside the best_model directory for redundancy/clarity
                with open(os.path.join(self.current_best_model_dir, "best_score.txt"), "w") as f:
                    f.write(str(self.best_eval_score))
                            
                    # Save comprehensive metrics for the best model
                    if hasattr(self, '_last_performance_report') and self._last_performance_report:
                        best_model_report_path = os.path.join("reports/best_model_reports", f"best_score_{driving_score:.1f}_episode_{i_episode:03d}.txt")
                        with open(best_model_report_path, "w") as f:
                            self._write_comprehensive_report(f, self._last_performance_report, f"BEST MODEL PERFORMANCE REPORT (Episode {i_episode})")
                        
                        # Also save in the model directory for backward compatibility
                        with open(os.path.join(self.current_best_model_dir, "performance_report.txt"), "w") as f:
                            self._write_comprehensive_report(f, self._last_performance_report, f"BEST MODEL PERFORMANCE REPORT (Episode {i_episode})")
                
            finally:
                # Restore original num_eval_episodes
                self.num_eval_episodes = original_eval_episodes
                
            logger.info(f"--- Finished Evaluation ---")

        # Periodic Checkpoint Saving (only on regular intervals, not phase evaluations)
        if i_episode % self.save_interval == 0:
            # Checkpoints are saved directly under model_base_save_dir now
            checkpoint_dir = os.path.join(self.model_base_save_dir, f"episode_{i_episode}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.agent.save(checkpoint_dir)
            
            with open(os.path.join(checkpoint_dir, "score.txt"), "w") as f:
                f.write(str(self.last_eval_score))
                if 'goal_rate' in locals():
                    f.write(f"\nEval Goal Rate: {goal_rate*100:.1f}%")
                
            logger.info(f"Saved checkpoint at episode {i_episode} with score: {self.last_eval_score:.2f}")

    def _save_evaluation_report(self, episode: int, eval_type: str, report: dict):
        """Save detailed evaluation report to file."""
        report_filename = f"episode_{episode:03d}_{eval_type}_evaluation.txt"
        report_path = os.path.join(self.reports_dir, report_filename)
        
        with open(report_path, "w") as f:
            title = f"EVALUATION REPORT - Episode {episode} ({eval_type.upper()})"
            self._write_comprehensive_report(f, report, title)

    def _write_comprehensive_report(self, file_obj, report: dict, title: str):
        """Write comprehensive evaluation report to file object."""
        file_obj.write(title + "\n")
        file_obj.write("="*len(title) + "\n")
        file_obj.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall Performance
        file_obj.write("OVERALL PERFORMANCE:\n")
        file_obj.write(f"   Driving Score: {report['overall_driving_score']:.1f}/100\n")
        file_obj.write(f"   Performance Grade: {report['performance_grade']}\n")
        file_obj.write(f"   Raw Reward Score: {report['raw_avg_score']:.2f}\n\n")
        
        # Success Metrics
        file_obj.write("SUCCESS METRICS:\n")
        file_obj.write(f"   Goal Completion Rate: {report['goal_success_rate']:.1f}%\n")
        file_obj.write(f"   Average Time to Goal: {report['avg_time_to_goal_seconds']:.1f} seconds\n")
        file_obj.write(f"   Path Efficiency: {report['avg_path_efficiency']:.2f}x optimal\n\n")
        
        # Safety Metrics
        file_obj.write("SAFETY METRICS:\n")
        file_obj.write(f"   Collision-Free Rate: {report['collision_free_rate']:.1f}%\n")
        file_obj.write(f"   Sidewalk-Free Rate: {report['sidewalk_free_rate']:.1f}%\n")
        file_obj.write(f"   Rule Compliance Rate: {report['rule_compliance_rate']:.1f}%\n")
        file_obj.write(f"   Total Violations: {report['total_rule_violations']} ({report['violations_per_episode']:.1f} per episode)\n\n")
        
        # Driving Quality
        file_obj.write("DRIVING QUALITY:\n")
        file_obj.write(f"   Average Speed: {report['avg_speed_kmh']:.1f} km/h\n")
        file_obj.write(f"   Max Speed Achieved: {report['max_speed_achieved_kmh']:.1f} km/h\n")
        file_obj.write(f"   Smoothness Score: {report['avg_smoothness_score']:.1f}/100\n\n")
        
        # Failure Analysis
        file_obj.write("FAILURE ANALYSIS:\n")
        for reason, count in report['termination_breakdown'].items():
            percentage = (count / report['num_episodes']) * 100
            file_obj.write(f"   {reason}: {count} episodes ({percentage:.1f}%)\n")
        file_obj.write("\n")
        
        # Performance Interpretation
        file_obj.write("INTERPRETATION:\n")
        if report['overall_driving_score'] >= 85:
            file_obj.write("   EXCELLENT! This model is ready for advanced scenarios.\n")
        elif report['overall_driving_score'] >= 70:
            file_obj.write("   GOOD performance. Minor improvements needed.\n")
        elif report['overall_driving_score'] >= 55:
            file_obj.write("   FAIR performance. Significant training still required.\n")
        else:
            file_obj.write("   POOR performance. Major improvements needed.\n")
        file_obj.write("\n")
        
        # Detailed Episode Breakdown
        file_obj.write("DETAILED METRICS:\n")
        file_obj.write(f"   Episodes Evaluated: {report['num_episodes']}\n")
        file_obj.write(f"   Total Collisions: {report['total_collisions']}\n")
        file_obj.write(f"   Total Sidewalk Violations: {report['total_sidewalk_violations']}\n")
        file_obj.write(f"   Total Rule Violations: {report['total_rule_violations']}\n")
        file_obj.write("="*60 + "\n")

    def _run_evaluation_episode(self, episode_num: int) -> Tuple[float, int, str, dict]:
        """Runs a single evaluation episode.
        
        Args:
            episode_num: The episode number being evaluated
            
        Returns:
            Tuple containing:
            - episode_score: Total score for the episode
            - steps_taken: Number of steps taken
            - termination_reason: Why the episode ended
            - detailed_metrics: Dictionary with detailed performance metrics
        """
        # Reset LSTM hidden state at the beginning of the evaluation episode
        if hasattr(self.agent, 'use_lstm') and self.agent.use_lstm:
            self.agent.reset_lstm_hidden_state(batch_size=1) # Batch size 1 for action selection
            
        state, _ = self.env.reset()
        episode_score = 0
        max_eval_steps = getattr(self.env, 'spec', {}).get('max_episode_steps', 1000) if hasattr(self.env, 'spec') else 1000
        
        # Track detailed metrics for this episode
        detailed_metrics = {
            'goal_reached': False,
            'collision_occurred': False,
            'sidewalk_violations': 0,
            'traffic_light_violations': 0,
            'time_to_goal_seconds': 0.0,
            'distance_traveled': 0.0,
            'initial_distance_to_goal': 0.0,
            'final_distance_to_goal': 0.0,
            'avg_speed_kmh': 0.0,
            'max_speed_kmh': 0.0,
            'smooth_driving_score': 0.0,
            'lane_keeping_score': 0.0,
            'path_efficiency': 0.0,
            'rule_violations': 0,
            'steps_taken': 0
        }
        
        # Track data for smooth driving calculation
        previous_location = None
        previous_steer = 0.0
        speed_history = []
        steer_changes = []
        
        # Get initial distance to goal
        if hasattr(self.env, 'dist_to_goal_debug'):
            detailed_metrics['initial_distance_to_goal'] = self.env.dist_to_goal_debug
        
        episode_start_time = time.time()
        
        for eval_step in range(max_eval_steps):
            action = self.agent.select_action(state, self.epsilon_eval)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            state = next_state
            episode_score += reward
            
            # Collect detailed metrics during the episode
            if hasattr(self.env, 'vehicle_manager') and self.env.vehicle_manager.get_vehicle():
                vehicle = self.env.vehicle_manager.get_vehicle()
                current_location = vehicle.get_location()
                current_speed_kmh = self.env.forward_speed_debug * 3.6
                
                # Track speed
                speed_history.append(current_speed_kmh)
                detailed_metrics['max_speed_kmh'] = max(detailed_metrics['max_speed_kmh'], current_speed_kmh)
                
                # Track distance traveled
                if previous_location:
                    distance_step = current_location.distance(previous_location)
                    detailed_metrics['distance_traveled'] += distance_step
                previous_location = current_location
                
                # Track steering smoothness
                current_steer = vehicle.get_control().steer
                if eval_step > 0:
                    steer_change = abs(current_steer - previous_steer)
                    steer_changes.append(steer_change)
                previous_steer = current_steer
            
            # Track violations
            if hasattr(self.env, 'collision_flag_debug') and self.env.collision_flag_debug:
                detailed_metrics['collision_occurred'] = True
                detailed_metrics['rule_violations'] += 1
                
            if hasattr(self.env, 'on_sidewalk_debug_flag') and self.env.on_sidewalk_debug_flag:
                detailed_metrics['sidewalk_violations'] += 1
                detailed_metrics['rule_violations'] += 1
            
            if done:
                termination_reason = info.get("termination_reason", "unknown")
                detailed_metrics['time_to_goal_seconds'] = time.time() - episode_start_time
                detailed_metrics['steps_taken'] = eval_step + 1
                
                # Final distance to goal
                if hasattr(self.env, 'dist_to_goal_debug'):
                    detailed_metrics['final_distance_to_goal'] = self.env.dist_to_goal_debug
                
                # Calculate derived metrics
                if len(speed_history) > 0:
                    detailed_metrics['avg_speed_kmh'] = sum(speed_history) / len(speed_history)
                
                # Goal reached check
                if termination_reason in ["goal_reached", "goal_reached_and_stopped"]:
                    detailed_metrics['goal_reached'] = True
                
                # Path efficiency (lower is better - 1.0 is perfect)
                if detailed_metrics['initial_distance_to_goal'] > 0:
                    detailed_metrics['path_efficiency'] = detailed_metrics['distance_traveled'] / detailed_metrics['initial_distance_to_goal']
                
                # Smooth driving score (0-100, higher is better)
                if len(steer_changes) > 0:
                    avg_steer_change = sum(steer_changes) / len(steer_changes)
                    detailed_metrics['smooth_driving_score'] = max(0, 100 - (avg_steer_change * 1000))
                else:
                    detailed_metrics['smooth_driving_score'] = 100
                
                logger.debug(f"  Eval Episode {episode_num}/{self.num_eval_episodes} "
                          f"Score: {episode_score:.2f}, "
                          f"Steps: {eval_step + 1}, "
                          f"Goal: {'✓' if detailed_metrics['goal_reached'] else '✗'}, "
                          f"Collisions: {1 if detailed_metrics['collision_occurred'] else 0}, "
                          f"Sidewalk: {detailed_metrics['sidewalk_violations']}, "
                          f"Speed: {detailed_metrics['avg_speed_kmh']:.1f}km/h, "
                          f"Termination: {termination_reason}")
                return episode_score, eval_step + 1, termination_reason, detailed_metrics
                
        # If we get here, we hit max steps
        detailed_metrics['time_to_goal_seconds'] = time.time() - episode_start_time
        detailed_metrics['steps_taken'] = max_eval_steps
        
        if len(speed_history) > 0:
            detailed_metrics['avg_speed_kmh'] = sum(speed_history) / len(speed_history)
        if detailed_metrics['initial_distance_to_goal'] > 0:
            detailed_metrics['path_efficiency'] = detailed_metrics['distance_traveled'] / detailed_metrics['initial_distance_to_goal']
        if len(steer_changes) > 0:
            avg_steer_change = sum(steer_changes) / len(steer_changes)
            detailed_metrics['smooth_driving_score'] = max(0, 100 - (avg_steer_change * 1000))
            
        return episode_score, max_eval_steps, "max_steps_reached", detailed_metrics

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
        
        # Collect detailed metrics across all episodes
        all_detailed_metrics = []
        
        # Store original pygame state and disable during evaluation
        original_pygame_state = self.env.enable_pygame_display
        self.env.enable_pygame_display = False

        try:
            for i in range(self.num_eval_episodes):
                episode_score, steps_taken, termination_reason, detailed_metrics = self._run_evaluation_episode(i + 1)
                
                # Update statistics
                total_score += episode_score
                termination_reasons[termination_reason] = termination_reasons.get(termination_reason, 0) + 1
                all_detailed_metrics.append(detailed_metrics)
                
                if termination_reason in ["goal_reached", "goal_reached_and_stopped"]:
                    goals_reached += 1
            
            # Calculate comprehensive performance metrics
            performance_report = self._calculate_performance_metrics(all_detailed_metrics, total_score, goals_reached, termination_reasons)
            
            # Store for TensorBoard logging and model saving
            self._last_performance_report = performance_report
            
            # Log traditional metrics for compatibility
            avg_score = total_score / self.num_eval_episodes if self.num_eval_episodes > 0 else 0
            goal_reached_rate = goals_reached / self.num_eval_episodes if self.num_eval_episodes > 0 else 0
            
            # Brief console summary (detailed report will be saved to file)
            logger.info(f"Evaluation complete: {self.num_eval_episodes} episodes")
            logger.info(f"  Average Score: {avg_score:.2f}")
            logger.info(f"  Goal Reached Rate: {goal_reached_rate*100:.1f}%")
            logger.info(f"  Overall Driving Score: {performance_report['overall_driving_score']:.1f}/100")
            
            return avg_score, goal_reached_rate
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            return 0.0, 0.0
        finally:
            self.env.enable_pygame_display = original_pygame_state  # Restore pygame state

    def _calculate_performance_metrics(self, all_metrics, total_score, goals_reached, termination_reasons):
        """Calculate comprehensive performance metrics from detailed episode data."""
        num_episodes = len(all_metrics)
        
        if num_episodes == 0:
            return {}
            
        # Success Metrics
        goal_success_rate = (goals_reached / num_episodes) * 100
        collision_free_rate = (sum(1 for m in all_metrics if not m['collision_occurred']) / num_episodes) * 100
        sidewalk_free_rate = (sum(1 for m in all_metrics if m['sidewalk_violations'] == 0) / num_episodes) * 100
        rule_compliance_rate = (sum(1 for m in all_metrics if m['rule_violations'] == 0) / num_episodes) * 100
        
        # Efficiency Metrics
        successful_episodes = [m for m in all_metrics if m['goal_reached']]
        if successful_episodes:
            avg_time_to_goal = sum(m['time_to_goal_seconds'] for m in successful_episodes) / len(successful_episodes)
            avg_path_efficiency = sum(m['path_efficiency'] for m in successful_episodes) / len(successful_episodes)
        else:
            avg_time_to_goal = 0.0
            avg_path_efficiency = 0.0
            
        # Speed and Driving Quality
        all_speeds = [m['avg_speed_kmh'] for m in all_metrics if m['avg_speed_kmh'] > 0]
        avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0.0
        max_speed_achieved = max((m['max_speed_kmh'] for m in all_metrics), default=0.0)
        
        # Smooth driving
        all_smoothness = [m['smooth_driving_score'] for m in all_metrics]
        avg_smoothness = sum(all_smoothness) / len(all_smoothness) if all_smoothness else 0.0
        
        # Safety Metrics
        total_collisions = sum(1 for m in all_metrics if m['collision_occurred'])
        total_sidewalk_violations = sum(m['sidewalk_violations'] for m in all_metrics)
        total_rule_violations = sum(m['rule_violations'] for m in all_metrics)
        
        # Calculate overall driving score (0-100)
        driving_score = self._calculate_overall_driving_score(
            goal_success_rate, collision_free_rate, sidewalk_free_rate, 
            rule_compliance_rate, avg_smoothness, avg_path_efficiency
        )
        
        return {
            'num_episodes': num_episodes,
            'raw_avg_score': total_score / num_episodes,
            
            # Success Metrics (%)
            'goal_success_rate': goal_success_rate,
            'collision_free_rate': collision_free_rate,
            'sidewalk_free_rate': sidewalk_free_rate,
            'rule_compliance_rate': rule_compliance_rate,
            
            # Efficiency Metrics
            'avg_time_to_goal_seconds': avg_time_to_goal,
            'avg_path_efficiency': avg_path_efficiency,
            
            # Speed and Quality Metrics
            'avg_speed_kmh': avg_speed,
            'max_speed_achieved_kmh': max_speed_achieved,
            'avg_smoothness_score': avg_smoothness,
            
            # Safety Metrics
            'total_collisions': total_collisions,
            'total_sidewalk_violations': total_sidewalk_violations,
            'total_rule_violations': total_rule_violations,
            'violations_per_episode': total_rule_violations / num_episodes,
            
            # Overall Assessment
            'overall_driving_score': driving_score,
            'termination_breakdown': termination_reasons,
            
            # Performance Grade
            'performance_grade': self._get_performance_grade(driving_score)
        }

    def _calculate_overall_driving_score(self, goal_rate, collision_free_rate, sidewalk_free_rate, rule_compliance_rate, smoothness, path_efficiency):
        """Calculate a single 0-100 driving score based on multiple factors."""
        # Weight different aspects of driving performance
        weights = {
            'success': 0.30,      # Goal completion is important
            'safety': 0.40,       # Safety is most important
            'compliance': 0.15,   # Rule compliance
            'efficiency': 0.10,   # Path efficiency
            'smoothness': 0.05    # Driving smoothness
        }
        
        # Safety score (average of collision-free and sidewalk-free rates)
        safety_score = (collision_free_rate + sidewalk_free_rate) / 2
        
        # Efficiency score (inverse of path efficiency - lower is better)
        efficiency_score = max(0, 100 - ((path_efficiency - 1.0) * 50)) if path_efficiency > 0 else 0
        
        overall_score = (
            weights['success'] * goal_rate +
            weights['safety'] * safety_score +
            weights['compliance'] * rule_compliance_rate +
            weights['efficiency'] * efficiency_score +
            weights['smoothness'] * smoothness
        )
        
        return min(100, max(0, overall_score))

    def _get_performance_grade(self, score):
        """Convert numerical score to letter grade."""
        if score >= 90: return "A (Excellent)"
        elif score >= 80: return "B (Good)"
        elif score >= 70: return "C (Fair)"
        elif score >= 60: return "D (Poor)"
        else: return "F (Failing)"

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

    def _adaptive_curriculum_adjustment(self, performance_report: dict, i_episode: int):
        """Adaptively adjust curriculum based on agent performance."""
        if not hasattr(self.env, 'curriculum_manager') or not self.env.curriculum_manager:
            return
            
        driving_score = performance_report.get('overall_driving_score', 0.0)
        goal_rate = performance_report.get('goal_success_rate', 0.0)
        
        # Track performance trends
        self.curriculum_progress_tracker['performance_trends'].append({
            'episode': i_episode,
            'driving_score': driving_score,
            'goal_rate': goal_rate,
            'phase': self.env.curriculum_manager.current_phase_idx
        })
        
        # Calculate performance trend
        if len(self.curriculum_progress_tracker['performance_trends']) >= 5:
            recent_scores = [p['driving_score'] for p in list(self.curriculum_progress_tracker['performance_trends'])[-5:]]
            trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
            
            # Adaptive evaluation interval based on performance stability
            if abs(trend) < 2.0:  # Performance is stable
                self.adaptive_eval_interval = min(self.max_eval_interval, self.adaptive_eval_interval + 5)
            else:  # Performance is changing rapidly
                self.adaptive_eval_interval = max(self.min_eval_interval, self.adaptive_eval_interval - 5)
            
            # Log adaptive changes
            if self.adaptive_eval_interval != self.eval_interval:
                logger.info(f"Adaptive eval interval adjusted to {self.adaptive_eval_interval} (trend: {trend:.2f})")

    def _adaptive_epsilon_decay(self, performance_report: dict, current_epsilon: float) -> float:
        """Adjust epsilon decay based on recent performance."""
        if not self.adaptive_epsilon_enabled:
            return self._update_epsilon(current_epsilon)
            
        driving_score = performance_report.get('overall_driving_score', 0.0)
        self.performance_buffer.append(driving_score)
        
        if len(self.performance_buffer) >= 5:
            recent_avg = np.mean(list(self.performance_buffer))
            
            # Adjust epsilon decay based on performance
            if recent_avg > 70:  # Good performance - decay epsilon faster
                adjusted_decay = self.base_epsilon_decay * 0.98
            elif recent_avg < 30:  # Poor performance - decay epsilon slower
                adjusted_decay = self.base_epsilon_decay * 1.02
            else:
                adjusted_decay = self.base_epsilon_decay
                
            # Apply adjusted decay
            new_epsilon = max(self.epsilon_end, adjusted_decay * current_epsilon)
            
            # Log significant changes
            if abs(adjusted_decay - self.base_epsilon_decay) > 0.001:
                logger.debug(f"Adaptive epsilon decay: {adjusted_decay:.4f} (performance: {recent_avg:.1f})")
                
            return new_epsilon
        
        return self._update_epsilon(current_epsilon)

    def _check_early_stopping(self, current_score: float, i_episode: int) -> bool:
        """Check if training should stop early based on performance plateau."""
        if not self.early_stopping_enabled:
            return False
            
        self.best_score_history.append(current_score)
        
        # Need sufficient history to make decision
        if len(self.best_score_history) < self.patience:
            return False
            
        # Check if recent performance shows improvement
        recent_best = max(list(self.best_score_history)[-10:])  # Best in last 10 evaluations
        historical_best = max(list(self.best_score_history)[:-10])  # Best before recent 10
        
        improvement = recent_best - historical_best
        
        if improvement < self.min_improvement:
            self.no_improvement_count += 1
            if self.no_improvement_count >= 3:  # 3 consecutive evaluations without improvement
                logger.info(f"Early stopping triggered at episode {i_episode}")
                logger.info(f"No significant improvement ({improvement:.3f} < {self.min_improvement}) for {self.no_improvement_count} evaluations")
                return True
        else:
            self.no_improvement_count = 0
            
        return False

    def _monitor_training_stability(self, episode_score: float, avg_loss: Optional[float]):
        """Monitor training stability and detect potential issues."""
        # Track reward variance
        self.stability_metrics['reward_variance_window'].append(episode_score)
        
        if avg_loss is not None:
            self.stability_metrics['loss_variance_window'].append(avg_loss)
            
            # Check for gradient explosion (very high loss)
            if avg_loss > 100:  # Configurable threshold
                self.stability_metrics['gradient_explosion_count'] += 1
                if self.stability_metrics['gradient_explosion_count'] > 5:
                    logger.warning("Potential gradient explosion detected - consider reducing learning rate")
        
        # Check for performance plateau
        if len(self.stability_metrics['reward_variance_window']) >= 50:
            recent_variance = np.var(list(self.stability_metrics['reward_variance_window'])[-20:])
            if recent_variance < 0.1:  # Very low variance indicates plateau
                self.stability_metrics['performance_plateau_count'] += 1
                if self.stability_metrics['performance_plateau_count'] % 10 == 0:
                    logger.info(f"Performance plateau detected - variance: {recent_variance:.4f}")

    def _log_training_analytics(self, i_episode: int, episode_score: float, steps_taken: int, 
                               termination_reason: str, current_epsilon: float):
        """Log advanced training analytics."""
        # Episode efficiency (lower is better)
        if termination_reason in ["goal_reached", "goal_reached_and_stopped"]:
            efficiency = steps_taken / 1000.0  # Normalized by max steps
            self.performance_analytics['episode_efficiency'].append(efficiency)
        
        # Exploration efficiency
        if hasattr(self.agent, 'training_metrics'):
            exploration_ratio = current_epsilon * (1.0 - episode_score / 100.0)  # Penalty for poor performance with high exploration
            self.performance_analytics['exploration_efficiency'].append(exploration_ratio)
        
        # Log to TensorBoard every 10 episodes
        if i_episode % 10 == 0:
            # Training stability metrics
            if self.stability_metrics['reward_variance_window']:
                reward_variance = np.var(list(self.stability_metrics['reward_variance_window']))
                self.tb_logger.log_scalar("stability/reward_variance", reward_variance, i_episode)
            
            if self.stability_metrics['loss_variance_window']:
                loss_variance = np.var(list(self.stability_metrics['loss_variance_window']))
                self.tb_logger.log_scalar("stability/loss_variance", loss_variance, i_episode)
            
            # Performance analytics
            if self.performance_analytics['episode_efficiency']:
                avg_efficiency = np.mean(self.performance_analytics['episode_efficiency'][-20:])
                self.tb_logger.log_scalar("analytics/episode_efficiency", avg_efficiency, i_episode)
            
            if self.performance_analytics['exploration_efficiency']:
                avg_exploration_eff = np.mean(self.performance_analytics['exploration_efficiency'][-20:])
                self.tb_logger.log_scalar("analytics/exploration_efficiency", avg_exploration_eff, i_episode)
            
            # Agent-specific metrics
            if hasattr(self.agent, 'training_metrics'):
                if self.agent.training_metrics['gradient_norms']:
                    avg_grad_norm = np.mean(self.agent.training_metrics['gradient_norms'][-20:])
                    self.tb_logger.log_scalar("training/avg_gradient_norm", avg_grad_norm, i_episode)
                
                if self.agent.training_metrics['learning_rates']:
                    current_lr = self.agent.training_metrics['learning_rates'][-1]
                    self.tb_logger.log_scalar("training/learning_rate", current_lr, i_episode) 