import torch
import numpy as np
from collections import deque
import os
import logging
from typing import Tuple, Optional, List, Dict, Any
import time
from datetime import datetime
import copy

# Assuming your project structure allows these imports from src/
# If run with `python -m src.main`, then these should work.
# from ..agents.dqn_agent import DQNAgent # Would be used if passed as type hint
# from ..environments.carla_env import CarlaEnv
# from ..utils.logger import Logger

logger = logging.getLogger(__name__) # Logger for this module

# Get project root directory for absolute path construction
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels from app/training/

class DQNTrainer:
    """
    Enhanced DQN trainer with curriculum learning, performance analytics, and robust model selection.
    
    BEST MODEL SELECTION SYSTEM:
    ============================
    
    This trainer uses 'overall_driving_score' (0-100 scale) for determining the best model,
    instead of phase-specific raw scores that can vary dramatically between curriculum phases.
    
    Overall Driving Score Components:
    - Success (30%): Goal completion rate
    - Safety (40%): Collision-free and sidewalk-free driving  
    - Compliance (15%): Traffic rule adherence
    - Efficiency (10%): Path optimization
    - Smoothness (5%): Driving quality
    
    Advantages of this approach:
    - Phase-agnostic: Consistent metric across all curriculum phases
    - Normalized: Always 0-100 scale for easy interpretation
    - Comprehensive: Balances multiple aspects of driving performance
    - Robust: Less susceptible to phase-specific scoring anomalies
    
    Legacy Compatibility:
    - Automatically detects and converts old raw score systems (>100)
    - Maintains backward compatibility with existing models
    - Clear logging distinguishes between score types
    
    The trainer continues to track raw avg_score for compatibility and analysis,
    but model selection is based solely on the standardized driving score.
    """
    
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
        self.max_steps_per_episode = getattr(args, 'max_steps_per_episode', 1000)
        
        # Check if curriculum learning is enabled
        self.use_curriculum = (hasattr(env, 'curriculum_manager') and 
                             env.curriculum_manager and 
                             env.curriculum_manager.phases)
        
        if self.use_curriculum:
            # Calculate total episodes from all curriculum phases
            total_curriculum_episodes = sum(phase.get('episodes', 0) for phase in env.curriculum_manager.phases)
            # Add buffer for phase repetitions and evaluation episodes
            max_repeats = getattr(env.curriculum_manager, 'max_phase_repeats', 3)
            eval_episodes_per_phase = getattr(env.curriculum_manager, 'evaluation_episodes', 5)
            
            # More conservative estimate: base episodes + potential repeats + evaluation overhead
            repeat_buffer = total_curriculum_episodes * (max_repeats - 1)  # Additional repeats beyond first attempt
            eval_buffer = len(env.curriculum_manager.phases) * eval_episodes_per_phase * max_repeats
            self.curriculum_total_episodes = total_curriculum_episodes + repeat_buffer + eval_buffer
            
            # Set num_episodes to be controlled by curriculum
            self.num_episodes = None  # Will be controlled by curriculum completion
            
            # Log detailed phase information
            logger.info(f"Curriculum learning enabled: {len(env.curriculum_manager.phases)} phases")
            logger.info(f"Total base episodes: {total_curriculum_episodes}")
            logger.info(f"Max with repeats and evaluation: {self.curriculum_total_episodes}")
            logger.info("Phase breakdown:")
            for i, phase in enumerate(env.curriculum_manager.phases, 1):
                logger.info(f"  Phase {i}: {phase.get('name', 'Unknown')} - {phase.get('episodes', 0)} episodes")
        else:
            # Fallback for non-curriculum training
            self.num_episodes = getattr(args, 'num_episodes', 1000)
            self.curriculum_total_episodes = None
            logger.warning("No curriculum manager found. Using fallback num_episodes from args.")
        self.epsilon_start = getattr(args, 'epsilon_start', 1.0)
        self.epsilon_end = getattr(args, 'epsilon_end', 0.01)
        self.epsilon_decay = getattr(args, 'epsilon_decay', 0.995)
        
        # Evaluation parameters from args
        self.eval_interval = getattr(args, 'eval_interval', 25)
        self.num_eval_episodes = getattr(args, 'num_eval_episodes', 5)
        self.epsilon_eval = getattr(args, 'epsilon_eval', 0.01)

        # Comprehensive evaluation configuration
        self.comprehensive_eval_interval = getattr(args, 'comprehensive_eval_interval', 100)  # Every 100 episodes
        self.comprehensive_eval_min_phase = getattr(args, 'comprehensive_eval_min_phase', 3)  # Start after phase 3
        self.use_comprehensive_for_best_model = getattr(args, 'use_comprehensive_for_best_model', True)

        # Saving parameters from args
        self.save_interval = getattr(args, 'save_interval', 50)
        self.model_base_save_dir = args.save_dir 
        self.current_best_model_dir = os.path.join(self.model_base_save_dir, "best_model")

        self.start_episode = 1
        self.best_eval_score = -float('inf')  # Now tracks comprehensive driving score (0-100 scale)
        self.best_phase_score = -float('inf')  # Track best single-phase score separately
        self.last_eval_score = -float('inf')  # Still tracks raw avg_score for compatibility
        self.last_comprehensive_score = -float('inf')  # Track last comprehensive evaluation
        
        # Create training run reports directory (using absolute paths)
        self.training_start_time = datetime.now()
        self.run_id = f"training_run_{self.training_start_time.strftime('%Y%m%d_%H%M%S')}"
        self.reports_base_dir = os.path.join(project_root, "reports")
        self.reports_dir = os.path.join(self.reports_base_dir, "training_runs", self.run_id)
        self.best_model_reports_dir = os.path.join(self.reports_base_dir, "best_model_reports")
        
        # Create all reports directories
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.best_model_reports_dir, exist_ok=True)
        
        # Log training run info
        run_info_path = os.path.join(self.reports_dir, "training_info.txt")
        with open(run_info_path, "w") as f:
            f.write(f"Training Run: {self.run_id}\n")
            f.write(f"Started: {self.training_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if self.use_curriculum:
                f.write(f"Training Mode: Curriculum-based (episodes determined by phases)\n")
                f.write(f"Estimated Max Episodes: {self.curriculum_total_episodes}\n")
            else:
                f.write(f"Total Episodes: {self.num_episodes}\n")
            f.write(f"Model Save Dir: {self.model_base_save_dir}\n")
            f.write(f"Best Model Selection: Based on overall_driving_score (0-100 scale)\n")
            f.write("="*50 + "\n\n")
        
        logger.info(f"Training reports will be saved to: {os.path.abspath(self.reports_dir)}")
        
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
                        loaded_score = float(f.read().strip())
                        # Handle transition from old score system to driving score system
                        if loaded_score > 100:
                            # Old system used raw scores (could be > 100), convert to driving score estimate
                            # Use a conservative estimate: cap at 100 and scale down
                            estimated_driving_score = min(100, loaded_score * 0.5)  # Conservative scaling
                            logger.info(f"Loaded legacy score {loaded_score:.2f}, estimating driving score: {estimated_driving_score:.2f}")
                            self.best_eval_score = estimated_driving_score
                        else:
                            # New system uses driving scores (0-100 scale)
                            self.best_eval_score = loaded_score
                            logger.info(f"Loaded best driving score from {score_to_load}: {self.best_eval_score:.2f}")
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
        
        # Robust loss tracking
        self._loss_ema = None  # Exponential moving average of losses
        self._loss_trend_buffer = deque(maxlen=20)  # For trend analysis
        self._performance_confidence_buffer = deque(maxlen=10)  # For confidence intervals

        self._early_stopping_no_improvement_count = 0
        self._last_improvement_episode = 0
        
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

    def _calculate_robust_episode_loss(self, episode_losses: List[float]) -> float:
        """
        Calculate robust episode loss with outlier handling and stability measures.
        
        Args:
            episode_losses: List of loss values from the episode
            
        Returns:
            Robust average loss value
        """
        if not episode_losses:
            return None
            
        # Convert to numpy array and handle NaN/inf values
        losses = np.array(episode_losses, dtype=np.float32)
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(losses)
        if not np.any(valid_mask):
            logger.warning("All episode losses are NaN or infinite, returning None")
            return None
            
        valid_losses = losses[valid_mask]
        
        if len(valid_losses) == 1:
            return float(valid_losses[0])
            
        # Detect and handle outliers using IQR method
        q75, q25 = np.percentile(valid_losses, [75, 25])
        iqr = q75 - q25
        
        if iqr > 0:  # Only apply outlier removal if there's variance
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            outlier_mask = (valid_losses >= lower_bound) & (valid_losses <= upper_bound)
            
            if np.any(outlier_mask):
                filtered_losses = valid_losses[outlier_mask]
                outlier_count = len(valid_losses) - len(filtered_losses)
                if outlier_count > 0:
                    logger.debug(f"Filtered {outlier_count} outlier loss values from episode")
            else:
                # If all values are outliers, use median instead
                filtered_losses = valid_losses
                logger.debug("All loss values detected as outliers, using all values")
        else:
            filtered_losses = valid_losses
            
        # Use robust averaging method
        if len(filtered_losses) <= 3:
            # For small number of losses, use simple average
            robust_loss = np.mean(filtered_losses)
        else:
            # For larger sets, use combination of median and mean for robustness
            median_loss = np.median(filtered_losses)
            mean_loss = np.mean(filtered_losses)
            
            # Weight median more heavily if there's high variance (less stable)
            loss_std = np.std(filtered_losses)
            mean_loss_val = np.mean(filtered_losses)
            cv = loss_std / (mean_loss_val + 1e-8)  # Coefficient of variation
            
            if cv > 0.5:  # High variance - trust median more
                median_weight = 0.7
            elif cv > 0.2:  # Medium variance - balanced approach
                median_weight = 0.5
            else:  # Low variance - trust mean more
                median_weight = 0.3
                
            robust_loss = median_weight * median_loss + (1 - median_weight) * mean_loss
            
        # Apply exponential smoothing if we have a history
        if hasattr(self, '_loss_ema') and self._loss_ema is not None:
            # Exponential moving average with adaptive smoothing factor
            smoothing_factor = min(0.3, max(0.05, 0.3 / (len(filtered_losses) + 1)))
            robust_loss = smoothing_factor * robust_loss + (1 - smoothing_factor) * self._loss_ema
            
        # Update EMA for next calculation
        self._loss_ema = robust_loss
        
        return float(robust_loss)

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics for monitoring and analysis."""
        stats = {
            'loss_tracking': {
                'current_ema': self._loss_ema,
                'trend_buffer_size': len(self._loss_trend_buffer),
                'has_stable_losses': self._loss_ema is not None and len(self._loss_trend_buffer) > 10
            },
            'performance_tracking': {
                'confidence_buffer_size': len(self._performance_confidence_buffer),
                'performance_buffer_size': len(self.performance_buffer),
                'has_reliable_performance_data': len(self.performance_buffer) >= 5
            },
            'curriculum_progress': {
                'total_trends_tracked': len(self.curriculum_progress_tracker['performance_trends']),
                'difficulty_adjustments_made': len(self.curriculum_progress_tracker['difficulty_adjustments']),
                'adaptive_threshold_adjustments': self.curriculum_progress_tracker['adaptive_threshold_adjustments'],
                'current_eval_interval': getattr(self, 'adaptive_eval_interval', self.eval_interval)
            },
            'stability_metrics': {
                'reward_variance_tracking': len(self.stability_metrics['reward_variance_window']),
                'loss_variance_tracking': len(self.stability_metrics['loss_variance_window']),
                'gradient_explosion_count': self.stability_metrics['gradient_explosion_count'],
                'performance_plateau_count': self.stability_metrics['performance_plateau_count']
            }
        }
        
        # Add recent performance confidence interval if available
        if len(self._performance_confidence_buffer) >= 3:
            recent_scores = [entry['composite'] for entry in list(self._performance_confidence_buffer)[-5:]]
            if recent_scores:
                stats['recent_performance'] = {
                    'mean': np.mean(recent_scores),
                    'std': np.std(recent_scores),
                    'min': np.min(recent_scores),
                    'max': np.max(recent_scores),
                    'trend': np.polyfit(range(len(recent_scores)), recent_scores, 1)[0] if len(set(recent_scores)) > 1 else 0
                }
        
        return stats

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
                
                # Only append loss when agent actually performs a learning step
                # This prevents artificial loss averaging from None values
                if current_loss is not None:
                    episode_losses.append(current_loss)

                # Update state and score
                state = next_state
                episode_score += reward

                if done:
                    if terminated:
                        termination_reason = info.get("termination_reason", "terminated_by_env")
                    break

            # Calculate final metrics - only average actual loss values
            steps_taken = t_step + 1
            # Calculate robust episode loss metrics with outlier handling and stability measures
            if episode_losses:
                avg_episode_loss = self._calculate_robust_episode_loss(episode_losses)
            else:
                avg_episode_loss = None
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
        """Enhanced evaluation and saving with comprehensive multi-phase evaluation for best model selection."""
        # Check if curriculum phase evaluation should be triggered
        should_eval_phase = (hasattr(self.env, 'curriculum_manager') and 
                            self.env.curriculum_manager and 
                            self.env.curriculum_manager.should_evaluate_phase())
        
        # Regular evaluation interval or phase evaluation
        should_eval_regular = i_episode % self.eval_interval == 0
        
        # Check if comprehensive evaluation should be triggered
        should_eval_comprehensive = (
            self.use_comprehensive_for_best_model and
            self.use_curriculum and
            i_episode % self.comprehensive_eval_interval == 0 and
            hasattr(self.env, 'curriculum_manager') and
            self.env.curriculum_manager and
            self.env.curriculum_manager.get_current_phase_number() >= self.comprehensive_eval_min_phase
        )
        
        # Update agent with curriculum information if available
        if hasattr(self.agent, 'update_curriculum_phase') and hasattr(self.env, 'curriculum_manager'):
            if self.env.curriculum_manager:
                phase_info = {
                    'phase_number': getattr(self.env.curriculum_manager, 'current_phase_index', 0),
                    'difficulty': self.env.curriculum_manager.get_current_phase_difficulty() if hasattr(self.env.curriculum_manager, 'get_current_phase_difficulty') else 1.0
                }
                self.agent.update_curriculum_phase(phase_info)
        
        if should_eval_regular or should_eval_phase or should_eval_comprehensive:
            eval_type = "regular"
            if should_eval_comprehensive:
                logger.info(f"--- Starting Comprehensive Multi-Phase Evaluation after Episode {i_episode} ---")
                eval_type = "comprehensive"
                eval_episodes = min(3, self.env.curriculum_manager.evaluation_episodes) if hasattr(self.env, 'curriculum_manager') else 3
            elif should_eval_phase:
                logger.info(f"--- Starting Automatic Phase Evaluation after Episode {i_episode} ---")
                eval_type = "phase"
                eval_episodes = self.env.curriculum_manager.evaluation_episodes
            else:
                logger.info(f"--- Starting Regular Evaluation after Episode {i_episode} ---")
                eval_episodes = self.num_eval_episodes
            
            # Temporarily override num_eval_episodes for this evaluation
            original_eval_episodes = self.num_eval_episodes
            self.num_eval_episodes = eval_episodes
            
            try:
                # Run the appropriate type of evaluation
                if eval_type == "comprehensive":
                    # Run comprehensive evaluation across all phases
                    comprehensive_score = self._comprehensive_model_evaluation(i_episode)
                    self.last_comprehensive_score = comprehensive_score
                    
                    # Also run a regular evaluation for current phase metrics (for logging)
                    avg_score, goal_rate = self.evaluate_agent()
                    self.last_eval_score = avg_score
                    
                    # Use comprehensive score as the primary evaluation metric
                    driving_score = comprehensive_score
                    current_score_for_best_model = comprehensive_score
                    
                    # Create synthetic report for consistent logging
                    if hasattr(self, '_last_performance_report') and self._last_performance_report:
                        # Update the report to reflect comprehensive evaluation
                        self._last_performance_report['evaluation_type'] = 'comprehensive'
                        self._last_performance_report['comprehensive_score'] = comprehensive_score
                        self._last_performance_report['note'] = f'Comprehensive evaluation across {len(self.env.curriculum_manager.phases)} phases'
                    
                    logger.info(f"Comprehensive evaluation: {comprehensive_score:.1f}/100 (across all {len(self.env.curriculum_manager.phases)} phases)")
                    
                else:
                    # Run standard evaluation (single phase)
                    avg_score, goal_rate = self.evaluate_agent()
                    self.last_eval_score = avg_score
                    
                    # For best model selection, use different criteria based on settings
                    if self.use_comprehensive_for_best_model and eval_type == "phase":
                        # For phase evaluations, track phase score but don't use for best model
                        current_score_for_best_model = None  # Don't update best model on phase evaluations
                        if hasattr(self, '_last_performance_report') and self._last_performance_report:
                            phase_driving_score = self._last_performance_report.get('overall_driving_score', 0.0)
                            self.best_phase_score = max(self.best_phase_score, phase_driving_score)
                    else:
                        # Use single-phase score if comprehensive evaluation is disabled
                        if hasattr(self, '_last_performance_report') and self._last_performance_report:
                            current_score_for_best_model = self._last_performance_report.get('overall_driving_score', avg_score)
                        else:
                            current_score_for_best_model = avg_score
                
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
                if eval_type == "comprehensive":
                    driving_score = comprehensive_score
                    grade = self._get_performance_grade(comprehensive_score)
                else:
                    driving_score = report.get('overall_driving_score', 0.0)
                    grade = report.get('performance_grade', 'N/A')
                    
                goal_rate_pct = report.get('goal_success_rate', 0.0)
                collision_free_pct = report.get('collision_free_rate', 0.0)
                rule_compliance_pct = report.get('rule_compliance_rate', 0.0)
                
                report_filename = f"episode_{i_episode:03d}_{eval_type}_evaluation.txt"
                report_path = os.path.join(self.reports_dir, report_filename)
                
                eval_type_display = "Comprehensive Multi-Phase" if eval_type == "comprehensive" else eval_type.title()
                logger.info(f"Episode {i_episode}: {eval_type_display} Evaluation - Driving Score: {driving_score:.1f}/100 ({grade})")
                logger.info(f"  Goal Rate: {goal_rate_pct:.1f}% | Collision-Free: {collision_free_pct:.1f}% | Rule Compliance: {rule_compliance_pct:.1f}%")
                logger.info(f"  Detailed report saved to: {os.path.abspath(report_path)}")
                
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
                    
                    # Log best model tracking metrics
                    if hasattr(self, '_last_performance_report') and self._last_performance_report:
                        self.tb_logger.log_scalar("evaluation/current_driving_score", driving_score, i_episode)
                        self.tb_logger.log_scalar("evaluation/best_driving_score", self.best_eval_score, i_episode)
                    else:
                        self.tb_logger.log_scalar("evaluation/best_driving_score", self.best_eval_score, i_episode)
                    
                    # Log enhanced agent analytics if available
                    if hasattr(self.agent, 'get_training_analytics'):
                        analytics = self.agent.get_training_analytics()
                        
                        # Log loss and stability metrics
                        if 'avg_recent_loss' in analytics:
                            self.tb_logger.log_scalar("training/avg_recent_loss", analytics['avg_recent_loss'], i_episode)
                        if 'loss_std' in analytics:
                            self.tb_logger.log_scalar("training/loss_stability", 1.0 / (1.0 + analytics['loss_std']), i_episode)
                        if 'loss_trend' in analytics:
                            self.tb_logger.log_scalar("training/loss_trend", analytics['loss_trend'], i_episode)
                        
                        # Log improved loss metrics
                        if 'min_recent_loss' in analytics:
                            self.tb_logger.log_scalar("training/min_recent_loss", analytics['min_recent_loss'], i_episode)
                        if 'max_recent_loss' in analytics:
                            self.tb_logger.log_scalar("training/max_recent_loss", analytics['max_recent_loss'], i_episode)
                        if 'total_training_updates' in analytics:
                            self.tb_logger.log_scalar("training/total_training_updates", analytics['total_training_updates'], i_episode)
                        
                        # Log learning frequency metrics
                        if 'learning_frequency' in analytics:
                            self.tb_logger.log_scalar("training/learning_frequency", analytics['learning_frequency'], i_episode)
                        if 'learning_steps' in analytics:
                            self.tb_logger.log_scalar("training/learning_steps", analytics['learning_steps'], i_episode)
                            
                        # Log exploration metrics
                        if 'avg_exploration_rate' in analytics:
                            self.tb_logger.log_scalar("exploration/avg_rate", analytics['avg_exploration_rate'], i_episode)
                        if 'exploration_efficiency' in analytics:
                            self.tb_logger.log_scalar("exploration/efficiency", analytics['exploration_efficiency'], i_episode)
                        if 'action_entropy' in analytics:
                            self.tb_logger.log_scalar("exploration/action_entropy", analytics['action_entropy'], i_episode)
                            
                        # Log Q-value and gradient metrics
                        if 'avg_q_value' in analytics:
                            self.tb_logger.log_scalar("model/avg_q_value", analytics['avg_q_value'], i_episode)
                        if 'q_value_std' in analytics:
                            self.tb_logger.log_scalar("model/q_value_std", analytics['q_value_std'], i_episode)
                        if 'avg_grad_norm' in analytics:
                            self.tb_logger.log_scalar("training/avg_grad_norm", analytics['avg_grad_norm'], i_episode)
                        if 'grad_stability' in analytics:
                            self.tb_logger.log_scalar("training/grad_stability", analytics['grad_stability'], i_episode)
                    
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

                    # Enhanced best model selection: use comprehensive score when available, otherwise use driving score
                    if 'current_score_for_best_model' in locals() and current_score_for_best_model is not None:
                        current_score_for_evaluation = current_score_for_best_model
                        selection_method = f"{eval_type}_comprehensive" if eval_type == "comprehensive" else eval_type
                    else:
                        current_score_for_evaluation = driving_score if hasattr(self, '_last_performance_report') and self._last_performance_report else avg_score
                        selection_method = f"{eval_type}_fallback"
                    
                    if current_score_for_evaluation > self.best_eval_score:
                        logger.info(f"New best driving score: {current_score_for_evaluation:.2f} (previous: {self.best_eval_score:.2f})")
                        if hasattr(self, '_last_performance_report') and self._last_performance_report:
                            logger.info(f"  Raw Score: {avg_score:.2f} | Goal Rate: {goal_rate_pct:.1f}% | Safety: {(collision_free_pct + report.get('sidewalk_free_rate', 0.0))/2:.1f}%")
                        self.best_eval_score = current_score_for_evaluation
                        
                        os.makedirs(self.current_best_model_dir, exist_ok=True)
                        self.agent.save(self.current_best_model_dir, model_name="best_dqn_agent")
                        
                        # Save the best score to a file directly in the model_base_save_dir
                        with open(os.path.join(self.model_base_save_dir, "best_score.txt"), "w") as f:
                            f.write(str(self.best_eval_score))
                        # Also save it inside the best_model directory for redundancy/clarity
                        with open(os.path.join(self.current_best_model_dir, "best_score.txt"), "w") as f:
                            f.write(str(self.best_eval_score))
                            
                        # Save metadata about evaluation type
                        with open(os.path.join(self.current_best_model_dir, "evaluation_metadata.txt"), "w") as f:
                            f.write(f"Evaluation Type: {eval_type}\n")
                            f.write(f"Episode: {i_episode}\n")
                            f.write(f"Score: {self.best_eval_score:.2f}/100\n")
                            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            if eval_type == "phase":
                                f.write(f"Note: This model was selected based on current phase performance\n")
                            else:
                                f.write(f"Note: This model was selected based on comprehensive evaluation across all curriculum phases\n")
                            
                        # Save comprehensive metrics for the best model
                        if hasattr(self, '_last_performance_report') and self._last_performance_report:
                            best_model_report_path = os.path.join(self.best_model_reports_dir, f"best_score_{current_score_for_evaluation:.1f}_episode_{i_episode:03d}_{eval_type}.txt")
                            with open(best_model_report_path, "w") as f:
                                self._write_comprehensive_report(f, self._last_performance_report, f"BEST MODEL PERFORMANCE REPORT (Episode {i_episode})")
                                f.write(f"\nBEST MODEL SELECTION CRITERIA:\n")
                                f.write(f"   Selection Metric: {eval_type.title()} Driving Score (0-100 scale)\n")
                                if eval_type == "phase":
                                    f.write(f"   Evaluation: Single-phase performance (current phase only)\n")
                                    f.write(f"   Advantages: Phase-agnostic, normalized, comprehensive within phase\n")
                                else:
                                    f.write(f"   Evaluation: Tested across ALL {len(self.env.curriculum_manager.phases)} curriculum phases\n")
                                    f.write(f"   Advantages: Comprehensive, phase-weighted, robust performance assessment\n")
                                f.write(f"   Components: Success(30%) + Safety(40%) + Compliance(15%) + Efficiency(10%) + Smoothness(5%)\n")
                                f.write(f"   Current Best Score: {self.best_eval_score:.2f}/100\n")
                            
                            # Also save in the model directory for backward compatibility
                            with open(os.path.join(self.current_best_model_dir, "performance_report.txt"), "w") as f:
                                self._write_comprehensive_report(f, self._last_performance_report, f"BEST MODEL PERFORMANCE REPORT (Episode {i_episode})")
                                f.write(f"\nModel selected based on {eval_type} driving score: {current_score_for_evaluation:.2f}/100\n")
                
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
                f.write(f"\nNote: Checkpoint score is raw avg_score, not driving score")
                
            logger.info(f"Saved checkpoint at episode {i_episode} with raw score: {self.last_eval_score:.2f}")

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

    def _get_performance_grade(self, score: float) -> str:
        """Get performance grade based on driving score."""
        if score >= 85:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 65:
            return "C"
        elif score >= 55:
            return "D"
        else:
            return "F"

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
            f.write(f"\nNote: Final model score is raw avg_score, not driving score")
            
        logger.info(f"Saved final model to {final_model_dir} with raw score: {self.last_eval_score:.2f}")
        logger.info(f"Best driving score achieved during training: {self.best_eval_score:.2f}/100")

    def train_loop(self):
        """Main training loop that respects curriculum phases or runs for specified episodes."""
        # Training is controlled by curriculum phases when available
        if self.use_curriculum:
            logger.info(f"Starting curriculum-based training from episode {self.start_episode}.")
            logger.info(f"Training will run until all curriculum phases are completed")
        else:
            logger.info(f"Starting training loop for {self.num_episodes} episodes from episode {self.start_episode}.")
        
        scores_deque = deque(maxlen=100)  # For rolling average score
        current_epsilon = self.epsilon_start  # Epsilon for the upcoming episode
        i_episode = self.start_episode

        try:
            # Log initial curriculum state
            if self.use_curriculum and hasattr(self.env, 'curriculum_manager') and self.env.curriculum_manager:
                phase_name, _, _ = self.env.curriculum_manager.get_current_phase_details()
                current_phase_num = self.env.curriculum_manager.get_current_phase_number()
                total_phases = len(self.env.curriculum_manager.phases)
                logger.info(f"Starting with Phase {current_phase_num}/{total_phases}: {phase_name}")
            
            while True:
                # Check curriculum completion conditions
                if self.use_curriculum:
                    # Check if we're on the last phase AND have completed all its episodes AND evaluation
                    if (hasattr(self.env, 'curriculum_manager') and 
                        self.env.curriculum_manager and
                        self.env.curriculum_manager.current_phase_idx >= len(self.env.curriculum_manager.phases) - 1):
                        
                        current_phase_cfg = self.env.curriculum_manager.get_current_phase_config()
                        if (current_phase_cfg and 
                            self.env.curriculum_manager.episode_in_current_phase > current_phase_cfg["episodes"]):
                            
                            # Only stop if evaluation is disabled OR we've completed evaluation for the last phase
                            if (not self.env.curriculum_manager.evaluation_enabled or 
                                not self.env.curriculum_manager.should_evaluate_phase()):
                                logger.info("All curriculum phases completed successfully!")
                                logger.info(f"Total curriculum episodes completed: {self.env.curriculum_manager.total_episodes_tracked}")
                                logger.info(f"Final phase: {current_phase_cfg['name']}")
                                break
                    
                    # Safety check against runaway episodes - only warn but don't stop
                    if i_episode > self.curriculum_total_episodes:
                        if i_episode % 100 == 0:  # Log warning every 100 episodes
                            logger.warning(f"Episode {i_episode} exceeds estimated curriculum limit ({self.curriculum_total_episodes}). This may indicate phase repetitions or evaluation overhead.")
                else:
                    # Traditional episode limit check
                    if i_episode > self.num_episodes:
                        break
                # Run episode with current epsilon
                episode_score, avg_episode_loss, steps_taken, termination_reason, runtime_error = \
                    self._run_episode(i_episode, current_epsilon)

                if runtime_error:
                    logger.warning(f"Episode {i_episode} ended due to a runtime error. Stopping training loop.")
                    break 

                # Update statistics
                scores_deque.append(episode_score)
                avg_score_deque = np.mean(scores_deque)
                
                # Log episode summary with curriculum information
                if self.use_curriculum and hasattr(self.env, 'curriculum_manager') and self.env.curriculum_manager:
                    phase_name, ep_in_phase, total_eps_in_phase = self.env.curriculum_manager.get_current_phase_details()
                    current_phase_num = self.env.curriculum_manager.get_current_phase_number()
                    total_phases = len(self.env.curriculum_manager.phases)
                    display_total = f"Phase {current_phase_num}/{total_phases} ({ep_in_phase}/{total_eps_in_phase})"
                else:
                    display_total = self.num_episodes
                    
                self._log_episode_summary(
                    i_episode, display_total, episode_score, avg_score_deque,
                    current_epsilon, steps_taken, avg_episode_loss, termination_reason
                )
                
                # Update epsilon for next episode with adaptive decay if performance data is available
                if hasattr(self, '_last_performance_report') and self._last_performance_report:
                    current_epsilon = self._adaptive_epsilon_decay(self._last_performance_report, current_epsilon)
                else:
                    current_epsilon = self._update_epsilon(current_epsilon)

                # Handle evaluation and model saving
                self._handle_evaluation_and_saving(i_episode)
                
                # Apply robust training adaptations if we have recent performance data
                if hasattr(self, '_last_performance_report') and self._last_performance_report:
                    self._adaptive_curriculum_adjustment(self._last_performance_report, i_episode)
                
                # Monitor training stability
                self._monitor_training_stability(episode_score, avg_episode_loss)
                
                # Check early stopping conditions (only for non-curriculum training)
                if not self.use_curriculum and hasattr(self, '_check_early_stopping'):
                    if self._check_early_stopping(avg_score_deque, i_episode):
                        logger.info(f"Early stopping triggered at episode {i_episode}")
                        break
                
                # Increment episode counter for next iteration
                i_episode += 1

        except KeyboardInterrupt:
            logger.info(f"Training loop interrupted by user at episode {i_episode}.")
        finally:
            self._save_final_model()
            logger.info("Training loop finished.") 

    def _adaptive_curriculum_adjustment(self, performance_report: dict, i_episode: int):
        """Robust adaptive curriculum adjustments based on comprehensive performance analysis."""
        if not hasattr(self.env, 'curriculum_manager') or not self.env.curriculum_manager:
            return
            
        # Collect comprehensive performance metrics
        driving_score = performance_report.get('overall_driving_score', 0.0)
        goal_rate = performance_report.get('goal_success_rate', 0.0)
        safety_score = (performance_report.get('collision_free_rate', 0.0) + 
                       performance_report.get('sidewalk_free_rate', 0.0)) / 2
        
        current_phase = self.env.curriculum_manager.current_phase_idx
        
        # Track performance trends with enhanced metrics
        performance_entry = {
            'episode': i_episode,
            'driving_score': driving_score,
            'goal_rate': goal_rate,
            'safety_score': safety_score,
            'phase': current_phase,
            'composite_score': 0.4 * driving_score + 0.3 * goal_rate + 0.3 * safety_score
        }
        
        self.curriculum_progress_tracker['performance_trends'].append(performance_entry)
        
        # Robust trend analysis with sufficient data
        if len(self.curriculum_progress_tracker['performance_trends']) >= 8:
            recent_entries = list(self.curriculum_progress_tracker['performance_trends'])[-8:]
            
            # Calculate multiple trend indicators
            composite_scores = [p['composite_score'] for p in recent_entries]
            driving_scores = [p['driving_score'] for p in recent_entries]
            goal_rates = [p['goal_rate'] for p in recent_entries]
            
            # Robust trend calculation using linear regression
            x = np.arange(len(composite_scores))
            composite_trend = np.polyfit(x, composite_scores, 1)[0] if len(set(composite_scores)) > 1 else 0
            driving_trend = np.polyfit(x, driving_scores, 1)[0] if len(set(driving_scores)) > 1 else 0
            goal_trend = np.polyfit(x, goal_rates, 1)[0] if len(set(goal_rates)) > 1 else 0
            
            # Calculate performance stability
            composite_std = np.std(composite_scores)
            driving_std = np.std(driving_scores)
            
            # Detect performance patterns
            recent_avg = np.mean(composite_scores[-4:])  # Last 4 evaluations
            historical_avg = np.mean(composite_scores[:-4])  # Earlier evaluations
            
            # Adaptive evaluation interval based on comprehensive analysis
            stability_factor = min(composite_std, driving_std)  # Lower is more stable
            trend_strength = abs(composite_trend)
            
            # Calculate new evaluation interval
            if stability_factor < 3.0 and trend_strength < 0.5:
                # Performance is stable and not trending strongly
                self.adaptive_eval_interval = min(self.max_eval_interval, self.adaptive_eval_interval + 3)
                adjustment_reason = "stable performance"
            elif trend_strength > 2.0 or stability_factor > 10.0:
                # Performance is changing rapidly or very unstable
                self.adaptive_eval_interval = max(self.min_eval_interval, self.adaptive_eval_interval - 2)
                adjustment_reason = f"high volatility (std: {stability_factor:.1f}, trend: {trend_strength:.1f})"
            elif recent_avg > historical_avg + 5.0:
                # Significant recent improvement
                self.adaptive_eval_interval = max(self.min_eval_interval, self.adaptive_eval_interval - 1)
                adjustment_reason = f"improvement detected (+{recent_avg - historical_avg:.1f})"
            elif recent_avg < historical_avg - 5.0:
                # Significant recent decline
                self.adaptive_eval_interval = max(self.min_eval_interval, self.adaptive_eval_interval - 2)
                adjustment_reason = f"decline detected ({recent_avg - historical_avg:.1f})"
            else:
                adjustment_reason = None
            
            # Log adaptive changes with detailed reasoning
            if adjustment_reason and self.adaptive_eval_interval != self.eval_interval:
                logger.info(f"Adaptive eval interval adjusted to {self.adaptive_eval_interval} "
                           f"({adjustment_reason})")
                           
            # Track difficulty adjustments for analysis
            if abs(composite_trend) > 1.0:
                difficulty_assessment = {
                    'episode': i_episode,
                    'phase': current_phase,
                    'trend': composite_trend,
                    'stability': stability_factor,
                    'avg_performance': recent_avg
                }
                self.curriculum_progress_tracker['difficulty_adjustments'].append(difficulty_assessment)
                
                # Log significant performance patterns
                if composite_trend > 1.0:
                    logger.debug(f"Phase {current_phase + 1}: Strong improvement trend detected "
                               f"(+{composite_trend:.2f} per evaluation)")
                elif composite_trend < -1.0:
                    logger.debug(f"Phase {current_phase + 1}: Performance decline detected "
                               f"({composite_trend:.2f} per evaluation)")
            
            # Adaptive threshold adjustments for phase completion
            if hasattr(self.env.curriculum_manager, 'evaluation_enabled') and self.env.curriculum_manager.evaluation_enabled:
                # Suggest threshold adjustments based on performance patterns
                if stability_factor > 15.0:
                    self.curriculum_progress_tracker['adaptive_threshold_adjustments'] += 1
                    if self.curriculum_progress_tracker['adaptive_threshold_adjustments'] % 3 == 0:
                        logger.info(f"High performance variance detected in phase {current_phase + 1}. "
                                   f"Consider reviewing phase difficulty or success criteria.")

    def _adaptive_epsilon_decay(self, performance_report: dict, current_epsilon: float) -> float:
        """Robust adaptive epsilon decay based on performance trends and confidence intervals."""
        if not self.adaptive_epsilon_enabled:
            return self._update_epsilon(current_epsilon)
            
        # Collect multiple performance metrics for robust decision making
        driving_score = performance_report.get('overall_driving_score', 0.0)
        goal_rate = performance_report.get('goal_success_rate', 0.0)
        collision_rate = 100.0 - performance_report.get('collision_free_rate', 0.0)
        
        # Create composite performance score
        composite_score = (
            0.5 * driving_score +  # Overall driving quality
            0.3 * goal_rate +      # Success rate
            0.2 * max(0, 100 - collision_rate * 2)  # Safety (penalize collisions heavily)
        )
        
        self.performance_buffer.append(composite_score)
        self._performance_confidence_buffer.append({
            'driving_score': driving_score,
            'goal_rate': goal_rate,
            'collision_rate': collision_rate,
            'composite': composite_score
        })
        
        if len(self.performance_buffer) >= 5:
            # Calculate robust statistics
            performance_scores = np.array(list(self.performance_buffer))
            recent_avg = np.mean(performance_scores)
            recent_std = np.std(performance_scores)
            
            # Calculate confidence intervals for decision making
            confidence_level = 0.8
            margin_of_error = 1.28 * (recent_std / np.sqrt(len(performance_scores)))  # 80% confidence
            lower_bound = recent_avg - margin_of_error
            upper_bound = recent_avg + margin_of_error
            
            # Detect performance trend using linear regression
            if len(performance_scores) >= 3:
                x = np.arange(len(performance_scores))
                trend_slope = np.polyfit(x, performance_scores, 1)[0]
                trend_strength = abs(trend_slope)
            else:
                trend_slope = 0.0
                trend_strength = 0.0
            
            # Get curriculum context if available
            curriculum_factor = 1.0
            if (self.use_curriculum and hasattr(self.env, 'curriculum_manager') and 
                self.env.curriculum_manager):
                current_phase = self.env.curriculum_manager.get_current_phase_number()
                total_phases = len(self.env.curriculum_manager.phases)
                curriculum_progress = current_phase / total_phases
                # Early phases should be more conservative with epsilon reduction
                curriculum_factor = 0.5 + 0.5 * curriculum_progress
            
            # Determine epsilon decay adjustment based on robust analysis
            base_adjustment = 1.0
            
            # Performance-based adjustment with confidence consideration
            if lower_bound > 75:  # Confidently good performance
                base_adjustment *= 0.96  # Faster decay
            elif upper_bound > 65 and trend_slope > 0.5:  # Improving performance
                base_adjustment *= 0.98  # Moderate faster decay
            elif upper_bound < 40:  # Confidently poor performance
                base_adjustment *= 1.04  # Slower decay
            elif lower_bound < 50 and trend_slope < -0.5:  # Declining performance
                base_adjustment *= 1.02  # Moderate slower decay
            
            # Trend-based adjustment
            if trend_strength > 1.0:  # Strong trend detected
                if trend_slope > 0:  # Improving
                    base_adjustment *= 0.99
                else:  # Declining
                    base_adjustment *= 1.01
            
            # Stability-based adjustment
            if recent_std > 15:  # High performance variance - be conservative
                base_adjustment *= 1.01
            elif recent_std < 5:  # Low variance - can be more aggressive
                base_adjustment *= 0.99
            
            # Apply curriculum factor
            final_adjustment = (base_adjustment - 1.0) * curriculum_factor + 1.0
            adjusted_decay = self.base_epsilon_decay * final_adjustment
            
            # Clamp the adjustment to reasonable bounds
            adjusted_decay = np.clip(adjusted_decay, self.base_epsilon_decay * 0.95, self.base_epsilon_decay * 1.05)
            
            # Apply adjusted decay
            new_epsilon = max(self.epsilon_end, adjusted_decay * current_epsilon)
            
            # Log significant changes with detailed reasoning
            if abs(adjusted_decay - self.base_epsilon_decay) > 0.001:
                logger.debug(f"Adaptive epsilon decay: {adjusted_decay:.4f} "
                           f"(perf: {recent_avg:.1f}±{margin_of_error:.1f}, "
                           f"trend: {trend_slope:+.2f}, "
                           f"curriculum: {curriculum_factor:.2f})")
                
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
                    
            # Log robust training statistics
            training_stats = self.get_training_statistics()
            
            # Loss tracking metrics
            if training_stats['loss_tracking']['current_ema'] is not None:
                self.tb_logger.log_scalar("robust_training/loss_ema", 
                                        training_stats['loss_tracking']['current_ema'], i_episode)
            
            # Performance confidence metrics
            if 'recent_performance' in training_stats:
                perf = training_stats['recent_performance']
                self.tb_logger.log_scalar("robust_training/performance_mean", perf['mean'], i_episode)
                self.tb_logger.log_scalar("robust_training/performance_std", perf['std'], i_episode)
                self.tb_logger.log_scalar("robust_training/performance_trend", perf['trend'], i_episode)
                
            # Adaptation metrics
            self.tb_logger.log_scalar("robust_training/adaptive_eval_interval", 
                                    training_stats['curriculum_progress']['current_eval_interval'], i_episode)
            self.tb_logger.log_scalar("robust_training/gradient_explosions", 
                                    training_stats['stability_metrics']['gradient_explosion_count'], i_episode)
            self.tb_logger.log_scalar("robust_training/performance_plateaus", 
                                    training_stats['stability_metrics']['performance_plateau_count'], i_episode) 

    def _comprehensive_model_evaluation(self, i_episode: int) -> float:
        """
        Perform a comprehensive evaluation of the current model across ALL curriculum phases.
        This gives a true measure of model performance across the entire curriculum.
        
        Args:
            i_episode: Current training episode number
            
        Returns:
            Combined driving score across all phases (0-100 scale)
        """
        if not self.use_curriculum or not hasattr(self.env, 'curriculum_manager'):
            # Fall back to current phase evaluation for non-curriculum training
            return self._last_performance_report.get('overall_driving_score', 0.0) if hasattr(self, '_last_performance_report') and self._last_performance_report else 0.0
        
        logger.info(f"--- Starting Comprehensive Model Evaluation (All Phases) at Episode {i_episode} ---")
        
        # Store current curriculum state
        original_phase_idx = self.env.curriculum_manager.current_phase_idx
        original_episode_in_phase = self.env.curriculum_manager.episode_in_current_phase
        original_phase_repeat_count = self.env.curriculum_manager.phase_repeat_count
        original_pygame_state = self.env.enable_pygame_display
        
        # Disable pygame display during comprehensive evaluation
        self.env.enable_pygame_display = False
        
        phase_scores = []
        phase_weights = []
        total_evaluation_episodes = 0
        
        try:
            # Evaluate model on each curriculum phase
            for phase_idx in range(len(self.env.curriculum_manager.phases)):
                phase_config = self.env.curriculum_manager.phases[phase_idx]
                phase_name = phase_config.get('name', f'Phase{phase_idx+1}')
                
                logger.info(f"  Evaluating on {phase_name} (Phase {phase_idx+1}/{len(self.env.curriculum_manager.phases)})")
                
                # Set curriculum manager to this phase
                self.env.curriculum_manager.current_phase_idx = phase_idx
                self.env.curriculum_manager.episode_in_current_phase = 1
                self.env.curriculum_manager.phase_repeat_count = 0
                
                # Update environment configuration for this phase
                if hasattr(self.env, '_update_phase_configuration'):
                    self.env._update_phase_configuration()
                
                # Run evaluation episodes for this phase
                phase_eval_episodes = min(3, self.env.curriculum_manager.evaluation_episodes)  # Use fewer episodes for efficiency
                total_evaluation_episodes += phase_eval_episodes
                
                phase_metrics = []
                for eval_ep in range(phase_eval_episodes):
                    episode_score, steps_taken, termination_reason, detailed_metrics = self._run_evaluation_episode(eval_ep + 1)
                    phase_metrics.append(detailed_metrics)
                
                # Calculate performance for this phase
                phase_performance = self._calculate_performance_metrics(
                    phase_metrics, 
                    sum(m.get('steps_taken', 0) for m in phase_metrics),
                    sum(1 for m in phase_metrics if m.get('goal_reached', False)),
                    {}  # Termination reasons not needed for this calculation
                )
                
                phase_driving_score = phase_performance.get('overall_driving_score', 0.0)
                phase_scores.append(phase_driving_score)
                
                # Weight phases by their complexity (later phases are more important)
                phase_weight = 1.0 + (phase_idx * 0.2)  # Earlier phases have weight 1.0, later phases up to 3.0
                phase_weights.append(phase_weight)
                
                logger.info(f"    {phase_name}: {phase_driving_score:.1f}/100 (weight: {phase_weight:.1f})")
            
            # Calculate weighted average across all phases
            if phase_scores and phase_weights:
                weighted_sum = sum(score * weight for score, weight in zip(phase_scores, phase_weights))
                total_weight = sum(phase_weights)
                comprehensive_score = weighted_sum / total_weight if total_weight > 0 else 0.0
                
                logger.info(f"Comprehensive Evaluation Results:")
                logger.info(f"  Total evaluation episodes: {total_evaluation_episodes}")
                logger.info(f"  Phase scores: {[f'{s:.1f}' for s in phase_scores]}")
                logger.info(f"  Weighted average: {comprehensive_score:.1f}/100")
                
                # Save comprehensive evaluation report
                self._save_comprehensive_evaluation_report(i_episode, phase_scores, phase_weights, comprehensive_score)
                
                return comprehensive_score
            else:
                logger.warning("No phase scores calculated during comprehensive evaluation")
                return 0.0
                
        except Exception as e:
            logger.error(f"Error during comprehensive model evaluation: {e}", exc_info=True)
            return 0.0
        finally:
            # Restore original curriculum state
            self.env.curriculum_manager.current_phase_idx = original_phase_idx
            self.env.curriculum_manager.episode_in_current_phase = original_episode_in_phase
            self.env.curriculum_manager.phase_repeat_count = original_phase_repeat_count
            self.env.enable_pygame_display = original_pygame_state
            
            # Update environment configuration back to current phase
            if hasattr(self.env, '_update_phase_configuration'):
                self.env._update_phase_configuration()
            
            logger.info(f"--- Comprehensive Model Evaluation Complete ---")

    def _save_comprehensive_evaluation_report(self, episode: int, phase_scores: List[float], 
                                            phase_weights: List[float], comprehensive_score: float):
        """Save detailed comprehensive evaluation report."""
        report_path = os.path.join(self.reports_dir, f"episode_{episode:03d}_comprehensive_evaluation.txt")
        
        with open(report_path, "w") as f:
            f.write(f"COMPREHENSIVE MODEL EVALUATION REPORT - Episode {episode}\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EVALUATION METHODOLOGY:\n")
            f.write("  This evaluation tests the model across ALL curriculum phases to determine\n")
            f.write("  comprehensive performance rather than single-phase performance.\n\n")
            
            f.write("PHASE-BY-PHASE RESULTS:\n")
            for i, (score, weight) in enumerate(zip(phase_scores, phase_weights)):
                phase_name = self.env.curriculum_manager.phases[i].get('name', f'Phase{i+1}')
                f.write(f"  {phase_name:30s}: {score:5.1f}/100 (weight: {weight:.1f})\n")
            
            f.write(f"\nWEIGHTED AVERAGE SCORE: {comprehensive_score:.1f}/100\n")
            
            f.write(f"\nSCORE INTERPRETATION:\n")
            if comprehensive_score >= 85:
                f.write("  EXCELLENT! Model performs well across all curriculum phases.\n")
            elif comprehensive_score >= 70:
                f.write("  GOOD overall performance with room for improvement in some phases.\n")
            elif comprehensive_score >= 55:
                f.write("  FAIR performance. Model struggles with advanced phases.\n")
            else:
                f.write("  POOR performance. Model needs significant improvement.\n")
            
            f.write(f"\nNOTE: This comprehensive score is used for best model selection\n")
            f.write(f"instead of single-phase scores to ensure robust performance.\n")