#!/usr/bin/env python3
"""
Standalone Model Evaluation Script

This script allows you to evaluate any trained DQN model with comprehensive,
interpretable metrics that matter for self-driving car performance.

Usage:
    python evaluate_model.py --model_path models/best_model --num_episodes 10

Example Output:
    Evaluation completed - Driving Score: 78.5/100 (C)
      Goal Rate: 80.0% | Collision-Free: 90.0% | Rule Compliance: 85.0%
      Detailed report saved to: reports/standalone_evaluations/best_model_evaluation_20231201_150234.txt
"""

import argparse
import sys
import os
import logging
import time
from typing import Dict, List
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import project modules
from app.environments.carla_env import CarlaEnv
from app.agents.dqn_agent import DQNAgent
from app.training.dqn_trainer import DQNTrainer
import app.config as config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Standalone model evaluator with comprehensive metrics."""
    
    def __init__(self, model_path: str, num_episodes: int = 5, enable_display: bool = True):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the saved model directory
            num_episodes: Number of episodes to run for evaluation
            enable_display: Whether to show the pygame display during evaluation
        """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.enable_display = enable_display
        
        # Create reports directory for standalone evaluations
        os.makedirs("reports/standalone_evaluations", exist_ok=True)
        
        # Generate report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = os.path.basename(model_path).replace('/', '_')
        self.report_filename = f"{model_name}_evaluation_{timestamp}.txt"
        self.report_path = os.path.join("reports/standalone_evaluations", self.report_filename)
        
        # Initialize environment
        logger.info("Initializing CARLA environment...")
        self.env = CarlaEnv(
            host=config.ENV_HOST,
            port=config.ENV_PORT,
            town=config.ENV_TOWN,
            timestep=config.ENV_TIMESTEP,
            time_scale=config.ENV_TIME_SCALE,
            enable_pygame_display=enable_display
        )
        
        # Initialize agent
        logger.info(f"Loading model from: {model_path}")
        observation = self.env._get_observation()
        if hasattr(observation, 'spaces'):
            # Handle Dict observation space
            state_size = sum(space.shape[0] if len(space.shape) == 1 else space.shape[0] * space.shape[1] * space.shape[2] 
                           for space in observation.spaces.values() 
                           if hasattr(space, 'shape'))
        else:
            state_size = len(observation)
        
        action_size = config.NUM_DISCRETE_ACTIONS
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            lr=config.LEARNING_RATE,
            gamma=config.GAMMA,
            tau=config.TAU,
            device=config.DEVICE
        )
        
        # Load the trained model
        try:
            self.agent.load(model_path)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
        # Create a minimal trainer instance for the evaluation methods
        args_mock = type('Args', (), {
            'num_eval_episodes': num_episodes,
            'epsilon_eval': 0.01,  # Use very low epsilon for evaluation
            'save_dir': os.path.dirname(model_path)  # For compatibility
        })()
        
        self.trainer = DQNTrainer(
            agent=self.agent,
            env=self.env,
            replay_buffer=None,  # Not needed for evaluation
            tb_logger=None,      # Not needed for evaluation
            args=args_mock,
            device=config.DEVICE
        )

    def evaluate(self) -> Dict:
        """
        Run comprehensive evaluation of the model.
        
        Returns:
            Dictionary containing all performance metrics
        """
        logger.info(f"Starting evaluation of model: {os.path.basename(self.model_path)}")
        logger.info(f"Running {self.num_episodes} episodes...")
        
        try:
            # Use the trainer's evaluation methods
            avg_score, goal_rate = self.trainer.evaluate_agent()
            
            # Get the comprehensive report
            if hasattr(self.trainer, '_last_performance_report') and self.trainer._last_performance_report:
                report = self.trainer._last_performance_report
                
                # Save detailed report to file
                with open(self.report_path, "w") as f:
                    title = f"STANDALONE MODEL EVALUATION - {os.path.basename(self.model_path)}"
                    self.trainer._write_comprehensive_report(f, report, title)
                    
                    # Add evaluation details
                    f.write(f"\nEVALUATION DETAILS:\n")
                    f.write(f"   Model Path: {self.model_path}\n")
                    f.write(f"   Episodes Evaluated: {self.num_episodes}\n")
                    f.write(f"   Display Enabled: {self.enable_display}\n")
                    f.write(f"   Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Minimal console output
                driving_score = report.get('overall_driving_score', 0.0)
                grade = report.get('performance_grade', 'N/A')
                goal_rate_pct = report.get('goal_success_rate', 0.0)
                collision_free_pct = report.get('collision_free_rate', 0.0)
                rule_compliance_pct = report.get('rule_compliance_rate', 0.0)
                
                logger.info(f"Evaluation completed - Driving Score: {driving_score:.1f}/100 ({grade})")
                logger.info(f"  Goal Rate: {goal_rate_pct:.1f}% | Collision-Free: {collision_free_pct:.1f}% | Rule Compliance: {rule_compliance_pct:.1f}%")
                logger.info(f"  Detailed report saved to: {self.report_path}")
                
                return {
                    'model_path': self.model_path,
                    'episodes_evaluated': self.num_episodes,
                    'avg_score': avg_score,
                    'goal_rate': goal_rate,
                    'driving_score': driving_score,
                    'performance_grade': grade,
                    'report_path': self.report_path,
                    'comprehensive_report': report
                }
            else:
                logger.warning("No comprehensive performance report available")
                return {
                    'model_path': self.model_path,
                    'episodes_evaluated': self.num_episodes,
                    'avg_score': avg_score,
                    'goal_rate': goal_rate
                }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'env'):
                self.env.close()
            logger.info("Cleanup completed.")
        except Exception as e:
            logger.warning(f"Warning during cleanup: {e}")

def main():
    """Main function to run model evaluation from command line."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN model with comprehensive self-driving metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate best model with 10 episodes
  python evaluate_model.py --model_path models/best_model --num_episodes 10
  
  # Quick evaluation with display disabled
  python evaluate_model.py --model_path models/episode_500 --num_episodes 3 --no_display
  
  # Evaluate specific checkpoint
  python evaluate_model.py --model_path models/model_checkpoints/episode_1000
        """
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help='Path to the saved model directory (e.g., models/best_model)'
    )
    
    parser.add_argument(
        '--num_episodes', 
        type=int, 
        default=5,
        help='Number of episodes to run for evaluation (default: 5)'
    )
    
    parser.add_argument(
        '--no_display', 
        action='store_true',
        help='Disable pygame display during evaluation (faster)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        return 1
    
    try:
        # Create and run evaluator
        evaluator = ModelEvaluator(
            model_path=args.model_path,
            num_episodes=args.num_episodes,
            enable_display=not args.no_display
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"Fatal error during evaluation: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 