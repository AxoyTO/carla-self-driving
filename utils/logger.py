import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir="runs", experiment_name="dqn_carla_experiment"):
        """
        Initializes a TensorBoard logger.
        Args:
            log_dir (str): The root directory for TensorBoard logs.
            experiment_name (str): Name for the current experiment.
        """
        self.timestamp = datetime.now().strftime('%Y%m%d-%H%M%S') # Store timestamp
        run_name = f"{experiment_name}_{self.timestamp}"
        self.run_dir = os.path.join(log_dir, run_name)
        
        self.writer = SummaryWriter(self.run_dir)
        print(f"TensorBoard logs will be saved to: {self.run_dir}")

    def get_timestamp(self) -> str:
        """Returns the timestamp string created during initialization."""
        return self.timestamp

    def log_scalar(self, tag, value, step):
        """
        Logs a scalar value.
        Args:
            tag (str): Data identifier (e.g., 'Loss/train', 'Episode/reward').
            value (float): Value to log.
            step (int): Global step or episode number.
        """
        if value is not None: # Only log if value is not None
            self.writer.add_scalar(tag, value, step)

    def log_episode_stats(self, episode, score, avg_score, epsilon, total_steps, avg_loss=None):
        """
        Logs common episode statistics.
        Args:
            episode (int): Current episode number.
            score (float): Score for the current episode.
            avg_score (float): Average score over the last 100 episodes.
            epsilon (float): Current epsilon value for exploration.
            total_steps (int): Total steps taken in the episode.
            avg_loss (float, optional): Average loss during the episode.
        """
        self.log_scalar("Episode/Score", score, episode)
        self.log_scalar("Episode/AverageScore_Last100", avg_score, episode)
        self.log_scalar("Episode/Epsilon", epsilon, episode)
        self.log_scalar("Episode/Steps", total_steps, episode)
        if avg_loss is not None:
            self.log_scalar("Episode/AverageLoss", avg_loss, episode)

    def close(self):
        """
        Closes the SummaryWriter.
        """
        self.writer.close()
        print(f"TensorBoard writer closed. Logs saved in {self.run_dir}")
