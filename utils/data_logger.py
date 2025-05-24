import os
import carla
import logging
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

class DataLogger:
    """Handles saving of simulation data, including sensor readings and episode info."""

    def __init__(self, base_save_path: str, run_name: Optional[str] = None, 
                 logger: Optional[logging.Logger] = None):
        """Initialize the DataLogger.
        Args:
            base_save_path: The root directory where all run data will be saved.
            run_name: Optional specific name for this run. If None, a timestamped name is generated.
            logger: Optional logger instance.
        """
        self.logger = logger if logger else logging.getLogger(__name__ + ".DataLogger")
        self.run_name = run_name if run_name else datetime.now().strftime('%Y%m%d-%H%M%S')
        self.current_run_save_path = os.path.join(base_save_path, self.run_name)
        self._sensor_save_dirs: Dict[str, str] = {}

        try:
            os.makedirs(self.current_run_save_path, exist_ok=True)
            self.logger.info(f"DataLogger initialized. Run data will be saved in: {self.current_run_save_path}")
        except Exception as e:
            self.logger.error(f"Could not create base run directory {self.current_run_save_path}: {e}", exc_info=True)
            # Potentially raise or handle this more gracefully if path is critical
            self.current_run_save_path = None # Indicate failure to init save path

    def setup_sensor_save_subdirs(self, sensor_keys: List[str]):
        """Creates subdirectories for each specified sensor key within the current run path.
        Args:
            sensor_keys: A list of strings, where each key is a sensor identifier (e.g., 'rgb_camera', 'lidar').
        """
        if not self.current_run_save_path:
            self.logger.error("Cannot setup sensor subdirectories, run save path not initialized.")
            return

        for sensor_key in sensor_keys:
            s_path = os.path.join(self.current_run_save_path, sensor_key)
            try:
                os.makedirs(s_path, exist_ok=True)
                self._sensor_save_dirs[sensor_key] = s_path
            except Exception as e:
                self.logger.error(f"Could not create subdirectory for sensor {sensor_key} at {s_path}: {e}")
        if self._sensor_save_dirs:
            self.logger.info(f"Created subdirectories for sensor data saving: {list(self._sensor_save_dirs.keys())}")

    def save_sensor_data(self, episode_count: int, step_count: int, sensor_key: str, data: Any):
        """Saves data for a specific sensor to disk.
        Args:
            episode_count: Current episode number.
            step_count: Current step number within the episode.
            sensor_key: Identifier for the sensor (e.g., 'rgb_camera', 'lidar').
            data: The sensor data to save.
        """
        if not self.current_run_save_path or sensor_key not in self._sensor_save_dirs:
            # self.logger.debug(f"Sensor data saving skipped for {sensor_key}: path or subdir not ready.")
            return

        save_dir = self._sensor_save_dirs[sensor_key]
        filename_base = f"ep{episode_count:04d}_step{step_count:05d}"

        try:
            if isinstance(data, carla.Image):
                # Determine format based on typical usage for sensor types
                if sensor_key.startswith('rgb') or sensor_key.startswith('display_rgb') or sensor_key == 'spectator_camera':
                    data.save_to_disk(os.path.join(save_dir, f"{filename_base}.png"))
                elif sensor_key.startswith('depth') or sensor_key.startswith('display_depth'):
                    # Saving depth as raw might be better, but PNG is common for visualization
                    data.save_to_disk(os.path.join(save_dir, f"{filename_base}.png"), carla.ColorConverter.LogarithmicDepth)
                elif sensor_key.startswith('semantic') or sensor_key.startswith('display_semantic'):
                    data.save_to_disk(os.path.join(save_dir, f"{filename_base}.png")) # Saved with cityscapes palette by default
                else:
                    data.save_to_disk(os.path.join(save_dir, f"{filename_base}_{sensor_key}.png"))
            
            elif isinstance(data, carla.LidarMeasurement) and sensor_key == 'lidar':
                data.save_to_disk(os.path.join(save_dir, f"{filename_base}.ply"))
            
            elif isinstance(data, carla.SemanticLidarMeasurement) and sensor_key == 'semantic_lidar':
                data.save_to_disk(os.path.join(save_dir, f"{filename_base}_semantic.ply"))
            
            elif isinstance(data, carla.GnssMeasurement) and sensor_key == 'gnss':
                gnss_dict = {'frame': data.frame, 'timestamp': data.timestamp,
                               'latitude': data.latitude, 'longitude': data.longitude, 'altitude': data.altitude}
                with open(os.path.join(save_dir, f"{filename_base}.json"), 'w') as f:
                    json.dump(gnss_dict, f, indent=4)
            
            elif isinstance(data, carla.IMUMeasurement) and sensor_key == 'imu':
                imu_dict = {
                    'frame': data.frame, 'timestamp': data.timestamp,
                    'accelerometer': {'x': data.accelerometer.x, 'y': data.accelerometer.y, 'z': data.accelerometer.z},
                    'gyroscope': {'x': data.gyroscope.x, 'y': data.gyroscope.y, 'z': data.gyroscope.z},
                    'compass': data.compass
                }
                with open(os.path.join(save_dir, f"{filename_base}.json"), 'w') as f:
                    json.dump(imu_dict, f, indent=4)
            
            elif isinstance(data, carla.RadarMeasurement) and sensor_key == 'radar':
                detections = []
                for detection in data:
                    detections.append({
                        'altitude': detection.altitude,
                        'azimuth': detection.azimuth,
                        'depth': detection.depth,
                        'velocity': detection.velocity
                    })
                radar_dict = {'frame': data.frame, 'timestamp': data.timestamp, 'detections': detections}
                with open(os.path.join(save_dir, f"{filename_base}.json"), 'w') as f:
                    json.dump(radar_dict, f, indent=4)
            
            # Add other sensor types and their saving logic here (e.g., collision to JSON/txt)
            elif sensor_key == 'collision' and isinstance(data, dict): # Assuming collision data is passed as a dict
                 with open(os.path.join(save_dir, f"{filename_base}.json"), 'w') as f:
                    json.dump(data, f, indent=4)

            # self.logger.debug(f"Saved data for sensor {sensor_key} to {os.path.join(save_dir, filename_base + '.*')}")
        except Exception as e:
            self.logger.error(f"Error saving data for sensor {sensor_key}: {e}", exc_info=True)

    def log_episode_summary(self, episode_num: int, summary_data: Dict[str, Any]):
        """Logs summary data for a completed episode to a JSON file.
        Args:
            episode_num: The episode number that just finished.
            summary_data: A dictionary containing data to log (e.g., score, steps, termination reason).
        """
        if not self.current_run_save_path:
            self.logger.error("Cannot log episode summary, run save path not initialized.")
            return

        ep_summary_dir = os.path.join(self.current_run_save_path, "episode_summaries")
        try:
            os.makedirs(ep_summary_dir, exist_ok=True)
            file_path = os.path.join(ep_summary_dir, f"episode_{episode_num:04d}_summary.json")
            with open(file_path, 'w') as f:
                json.dump(summary_data, f, indent=4)
            self.logger.debug(f"Logged episode {episode_num} summary to {file_path}")
        except Exception as e:
            self.logger.error(f"Error logging episode summary for episode {episode_num}: {e}", exc_info=True)

    def get_run_save_path(self) -> Optional[str]:
        return self.current_run_save_path 