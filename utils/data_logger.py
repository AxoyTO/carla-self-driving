import os
import carla
import logging
import json
import numpy as np
import asyncio
import aiofiles
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Performance optimization imports
try:
    import numba
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

import multiprocessing as mp
MULTIPROCESS_AVAILABLE = True

@dataclass
class SensorDataItem:
    """Data structure for sensor data queue items."""
    episode_count: int
    step_count: int
    sensor_key: str
    data: Any
    timestamp: float
    priority: int = 1  # Lower number = higher priority

# Numba-optimized functions for data preprocessing
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _compress_image_data_numba(image_array: np.ndarray, compression_factor: float) -> np.ndarray:
        """Compress image data by reducing resolution using Numba JIT."""
        height, width = image_array.shape[:2]
        new_height = int(height * compression_factor)
        new_width = int(width * compression_factor)
        
        if len(image_array.shape) == 3:  # Color image
            compressed = np.zeros((new_height, new_width, image_array.shape[2]), dtype=image_array.dtype)
            for i in range(new_height):
                for j in range(new_width):
                    orig_i = int(i / compression_factor)
                    orig_j = int(j / compression_factor)
                    if orig_i < height and orig_j < width:
                        for c in range(image_array.shape[2]):
                            compressed[i, j, c] = image_array[orig_i, orig_j, c]
        else:  # Grayscale image
            compressed = np.zeros((new_height, new_width), dtype=image_array.dtype)
            for i in range(new_height):
                for j in range(new_width):
                    orig_i = int(i / compression_factor)
                    orig_j = int(j / compression_factor)
                    if orig_i < height and orig_j < width:
                        compressed[i, j] = image_array[orig_i, orig_j]
        
        return compressed

    @jit(nopython=True, cache=True)
    def _filter_lidar_points_numba(points: np.ndarray, max_distance: float, 
                                   min_z: float, max_z: float) -> np.ndarray:
        """Filter LIDAR points by distance and height using Numba JIT."""
        valid_indices = []
        for i in range(points.shape[0]):
            x, y, z = points[i, 0], points[i, 1], points[i, 2]
            distance = np.sqrt(x*x + y*y)
            if distance <= max_distance and min_z <= z <= max_z:
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return np.empty((0, points.shape[1]), dtype=points.dtype)
        
        filtered_points = np.zeros((len(valid_indices), points.shape[1]), dtype=points.dtype)
        for i in range(len(valid_indices)):
            for j in range(points.shape[1]):
                filtered_points[i, j] = points[valid_indices[i], j]
        
        return filtered_points

def preprocess_image_data(image_data: carla.Image, compression_factor: float = 0.5) -> np.ndarray:
    """Preprocess image data for efficient storage."""
    try:
        # Convert CARLA image to numpy array
        array = np.frombuffer(image_data.raw_data, dtype=np.uint8)
        array = array.reshape((image_data.height, image_data.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        
        # Apply compression if requested
        if compression_factor < 1.0 and NUMBA_AVAILABLE:
            return _compress_image_data_numba(array, compression_factor)
        else:
            return array
            
    except Exception as e:
        logging.error(f"Error preprocessing image data: {e}")
        return np.zeros((84, 84, 3), dtype=np.uint8)  # Fallback

def preprocess_lidar_data(lidar_data: carla.LidarMeasurement, 
                         max_distance: float = 50.0, 
                         min_z: float = -2.0, 
                         max_z: float = 10.0) -> np.ndarray:
    """Preprocess LIDAR data for efficient storage."""
    try:
        # Convert to numpy array
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = points.reshape((-1, 4))  # x, y, z, intensity
        
        # Filter points if numba is available
        if NUMBA_AVAILABLE and points.shape[0] > 0:
            return _filter_lidar_points_numba(points, max_distance, min_z, max_z)
        else:
            # Fallback filtering
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            valid_mask = (distances <= max_distance) & (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
            return points[valid_mask]
            
    except Exception as e:
        logging.error(f"Error preprocessing LIDAR data: {e}")
        return np.zeros((0, 4), dtype=np.float32)  # Fallback

class AsyncDataLogger:
    """High-performance data logger with async I/O and multiprocessing capabilities."""

    def __init__(self, base_save_path: str, run_name: Optional[str] = None, 
                 logger: Optional[logging.Logger] = None, 
                 max_workers: int = 4, 
                 use_multiprocessing: bool = True,
                 enable_compression: bool = True,
                 compression_factor: float = 0.7):
        """
        Initialize the AsyncDataLogger with performance optimizations.
        
        Args:
            base_save_path: The root directory where all run data will be saved
            run_name: Optional specific name for this run
            logger: Optional logger instance
            max_workers: Maximum number of worker threads/processes
            use_multiprocessing: Whether to use multiprocessing for data preprocessing
            enable_compression: Whether to enable data compression
            compression_factor: Compression factor for image data (0.1-1.0)
        """
        self.logger = logger if logger else logging.getLogger(__name__ + ".AsyncDataLogger")
        self.run_name = run_name if run_name else datetime.now().strftime('%Y%m%d-%H%M%S')
        self.current_run_save_path = Path(base_save_path) / self.run_name
        self._sensor_save_dirs: Dict[str, Path] = {}
        
        # Performance configuration
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing and MULTIPROCESS_AVAILABLE
        self.enable_compression = enable_compression
        self.compression_factor = max(0.1, min(1.0, compression_factor))
        
        # Async processing components
        self.data_queue = Queue(maxsize=1000)  # Larger queue for burst processing
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers) if self.use_multiprocessing else None
        
        # Background processing
        self._processing_thread = None
        self._stop_processing = threading.Event()
        self._stats = {
            'items_processed': 0,
            'items_queued': 0,
            'processing_time_total': 0.0,
            'last_processing_time': 0.0
        }
        
        try:
            self.current_run_save_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"AsyncDataLogger initialized. Run data will be saved in: {self.current_run_save_path}")
            self._start_background_processing()
        except Exception as e:
            self.logger.error(f"Could not create base run directory {self.current_run_save_path}: {e}", exc_info=True)
            self.current_run_save_path = None

    def _start_background_processing(self):
        """Start background thread for processing data queue."""
        self._processing_thread = threading.Thread(target=self._process_data_queue, daemon=True)
        self._processing_thread.start()
        self.logger.info("Background data processing thread started")

    def _process_data_queue(self):
        """Background thread function to process queued data."""
        while not self._stop_processing.is_set():
            try:
                # Get data from queue with timeout
                try:
                    data_item = self.data_queue.get(timeout=1.0)
                except:
                    continue  # Timeout, check stop condition
                
                if data_item is None:  # Sentinel value to stop
                    break
                
                start_time = time.time()
                
                # Process the data item
                self._process_single_item(data_item)
                
                # Update statistics
                processing_time = time.time() - start_time
                self._stats['items_processed'] += 1
                self._stats['processing_time_total'] += processing_time
                self._stats['last_processing_time'] = processing_time
                
                self.data_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in background data processing: {e}", exc_info=True)
                if not self.data_queue.empty():
                    self.data_queue.task_done()

    def _process_single_item(self, data_item: SensorDataItem):
        """Process a single data item asynchronously."""
        try:
            if self.use_multiprocessing and data_item.sensor_key in ['rgb_camera', 'lidar']:
                # Use multiprocessing for computationally intensive preprocessing
                future = self.process_pool.submit(
                    self._preprocess_and_save_mp, 
                    data_item
                )
                # Don't wait for completion to avoid blocking
            else:
                # Use thread pool for I/O bound operations
                future = self.thread_pool.submit(
                    self._save_sensor_data_sync, 
                    data_item.episode_count,
                    data_item.step_count,
                    data_item.sensor_key,
                    data_item.data
                )
                
        except Exception as e:
            self.logger.error(f"Error processing data item for {data_item.sensor_key}: {e}", exc_info=True)

    def setup_sensor_save_subdirs(self, sensor_keys: List[str]):
        """Creates subdirectories for each specified sensor key."""
        if not self.current_run_save_path:
            self.logger.error("Cannot setup sensor subdirectories, run save path not initialized.")
            return

        for sensor_key in sensor_keys:
            s_path = self.current_run_save_path / sensor_key
            try:
                s_path.mkdir(exist_ok=True)
                self._sensor_save_dirs[sensor_key] = s_path
            except Exception as e:
                self.logger.error(f"Could not create subdirectory for sensor {sensor_key} at {s_path}: {e}")
        
        if self._sensor_save_dirs:
            self.logger.info(f"Created subdirectories for sensor data saving: {list(self._sensor_save_dirs.keys())}")

    def save_sensor_data(self, episode_count: int, step_count: int, sensor_key: str, data: Any, priority: int = 1):
        """
        Queue sensor data for asynchronous saving.
        
        Args:
            episode_count: Current episode number
            step_count: Current step number
            sensor_key: Identifier for the sensor
            data: The sensor data to save
            priority: Priority level (lower = higher priority)
        """
        if not self.current_run_save_path or sensor_key not in self._sensor_save_dirs:
            return

        try:
            data_item = SensorDataItem(
                episode_count=episode_count,
                step_count=step_count,
                sensor_key=sensor_key,
                data=data,
                timestamp=time.time(),
                priority=priority
            )
            
            # Add to queue (non-blocking)
            try:
                self.data_queue.put_nowait(data_item)
                self._stats['items_queued'] += 1
            except:
                # Queue is full, log warning but don't block
                self.logger.warning(f"Data queue full, dropping {sensor_key} data for ep{episode_count}_step{step_count}")
                
        except Exception as e:
            self.logger.error(f"Error queuing sensor data for {sensor_key}: {e}", exc_info=True)

    @staticmethod
    def _preprocess_and_save_mp(data_item: SensorDataItem) -> bool:
        """Multiprocessing function for preprocessing and saving data."""
        try:
            # This runs in a separate process
            if data_item.sensor_key == 'rgb_camera' and isinstance(data_item.data, carla.Image):
                processed_data = preprocess_image_data(data_item.data, 0.7)
                # Save processed data
                filename = f"ep{data_item.episode_count:04d}_step{data_item.step_count:05d}_processed.npy"
                save_path = Path(data_item.data.save_dir) / filename
                np.save(save_path, processed_data)
                return True
            elif data_item.sensor_key == 'lidar' and isinstance(data_item.data, carla.LidarMeasurement):
                processed_data = preprocess_lidar_data(data_item.data)
                # Save processed data
                filename = f"ep{data_item.episode_count:04d}_step{data_item.step_count:05d}_processed.npy"
                save_path = Path(data_item.data.save_dir) / filename
                np.save(save_path, processed_data)
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Error in multiprocess preprocessing: {e}")
            return False

    def _save_sensor_data_sync(self, episode_count: int, step_count: int, sensor_key: str, data: Any):
        """Synchronous sensor data saving with optimizations."""
        save_dir = self._sensor_save_dirs[sensor_key]
        filename_base = f"ep{episode_count:04d}_step{step_count:05d}"

        try:
            if isinstance(data, carla.Image):
                # Handle different image types with compression options
                if self.enable_compression and sensor_key.startswith('rgb'):
                    # Save both compressed and original for RGB cameras
                    compressed_data = preprocess_image_data(data, self.compression_factor)
                    np.save(save_dir / f"{filename_base}_compressed.npy", compressed_data)
                
                # Save original based on sensor type
                if sensor_key.startswith('rgb'):
                    data.save_to_disk(str(save_dir / f"{filename_base}.png"))
                elif sensor_key.startswith('depth'):
                    data.save_to_disk(str(save_dir / f"{filename_base}.png"), carla.ColorConverter.LogarithmicDepth)
                elif sensor_key.startswith('semantic'):
                    data.save_to_disk(str(save_dir / f"{filename_base}.png"))
                else:
                    data.save_to_disk(str(save_dir / f"{filename_base}_{sensor_key}.png"))
            
            elif isinstance(data, carla.LidarMeasurement):
                if self.enable_compression:
                    # Save preprocessed/filtered LIDAR data
                    processed_data = preprocess_lidar_data(data)
                    np.save(save_dir / f"{filename_base}_filtered.npy", processed_data)
                # Save original
                data.save_to_disk(str(save_dir / f"{filename_base}.ply"))
            
            elif isinstance(data, carla.SemanticLidarMeasurement):
                data.save_to_disk(str(save_dir / f"{filename_base}_semantic.ply"))
            
            elif isinstance(data, carla.GnssMeasurement):
                gnss_dict = {
                    'frame': data.frame, 'timestamp': data.timestamp,
                    'latitude': data.latitude, 'longitude': data.longitude, 'altitude': data.altitude
                }
                with open(save_dir / f"{filename_base}.json", 'w') as f:
                    json.dump(gnss_dict, f, indent=2)  # Reduced indent for smaller files
            
            elif isinstance(data, carla.IMUMeasurement):
                imu_dict = {
                    'frame': data.frame, 'timestamp': data.timestamp,
                    'accelerometer': {'x': data.accelerometer.x, 'y': data.accelerometer.y, 'z': data.accelerometer.z},
                    'gyroscope': {'x': data.gyroscope.x, 'y': data.gyroscope.y, 'z': data.gyroscope.z},
                    'compass': data.compass
                }
                with open(save_dir / f"{filename_base}.json", 'w') as f:
                    json.dump(imu_dict, f, indent=2)
            
            elif isinstance(data, carla.RadarMeasurement):
                detections = [
                    {
                        'altitude': detection.altitude,
                        'azimuth': detection.azimuth,
                        'depth': detection.depth,
                        'velocity': detection.velocity
                    }
                    for detection in data
                ]
                radar_dict = {'frame': data.frame, 'timestamp': data.timestamp, 'detections': detections}
                with open(save_dir / f"{filename_base}.json", 'w') as f:
                    json.dump(radar_dict, f, indent=2)
            
            elif sensor_key == 'collision' and isinstance(data, dict):
                with open(save_dir / f"{filename_base}.json", 'w') as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving data for sensor {sensor_key}: {e}", exc_info=True)

    async def save_episode_summary_async(self, episode_num: int, summary_data: Dict[str, Any]):
        """Asynchronously save episode summary data."""
        if not self.current_run_save_path:
            self.logger.error("Cannot log episode summary, run save path not initialized.")
            return

        ep_summary_dir = self.current_run_save_path / "episode_summaries"
        try:
            ep_summary_dir.mkdir(exist_ok=True)
            file_path = ep_summary_dir / f"episode_{episode_num:04d}_summary.json"
            
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(summary_data, indent=2))
            
            self.logger.debug(f"Logged episode {episode_num} summary to {file_path}")
        except Exception as e:
            self.logger.error(f"Error logging episode summary for episode {episode_num}: {e}", exc_info=True)

    def log_episode_summary(self, episode_num: int, summary_data: Dict[str, Any]):
        """Synchronous episode summary logging for backward compatibility."""
        if not self.current_run_save_path:
            self.logger.error("Cannot log episode summary, run save path not initialized.")
            return

        ep_summary_dir = self.current_run_save_path / "episode_summaries"
        try:
            ep_summary_dir.mkdir(exist_ok=True)
            file_path = ep_summary_dir / f"episode_{episode_num:04d}_summary.json"
            with open(file_path, 'w') as f:
                json.dump(summary_data, f, indent=2)
            self.logger.debug(f"Logged episode {episode_num} summary to {file_path}")
        except Exception as e:
            self.logger.error(f"Error logging episode summary for episode {episode_num}: {e}", exc_info=True)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        queue_size = self.data_queue.qsize() if hasattr(self.data_queue, 'qsize') else 0
        avg_processing_time = (
            self._stats['processing_time_total'] / max(1, self._stats['items_processed'])
        )
        
        return {
            'items_processed': self._stats['items_processed'],
            'items_queued': self._stats['items_queued'],
            'queue_size': queue_size,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'last_processing_time_ms': self._stats['last_processing_time'] * 1000,
            'multiprocessing_enabled': self.use_multiprocessing,
            'compression_enabled': self.enable_compression,
            'compression_factor': self.compression_factor,
            'max_workers': self.max_workers
        }

    def get_run_save_path(self) -> Optional[str]:
        return str(self.current_run_save_path) if self.current_run_save_path else None

    def shutdown(self):
        """Gracefully shutdown the data logger."""
        self.logger.info("Shutting down AsyncDataLogger...")
        
        # Stop background processing
        self._stop_processing.set()
        
        # Wait for queue to be processed
        if not self.data_queue.empty():
            self.logger.info("Waiting for data queue to be processed...")
            try:
                self.data_queue.join()  # Wait for all tasks to complete
            except:
                pass  # Queue might not support join()
        
        # Shutdown thread pool
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        # Shutdown process pool
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        # Wait for processing thread to finish
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5.0)
        
        stats = self.get_performance_stats()
        self.logger.info(f"AsyncDataLogger shutdown complete. Final stats: {stats}")

    def __del__(self):
        """Destructor to ensure clean shutdown."""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during cleanup

# Backward compatibility
DataLogger = AsyncDataLogger  # For existing code that uses DataLogger 