# Handles LIDAR sensor setup and data processing with performance optimizations

import carla
import numpy as np
import weakref
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Union

# Performance optimization imports
import numba
from numba import jit

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available. Install cupy for LIDAR GPU acceleration.")

from gymnasium import spaces

# Configure a logger for this handler
logger = logging.getLogger(__name__)

# Default LIDAR parameters
DEFAULT_LIDAR_CHANNELS = 64 # Moderate high-density
DEFAULT_LIDAR_RANGE = 50.0  # meters
DEFAULT_LIDAR_POINTS_PER_SECOND = 150000 # Moderate high-density
DEFAULT_LIDAR_ROTATION_FREQUENCY = 10.0 # Standard Hz
DEFAULT_LIDAR_UPPER_FOV = 15.0
DEFAULT_LIDAR_LOWER_FOV = -25.0
DEFAULT_PROCESSED_LIDAR_NUM_POINTS = 360 # Number of points after processing
DEFAULT_LIDAR_SENSOR_TICK = "0.05" # Default to match typical environment timestep
DEFAULT_LIDAR_NUM_POINTS_PROCESSED = 720 # Match observation space default if used

# Performance configuration
USE_GPU_ACCELERATION = CUPY_AVAILABLE
USE_ASYNC_PROCESSING = True
MAX_WORKER_THREADS = 2

# Thread pool for async operations
_lidar_thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS)

def get_lidar_observation_space(num_points=DEFAULT_PROCESSED_LIDAR_NUM_POINTS):
    """Returns the gymnasium.spaces.Box for processed LIDAR data."""
    return spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(num_points, 3),  # N_points x (x,y,z)
        dtype=np.float32
    )

def get_semantic_lidar_observation_space(num_points=DEFAULT_PROCESSED_LIDAR_NUM_POINTS):
    """Returns the gymnasium.spaces.Box for processed Semantic LIDAR data."""
    # Shape: (N_points, 4) where 4 = (x, y, z, object_tag)
    # object_tag is uint32, but for observation space, float32 is often used and then processed by model.
    # Alternatively, could have separate spaces for points and tags if model handles it differently.
    return spaces.Box(
        low=-np.inf, high=np.inf, # x,y,z can be anything, tag will be positive integer
        shape=(num_points, 4), 
        dtype=np.float32 # Using float32 for simplicity in DL models, tags can be cast/embedded
    )

# Numba JIT optimized functions for LIDAR processing
if numba:
    @jit(nopython=True, cache=True, parallel=True)
    def _subsample_points_numba(points: np.ndarray, target_size: int, indices: np.ndarray) -> np.ndarray:
        """Optimized point subsampling using Numba JIT."""
        result = np.empty((target_size, points.shape[1]), dtype=np.float32)
        for i in numba.prange(target_size):
            for j in range(points.shape[1]):
                result[i, j] = points[indices[i], j]
        return result

    @jit(nopython=True, cache=True)
    def _filter_points_by_distance_numba(points: np.ndarray, max_distance: float) -> np.ndarray:
        """Filter points by distance using Numba JIT."""
        valid_indices = []
        for i in range(points.shape[0]):
            x, y, z = points[i, 0], points[i, 1], points[i, 2]
            distance = np.sqrt(x*x + y*y + z*z)
            if distance <= max_distance:
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            return np.empty((0, points.shape[1]), dtype=np.float32)
        
        result = np.empty((len(valid_indices), points.shape[1]), dtype=np.float32)
        for i in range(len(valid_indices)):
            for j in range(points.shape[1]):
                result[i, j] = points[valid_indices[i], j]
        return result

    @jit(nopython=True, cache=True)
    def _normalize_points_numba(points: np.ndarray, max_range: float) -> np.ndarray:
        """Normalize point coordinates using Numba JIT."""
        normalized = np.empty_like(points)
        for i in range(points.shape[0]):
            for j in range(3):  # Only normalize x, y, z coordinates
                normalized[i, j] = points[i, j] / max_range
            # Copy remaining dimensions (like semantic tags) unchanged
            for j in range(3, points.shape[1]):
                normalized[i, j] = points[i, j]
        return normalized

# GPU-accelerated functions using CuPy
def _process_lidar_gpu(points: np.ndarray, num_points_to_keep: int, max_distance: Optional[float] = None) -> np.ndarray:
    """GPU-accelerated LIDAR processing using CuPy."""
    if not CUPY_AVAILABLE:
        return _process_lidar_fallback(points, num_points_to_keep, max_distance)
    
    try:
        # Transfer to GPU
        points_gpu = cp.asarray(points)
        
        # Filter by distance if specified
        if max_distance is not None:
            distances = cp.sqrt(cp.sum(points_gpu[:, :3]**2, axis=1))
            valid_mask = distances <= max_distance
            points_gpu = points_gpu[valid_mask]
        
        # Subsample or pad
        if points_gpu.shape[0] > num_points_to_keep:
            # Random subsampling on GPU
            indices = cp.random.choice(points_gpu.shape[0], num_points_to_keep, replace=False)
            processed_points = points_gpu[indices]
        elif points_gpu.shape[0] < num_points_to_keep:
            # Pad with zeros
            pad_width = num_points_to_keep - points_gpu.shape[0]
            zeros_pad = cp.zeros((pad_width, points_gpu.shape[1]), dtype=cp.float32)
            processed_points = cp.concatenate([points_gpu, zeros_pad], axis=0)
        else:
            processed_points = points_gpu
        
        # Transfer back to CPU
        return cp.asnumpy(processed_points).astype(np.float32)
        
    except Exception as e:
        logger.warning(f"GPU LIDAR processing failed, falling back to CPU: {e}")
        return _process_lidar_fallback(points, num_points_to_keep, max_distance)

def _process_semantic_lidar_gpu(points: np.ndarray, num_points_to_keep: int) -> np.ndarray:
    """GPU-accelerated semantic LIDAR processing using CuPy."""
    if not CUPY_AVAILABLE:
        return _process_semantic_lidar_fallback(points, num_points_to_keep)
    
    try:
        # Transfer to GPU
        points_gpu = cp.asarray(points)
        
        # Subsample or pad
        if points_gpu.shape[0] > num_points_to_keep:
            indices = cp.random.choice(points_gpu.shape[0], num_points_to_keep, replace=False)
            processed_points = points_gpu[indices]
        elif points_gpu.shape[0] < num_points_to_keep:
            pad_width = num_points_to_keep - points_gpu.shape[0]
            zeros_pad = cp.zeros((pad_width, 4), dtype=cp.float32)
            processed_points = cp.concatenate([points_gpu, zeros_pad], axis=0)
        else:
            processed_points = points_gpu
        
        return cp.asnumpy(processed_points).astype(np.float32)
        
    except Exception as e:
        logger.warning(f"GPU semantic LIDAR processing failed, falling back to CPU: {e}")
        return _process_semantic_lidar_fallback(points, num_points_to_keep)

# Fallback functions for when optimizations are not available
def _process_lidar_fallback(points: np.ndarray, num_points_to_keep: int, max_distance: Optional[float] = None) -> np.ndarray:
    """Fallback LIDAR processing without optimizations."""
    if max_distance is not None:
        distances = np.sqrt(np.sum(points[:, :3]**2, axis=1))
        points = points[distances <= max_distance]
    
    if points.shape[0] == 0:
        return np.zeros((num_points_to_keep, points.shape[1]), dtype=np.float32)
    
    if points.shape[0] > num_points_to_keep:
        indices = np.random.choice(points.shape[0], num_points_to_keep, replace=False)
        processed_points = points[indices]
    elif points.shape[0] < num_points_to_keep:
        pad_width = num_points_to_keep - points.shape[0]
        processed_points = np.pad(points, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
    else:
        processed_points = points
        
    return processed_points.astype(np.float32)

def _process_semantic_lidar_fallback(points: np.ndarray, num_points_to_keep: int) -> np.ndarray:
    """Fallback semantic LIDAR processing without optimizations."""
    if points.shape[0] == 0:
        return np.zeros((num_points_to_keep, 4), dtype=np.float32)
    
    if points.shape[0] > num_points_to_keep:
        indices = np.random.choice(points.shape[0], num_points_to_keep, replace=False)
        processed_points = points[indices]
    elif points.shape[0] < num_points_to_keep:
        pad_width = num_points_to_keep - points.shape[0]
        processed_points = np.pad(points, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
    else:
        processed_points = points
        
    return processed_points.astype(np.float32)

# Thread pool processing functions
def _process_lidar_with_background_optimization(points: np.ndarray, num_points_to_keep: int) -> np.ndarray:
    """Process LIDAR data with best available optimization in background thread."""
    if numba:
        return _process_lidar_with_numba(points, num_points_to_keep)
    elif USE_GPU_ACCELERATION:
        return _process_lidar_gpu(points, num_points_to_keep, None)
    else:
        return _process_lidar_fallback(points, num_points_to_keep, None)

def _process_lidar_with_numba(points: np.ndarray, num_points_to_keep: int) -> np.ndarray:
    """Process LIDAR data using Numba optimizations."""
    if points.shape[0] == 0:
        return np.zeros((num_points_to_keep, points.shape[1]), dtype=np.float32)
    
    if points.shape[0] > num_points_to_keep:
        indices = np.random.choice(points.shape[0], num_points_to_keep, replace=False)
        if numba:
            return _subsample_points_numba(points, num_points_to_keep, indices)
        else:
            return points[indices].astype(np.float32)
    elif points.shape[0] < num_points_to_keep:
        pad_width = num_points_to_keep - points.shape[0]
        return np.pad(points, ((0, pad_width), (0, 0)), 'constant', constant_values=0).astype(np.float32)
    else:
        return points.astype(np.float32)

def _setup_lidar_blueprint(blueprint_library: carla.BlueprintLibrary, lidar_config: dict, is_semantic: bool) -> carla.ActorBlueprint:
    """Helper to configure a LIDAR blueprint (standard or semantic)."""
    sensor_type = 'sensor.lidar.ray_cast_semantic' if is_semantic else 'sensor.lidar.ray_cast'
    lidar_bp = blueprint_library.find(sensor_type)

    cfg = lidar_config if lidar_config else {}
    # Get attributes from config dict, falling back to defaults if not present
    lidar_bp.set_attribute('channels', str(cfg.get('channels', DEFAULT_LIDAR_CHANNELS)))
    lidar_bp.set_attribute('range', str(cfg.get('range', DEFAULT_LIDAR_RANGE)))
    lidar_bp.set_attribute('points_per_second', str(cfg.get('points_per_second', DEFAULT_LIDAR_POINTS_PER_SECOND)))
    lidar_bp.set_attribute('rotation_frequency', str(cfg.get('rotation_frequency', DEFAULT_LIDAR_ROTATION_FREQUENCY)))
    lidar_bp.set_attribute('upper_fov', str(cfg.get('upper_fov', DEFAULT_LIDAR_UPPER_FOV)))
    lidar_bp.set_attribute('lower_fov', str(cfg.get('lower_fov', DEFAULT_LIDAR_LOWER_FOV)))
    # sensor_tick is often passed directly in lidar_config by SensorManager/CarlaEnv using _get_scaled_sensor_tick
    if 'sensor_tick' in cfg:
        lidar_bp.set_attribute('sensor_tick', str(cfg['sensor_tick']))
    else:
        lidar_bp.set_attribute('sensor_tick', DEFAULT_LIDAR_SENSOR_TICK) # Fallback
    
    # Other attributes like dropoff_general_rate, dropoff_intensity_limit, dropoff_zero_intensity can be added here if needed
    # lidar_bp.set_attribute('dropoff_general_rate', str(cfg.get('dropoff_general_rate', 0.0)))
    # lidar_bp.set_attribute('dropoff_intensity_limit', str(cfg.get('dropoff_intensity_limit', 0.0)))
    # lidar_bp.set_attribute('dropoff_zero_intensity', str(cfg.get('dropoff_zero_intensity', 0.0)))
    return lidar_bp

def setup_lidar_sensor(world, vehicle, carla_env_weak_ref, lidar_config=None, transform=None,
                         sensor_key: str = 'lidar'): # Added sensor_key
    """Spawns and configures a (standard) Ray-Cast LIDAR sensor."""
    blueprint_library = world.get_blueprint_library()
    lidar_bp = _setup_lidar_blueprint(blueprint_library, lidar_config, is_semantic=False)

    if transform is None:
        transform = carla.Transform(carla.Location(x=0.0, z=2.5)) # Default on the roof

    lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    logger.debug(f"Spawned LIDAR Sensor: {lidar.id} at {transform} with range {lidar_bp.get_attribute('range')}m")

    def callback(data): # carla.LidarMeasurement
        me = carla_env_weak_ref()
        if me:
            me.latest_sensor_data[sensor_key] = data # Use provided sensor_key
    lidar.listen(callback)
    return lidar

def setup_semantic_lidar_sensor(world, vehicle, carla_env_weak_ref, semantic_lidar_config=None, transform=None, 
                                sensor_key: str = 'semantic_lidar'):
    """Spawns and configures a Semantic LIDAR sensor."""
    blueprint_library = world.get_blueprint_library()
    lidar_bp = _setup_lidar_blueprint(blueprint_library, semantic_lidar_config, is_semantic=True)

    if transform is None:
        transform = carla.Transform(carla.Location(x=0.0, z=2.5)) # Default on the roof

    semantic_lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    logger.debug(f"Spawned Semantic LIDAR Sensor ({sensor_key}): {semantic_lidar.id} at {transform} with range {lidar_bp.get_attribute('range')}m")

    def callback(data): # carla.SemanticLidarMeasurement
        me = carla_env_weak_ref()
        if me:
            me.latest_sensor_data[sensor_key] = data # Use provided sensor_key
    semantic_lidar.listen(callback)
    return semantic_lidar

def process_lidar_data(lidar_measurement: carla.LidarMeasurement, num_points_to_keep: int) -> np.ndarray:
    """Processes raw LIDAR data into a fixed-size NumPy array (simplified bird's-eye-view representation).
       Points are [x, y, z] in the sensor's local coordinate system.
       This version returns the raw (x,y,z) points up to num_points_to_keep.
    """
    if not lidar_measurement:
        return np.zeros((num_points_to_keep, 3), dtype=np.float32)

    points = np.frombuffer(lidar_measurement.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4)) # x, y, z, intensity
    
    # For observation, we usually care about x,y,z. Intensity can be used for visualization or advanced features.
    points_xyz = points[:, :3] # Get X, Y, Z

    if points_xyz.shape[0] == 0:
        return np.zeros((num_points_to_keep, 3), dtype=np.float32)
    
    # Use optimized processing
    if numba:
        return _process_lidar_with_numba(points_xyz, num_points_to_keep)
    elif USE_GPU_ACCELERATION:
        return _process_lidar_gpu(points_xyz, num_points_to_keep)
    else:
        return _process_lidar_fallback(points_xyz, num_points_to_keep)

def process_semantic_lidar_data(semantic_lidar_measurement: carla.SemanticLidarMeasurement, num_points_to_keep: int) -> np.ndarray:
    """Processes raw Semantic LIDAR data into a fixed-size NumPy array.
       Returns [x, y, z, object_tag] for each point.
    """
    if not semantic_lidar_measurement:
        return np.zeros((num_points_to_keep, 4), dtype=np.float32) # x,y,z,tag

    # Data contains (x,y,z, cos_angle, obj_idx, obj_tag)
    # We want x,y,z, obj_tag
    data = np.frombuffer(semantic_lidar_measurement.raw_data, dtype=np.dtype([
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('CosAngle', 'f4'), ('ObjIdx', 'u4'), ('ObjTag', 'u4')
    ]))

    if data.shape[0] == 0:
        return np.zeros((num_points_to_keep, 4), dtype=np.float32)

    # Select x, y, z, and object_tag
    points_xyz_tag = np.stack((data['x'], data['y'], data['z'], data['ObjTag']), axis=-1)

    # Use optimized processing
    if USE_GPU_ACCELERATION:
        return _process_semantic_lidar_gpu(points_xyz_tag, num_points_to_keep)
    else:
        return _process_semantic_lidar_fallback(points_xyz_tag, num_points_to_keep)

def _parse_lidar_cb_async(weak_self, lidar_data: carla.LidarMeasurement, sensor_key: str, num_points_processed: int):
    """Optimized async LIDAR callback with performance enhancements."""
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'):
        return

    try:
        # Store raw data for visualizers
        self.latest_sensor_data[sensor_key + '_raw'] = lidar_data
        
        # Convert to numpy array
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points_xyz = points[:, :3]
        
        # Process based on availability of optimizations
        if USE_ASYNC_PROCESSING:
            # Use thread pool for background processing instead of asyncio
            def process_in_background():
                try:
                    if numba:
                        return _process_lidar_with_numba(points_xyz, num_points_processed)
                    elif USE_GPU_ACCELERATION:
                        return _process_lidar_gpu(points_xyz, num_points_processed, None)
                    else:
                        return _process_lidar_fallback(points_xyz, num_points_processed, None)
                except Exception as e:
                    logger.error(f"Background LIDAR processing failed for {sensor_key}: {e}")
                    return np.zeros((num_points_processed, 3), dtype=np.float32)
            
            # Submit to thread pool and update sensor data when complete
            future = _lidar_thread_pool.submit(process_in_background)
            
            def update_sensor_data(future):
                try:
                    result = future.result()
                    self.latest_sensor_data[sensor_key] = result
                except Exception as e:
                    logger.error(f"Thread pool LIDAR processing failed for {sensor_key}: {e}")
                    self.latest_sensor_data[sensor_key] = np.zeros((num_points_processed, 3), dtype=np.float32)
            
            future.add_done_callback(update_sensor_data)
        else:
            # Synchronous processing with optimizations
            processed_data = process_lidar_data(lidar_data, num_points_processed)
            self.latest_sensor_data[sensor_key] = processed_data
            
    except Exception as e:
        logger.error(f"Error processing LIDAR data for {sensor_key}: {e}")
        # Set fallback data
        self.latest_sensor_data[sensor_key] = np.zeros((num_points_processed, 3), dtype=np.float32)

def _parse_semantic_lidar_cb_async(weak_self, sem_lidar_data: carla.SemanticLidarMeasurement, sensor_key: str, num_points_processed: int):
    """Optimized async semantic LIDAR callback with performance enhancements."""
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'):
        return

    try:
        # Store raw data for visualizers
        self.latest_sensor_data[sensor_key + '_raw'] = sem_lidar_data
        
        # Convert to numpy array
        data = np.frombuffer(sem_lidar_data.raw_data, dtype=np.dtype([
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('CosAngle', 'f4'), ('ObjIdx', 'u4'), ('ObjTag', 'u4')
        ]))
        
        if data.shape[0] > 0:
            points_xyz_tag = np.stack((data['x'], data['y'], data['z'], data['ObjTag']), axis=-1)
        else:
            points_xyz_tag = np.zeros((0, 4), dtype=np.float32)
        
        # Process based on availability of optimizations
        if USE_ASYNC_PROCESSING:
            # Use thread pool for background processing instead of asyncio
            def process_in_background():
                try:
                    if USE_GPU_ACCELERATION:
                        return _process_semantic_lidar_gpu(points_xyz_tag, num_points_processed)
                    else:
                        return _process_semantic_lidar_fallback(points_xyz_tag, num_points_processed)
                except Exception as e:
                    logger.error(f"Background semantic LIDAR processing failed for {sensor_key}: {e}")
                    return np.zeros((num_points_processed, 4), dtype=np.float32)
            
            # Submit to thread pool and update sensor data when complete
            future = _lidar_thread_pool.submit(process_in_background)
            
            def update_sensor_data(future):
                try:
                    result = future.result()
                    self.latest_sensor_data[sensor_key] = result
                except Exception as e:
                    logger.error(f"Thread pool semantic LIDAR processing failed for {sensor_key}: {e}")
                    self.latest_sensor_data[sensor_key] = np.zeros((num_points_processed, 4), dtype=np.float32)
            
            future.add_done_callback(update_sensor_data)
        else:
            # Synchronous processing with optimizations
            processed_data = process_semantic_lidar_data(sem_lidar_data, num_points_processed)
            self.latest_sensor_data[sensor_key] = processed_data
            
    except Exception as e:
        logger.error(f"Error processing semantic LIDAR data for {sensor_key}: {e}")
        # Set fallback data
        self.latest_sensor_data[sensor_key] = np.zeros((num_points_processed, 4), dtype=np.float32)

# Legacy callbacks for backward compatibility
def _parse_lidar_cb(weak_self, lidar_data: carla.LidarMeasurement, sensor_key: str, num_points_processed: int):
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): 
        return
    self.latest_sensor_data[sensor_key + '_raw'] = lidar_data 
    processed_numpy_array = process_lidar_data(lidar_data, num_points_processed)
    self.latest_sensor_data[sensor_key] = processed_numpy_array

def _parse_semantic_lidar_cb(weak_self, sem_lidar_data: carla.SemanticLidarMeasurement, sensor_key: str, num_points_processed: int):
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): 
        return
    self.latest_sensor_data[sensor_key + '_raw'] = sem_lidar_data
    processed_numpy_array = process_semantic_lidar_data(sem_lidar_data, num_points_processed)
    self.latest_sensor_data[sensor_key] = processed_numpy_array

# Enhanced setup functions with optimized callbacks
def setup_lidar_sensor_optimized(world, vehicle, env_ref, lidar_config, transform, sensor_key='lidar'):
    """Setup LIDAR sensor with optimized callback."""
    lidar_bp = _setup_lidar_blueprint(world.get_blueprint_library(), lidar_config, is_semantic=False)
    lidar_sensor = world.try_spawn_actor(lidar_bp, transform, attach_to=vehicle)
    if lidar_sensor:
        num_processed = lidar_config.get('num_points_processed', DEFAULT_LIDAR_NUM_POINTS_PROCESSED)
        lidar_sensor.listen(lambda data: _parse_lidar_cb_async(env_ref, data, sensor_key, num_processed))
    return lidar_sensor

def setup_semantic_lidar_sensor_optimized(world, vehicle, env_ref, semantic_lidar_config, transform, sensor_key='semantic_lidar'):
    """Setup semantic LIDAR sensor with optimized callback."""
    lidar_bp = _setup_lidar_blueprint(world.get_blueprint_library(), semantic_lidar_config, is_semantic=True)
    lidar_sensor = world.try_spawn_actor(lidar_bp, transform, attach_to=vehicle)
    if lidar_sensor:
        num_processed = semantic_lidar_config.get('num_points_processed', DEFAULT_LIDAR_NUM_POINTS_PROCESSED)
        lidar_sensor.listen(lambda data: _parse_semantic_lidar_cb_async(env_ref, data, sensor_key, num_processed))
    return lidar_sensor

# Performance monitoring
def get_lidar_performance_stats() -> dict:
    """Get performance statistics for LIDAR processing."""
    return {
        'numba_available': numba,
        'cupy_available': CUPY_AVAILABLE,
        'gpu_acceleration': USE_GPU_ACCELERATION,
        'async_processing': USE_ASYNC_PROCESSING,
        'max_worker_threads': MAX_WORKER_THREADS
    } 