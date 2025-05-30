# Handles RGB, Depth, and Semantic Segmentation cameras with performance optimizations

import carla
import numpy as np
import weakref
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from typing import Optional, Tuple, Callable, Any
import time

# Performance optimization imports
try:
    import numba
    from numba import jit, cuda
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba not available. Install numba for JIT compilation optimizations.")

try:
    import cupy as cp
    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available. Install cupy for GPU acceleration.")

from gymnasium import spaces

# Configure a logger for this handler
logger = logging.getLogger(__name__)

# Default camera parameters (can be overridden)
DEFAULT_CAMERA_IMAGE_WIDTH = 84
DEFAULT_CAMERA_IMAGE_HEIGHT = 84
DEFAULT_CAMERA_FOV = 90.0

# Performance configuration
USE_GPU_ACCELERATION = CUPY_AVAILABLE
USE_ASYNC_PROCESSING = True
MAX_WORKER_THREADS = 4

# Thread pool for async operations
_thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS)

def get_camera_observation_spaces(
    image_width=DEFAULT_CAMERA_IMAGE_WIDTH, 
    image_height=DEFAULT_CAMERA_IMAGE_HEIGHT
):
    """
    Returns an OrderedDict of gymnasium.spaces.Box for RGB, Depth, and Semantic Segmentation cameras.
    """
    obs_spaces = OrderedDict()
    obs_spaces['rgb_camera'] = spaces.Box(
        low=0, high=255,
        shape=(3, image_height, image_width),  # C, H, W
        dtype=np.uint8
    )
    obs_spaces['depth_camera'] = spaces.Box(
        low=0.0, high=1.0,  # Normalized depth
        shape=(1, image_height, image_width),  # C, H, W (single channel)
        dtype=np.float32
    )
    obs_spaces['semantic_camera'] = spaces.Box(
        low=0, high=28,  # Max 28 classes in CARLA (check documentation for specific town/version)
        shape=(1, image_height, image_width),  # C, H, W (single channel with class labels)
        dtype=np.uint8
    )
    return obs_spaces

# Numba JIT optimized functions for image processing
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True, parallel=True)
    def _process_rgb_numba(bgra_array: np.ndarray) -> np.ndarray:
        """Optimized RGB conversion using Numba JIT."""
        height, width = bgra_array.shape[:2]
        rgb_array = np.empty((3, height, width), dtype=np.uint8)
        
        for i in numba.prange(height):
            for j in range(width):
                # BGRA to RGB conversion
                rgb_array[0, i, j] = bgra_array[i, j, 2]  # R
                rgb_array[1, i, j] = bgra_array[i, j, 1]  # G
                rgb_array[2, i, j] = bgra_array[i, j, 0]  # B
        
        return rgb_array

    @jit(nopython=True, cache=True, parallel=True)
    def _process_depth_numba(bgra_array: np.ndarray, max_depth: float = 100.0) -> np.ndarray:
        """Optimized depth conversion using Numba JIT."""
        height, width = bgra_array.shape[:2]
        depth_array = np.empty((1, height, width), dtype=np.float32)
        
        for i in numba.prange(height):
            for j in range(width):
                # CARLA depth encoding: R + G*256 + B*256*256
                b = float(bgra_array[i, j, 0])
                g = float(bgra_array[i, j, 1])
                r = float(bgra_array[i, j, 2])
                
                depth_encoded = r + g * 256.0 + b * 256.0 * 256.0
                depth_meters = depth_encoded / 16777215.0 * 1000.0
                normalized_depth = min(depth_meters / max_depth, 1.0)
                
                depth_array[0, i, j] = normalized_depth
        
        return depth_array

    @jit(nopython=True, cache=True, parallel=True)
    def _process_semantic_numba(bgra_array: np.ndarray) -> np.ndarray:
        """Optimized semantic segmentation processing using Numba JIT."""
        height, width = bgra_array.shape[:2]
        semantic_array = np.empty((1, height, width), dtype=np.uint8)
        
        for i in numba.prange(height):
            for j in range(width):
                # R channel contains semantic labels
                semantic_array[0, i, j] = bgra_array[i, j, 2]
        
        return semantic_array

# GPU-accelerated functions using CuPy
def _process_rgb_gpu(bgra_array: np.ndarray) -> np.ndarray:
    """GPU-accelerated RGB conversion using CuPy."""
    if not CUPY_AVAILABLE:
        return _process_rgb_fallback(bgra_array)
    
    try:
        # Transfer to GPU
        bgra_gpu = cp.asarray(bgra_array)
        
        # Convert BGRA to RGB
        rgb_gpu = cp.empty((3, bgra_gpu.shape[0], bgra_gpu.shape[1]), dtype=cp.uint8)
        rgb_gpu[0] = bgra_gpu[:, :, 2]  # R
        rgb_gpu[1] = bgra_gpu[:, :, 1]  # G
        rgb_gpu[2] = bgra_gpu[:, :, 0]  # B
        
        # Transfer back to CPU
        return cp.asnumpy(rgb_gpu)
    except Exception as e:
        logger.warning(f"GPU RGB processing failed, falling back to CPU: {e}")
        return _process_rgb_fallback(bgra_array)

def _process_depth_gpu(bgra_array: np.ndarray, max_depth: float = 100.0) -> np.ndarray:
    """GPU-accelerated depth conversion using CuPy."""
    if not CUPY_AVAILABLE:
        return _process_depth_fallback(bgra_array, max_depth)
    
    try:
        # Transfer to GPU
        bgra_gpu = cp.asarray(bgra_array, dtype=cp.float32)
        
        # Extract channels
        b = bgra_gpu[:, :, 0]
        g = bgra_gpu[:, :, 1]
        r = bgra_gpu[:, :, 2]
        
        # CARLA depth encoding
        depth_encoded = r + g * 256.0 + b * 256.0 * 256.0
        depth_meters = depth_encoded / 16777215.0 * 1000.0
        normalized_depth = cp.clip(depth_meters / max_depth, 0.0, 1.0)
        
        # Reshape and transfer back
        result = cp.expand_dims(normalized_depth, axis=0)
        return cp.asnumpy(result).astype(np.float32)
    except Exception as e:
        logger.warning(f"GPU depth processing failed, falling back to CPU: {e}")
        return _process_depth_fallback(bgra_array, max_depth)

def _process_semantic_gpu(bgra_array: np.ndarray) -> np.ndarray:
    """GPU-accelerated semantic segmentation processing using CuPy."""
    if not CUPY_AVAILABLE:
        return _process_semantic_fallback(bgra_array)
    
    try:
        # Transfer to GPU
        bgra_gpu = cp.asarray(bgra_array)
        
        # Extract R channel (semantic labels)
        semantic_gpu = bgra_gpu[:, :, 2]
        result = cp.expand_dims(semantic_gpu, axis=0)
        
        # Transfer back to CPU
        return cp.asnumpy(result).astype(np.uint8)
    except Exception as e:
        logger.warning(f"GPU semantic processing failed, falling back to CPU: {e}")
        return _process_semantic_fallback(bgra_array)

# Fallback functions for when optimizations are not available
def _process_rgb_fallback(bgra_array: np.ndarray) -> np.ndarray:
    """Fallback RGB conversion without optimizations."""
    rgb_hwc = bgra_array[:, :, :3][:, :, ::-1]  # BGRA to RGB
    return np.transpose(rgb_hwc, (2, 0, 1)).astype(np.uint8)

def _process_depth_fallback(bgra_array: np.ndarray, max_depth: float = 100.0) -> np.ndarray:
    """Fallback depth conversion without optimizations."""
    b_channel = bgra_array[:, :, 0].astype(np.float32)
    g_channel = bgra_array[:, :, 1].astype(np.float32)
    r_channel = bgra_array[:, :, 2].astype(np.float32)
    
    depth_encoded_val = r_channel + g_channel * 256.0 + b_channel * 256.0 * 256.0
    depth_meters = depth_encoded_val / 16777215.0 * 1000.0
    normalized_depth = np.clip(depth_meters / max_depth, 0.0, 1.0)
    
    return np.reshape(normalized_depth.astype(np.float32), (1, bgra_array.shape[0], bgra_array.shape[1]))

def _process_semantic_fallback(bgra_array: np.ndarray) -> np.ndarray:
    """Fallback semantic segmentation processing without optimizations."""
    labels_hw = bgra_array[:, :, 2]  # R channel contains semantic labels
    return np.reshape(labels_hw.astype(np.uint8), (1, bgra_array.shape[0], bgra_array.shape[1]))

# Thread pool processing functions
def _process_image_with_background_optimization(bgra_array: np.ndarray, image_type: str) -> np.ndarray:
    """Process image data with best available optimization in background thread."""
    return _process_image_sync(bgra_array, image_type)

def _setup_camera_blueprint(blueprint_library, sensor_type, image_width, image_height, fov, sensor_tick):
    """Helper to configure a camera blueprint."""
    camera_bp = blueprint_library.find(sensor_type)
    camera_bp.set_attribute('image_size_x', str(image_width))
    camera_bp.set_attribute('image_size_y', str(image_height))
    camera_bp.set_attribute('fov', str(fov))
    camera_bp.set_attribute('sensor_tick', str(sensor_tick))
    return camera_bp

def _parse_image_cb(weak_self, image, sensor_key):
    """Legacy callback for backward compatibility."""
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): 
        return
    
    # Convert CARLA image to NumPy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3][:, :, ::-1]  # BGRA to RGB
    
    # Transpose to CHW format
    processed_image = np.transpose(array, (2, 0, 1)).astype(np.uint8)
    self.latest_sensor_data[sensor_key] = processed_image

# High-performance async callback
def _parse_camera_image_cb_async(weak_self, image: carla.Image, sensor_key: str, image_type: str):
    """Optimized async camera callback with performance enhancements."""
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'):
        return

    try:
        # Convert raw data to numpy array
        bgra_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            (image.height, image.width, 4)
        )
        
        # Process based on availability of optimizations
        if USE_ASYNC_PROCESSING:
            # Use thread pool for background processing instead of asyncio
            def process_in_background():
                try:
                    return _process_image_sync(bgra_array, image_type)
                except Exception as e:
                    logger.error(f"Background image processing failed for {sensor_key}: {e}")
                    # Return appropriate fallback data
                    if image_type == 'rgb':
                        fallback_shape = (3, bgra_array.shape[0], bgra_array.shape[1])
                        fallback_dtype = np.uint8
                    elif image_type == 'depth':
                        fallback_shape = (1, bgra_array.shape[0], bgra_array.shape[1])
                        fallback_dtype = np.float32
                    else:  # semantic
                        fallback_shape = (1, bgra_array.shape[0], bgra_array.shape[1])
                        fallback_dtype = np.uint8
                    return np.zeros(fallback_shape, dtype=fallback_dtype)
            
            # Submit to thread pool and update sensor data when complete
            future = _thread_pool.submit(process_in_background)
            
            def update_sensor_data(future):
                try:
                    result = future.result()
                    self.latest_sensor_data[sensor_key] = result
                except Exception as e:
                    logger.error(f"Thread pool image processing failed for {sensor_key}: {e}")
                    # Set fallback data
                    if image_type == 'rgb':
                        fallback_shape = (3, bgra_array.shape[0], bgra_array.shape[1])
                        fallback_dtype = np.uint8
                    elif image_type == 'depth':
                        fallback_shape = (1, bgra_array.shape[0], bgra_array.shape[1])
                        fallback_dtype = np.float32
                    else:  # semantic
                        fallback_shape = (1, bgra_array.shape[0], bgra_array.shape[1])
                        fallback_dtype = np.uint8
                    self.latest_sensor_data[sensor_key] = np.zeros(fallback_shape, dtype=fallback_dtype)
            
            future.add_done_callback(update_sensor_data)
        else:
            # Synchronous processing with optimizations
            processed_data = _process_image_sync(bgra_array, image_type)
            self.latest_sensor_data[sensor_key] = processed_data
            
    except Exception as e:
        logger.error(f"Error processing {image_type} camera data for {sensor_key}: {e}")
        # Set fallback data
        if image_type == 'rgb':
            fallback_shape = (3, image.height, image.width)
            fallback_dtype = np.uint8
        elif image_type == 'depth':
            fallback_shape = (1, image.height, image.width)
            fallback_dtype = np.float32
        else:  # semantic
            fallback_shape = (1, image.height, image.width)
            fallback_dtype = np.uint8
            
        self.latest_sensor_data[sensor_key] = np.zeros(fallback_shape, dtype=fallback_dtype)

def _process_image_sync(bgra_array: np.ndarray, image_type: str) -> np.ndarray:
    """Synchronous image processing with optimizations."""
    if image_type == 'rgb':
        if NUMBA_AVAILABLE:
            return _process_rgb_numba(bgra_array)
        elif USE_GPU_ACCELERATION:
            return _process_rgb_gpu(bgra_array)
        else:
            return _process_rgb_fallback(bgra_array)
    
    elif image_type == 'depth':
        if NUMBA_AVAILABLE:
            return _process_depth_numba(bgra_array)
        elif USE_GPU_ACCELERATION:
            return _process_depth_gpu(bgra_array)
        else:
            return _process_depth_fallback(bgra_array)
    
    elif image_type == 'semantic':
        if NUMBA_AVAILABLE:
            return _process_semantic_numba(bgra_array)
        elif USE_GPU_ACCELERATION:
            return _process_semantic_gpu(bgra_array)
        else:
            return _process_semantic_fallback(bgra_array)
    
    else:
        raise ValueError(f"Unknown image type: {image_type}")

def setup_rgb_camera(world, vehicle, env_ref, image_width, image_height, fov, sensor_tick, transform, sensor_key):
    """Spawns and configures an RGB camera with performance optimizations."""
    camera_bp = _setup_camera_blueprint(world.get_blueprint_library(), 'sensor.camera.rgb',
                                     image_width, image_height, fov, sensor_tick)
    
    if transform is None:
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    
    camera = world.try_spawn_actor(camera_bp, transform, attach_to=vehicle)
    if camera:
        camera.listen(lambda image: _parse_camera_image_cb_async(env_ref, image, sensor_key, 'rgb'))
    logger.debug(f"Spawned RGB Camera ({sensor_key}): {camera.id} at {transform}")
    return camera

def setup_depth_camera(world, vehicle, env_ref, image_width, image_height, fov, sensor_tick, transform, sensor_key):
    """Spawns and configures a Depth camera with performance optimizations."""
    camera_bp = _setup_camera_blueprint(world.get_blueprint_library(), 'sensor.camera.depth',
                                     image_width, image_height, fov, sensor_tick)
    if transform is None:
        transform = carla.Transform(carla.Location(x=1.5, y=0.1, z=2.4))
    
    camera = world.try_spawn_actor(camera_bp, transform, attach_to=vehicle)
    if camera:
        camera.listen(lambda image: _parse_camera_image_cb_async(env_ref, image, sensor_key, 'depth'))
    logger.debug(f"Spawned Depth Camera ({sensor_key}): {camera.id} at {transform}")
    return camera

def setup_semantic_segmentation_camera(world, vehicle, env_ref, image_width, image_height, fov, sensor_tick, transform, sensor_key):
    """Spawns and configures a Semantic Segmentation camera with performance optimizations."""
    camera_bp = _setup_camera_blueprint(world.get_blueprint_library(), 'sensor.camera.semantic_segmentation',
                                     image_width, image_height, fov, sensor_tick)
    if transform is None:
        transform = carla.Transform(carla.Location(x=1.5, y=-0.1, z=2.4))
        
    camera = world.try_spawn_actor(camera_bp, transform, attach_to=vehicle)
    if camera:
        camera.listen(lambda image: _parse_camera_image_cb_async(env_ref, image, sensor_key, 'semantic'))
    logger.debug(f"Spawned Semantic Segmentation Camera ({sensor_key}): {camera.id} at {transform}")
    return camera

# Legacy processing functions maintained for backward compatibility
def process_rgb_camera_data(raw_data, target_shape_chw):
    """Processes raw RGB camera data (carla.Image) to a CHW NumPy array."""
    if raw_data is None:
        return np.zeros(target_shape_chw, dtype=np.uint8)
    
    img_bgra = np.array(raw_data.raw_data).reshape((raw_data.height, raw_data.width, 4))
    img_rgb = img_bgra[:, :, :3][:, :, ::-1]  # BGRA to BGR to RGB
    return np.transpose(img_rgb, (2, 0, 1)).astype(np.uint8)

def process_depth_camera_data(raw_data, target_shape_chw):
    """Processes raw Depth camera data (carla.Image) to a normalized CHW NumPy array."""
    if raw_data is None:
        return np.zeros(target_shape_chw, dtype=np.float32)

    depth_bgra = np.array(raw_data.raw_data).reshape((raw_data.height, raw_data.width, 4))
    array = depth_bgra.astype(np.float32)
    normalized_depth_map = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth_map /= 16777215.0
    
    return normalized_depth_map[np.newaxis, :, :].astype(np.float32)

def process_semantic_camera_data(raw_data, target_shape_chw):
    """Processes raw Semantic Segmentation camera data (carla.Image) to a CHW NumPy array."""
    if raw_data is None:
        return np.zeros(target_shape_chw, dtype=np.uint8)
    
    seg_array_bgra = np.array(raw_data.raw_data).reshape((raw_data.height, raw_data.width, 4))
    return seg_array_bgra[:, :, 2][np.newaxis, :, :].astype(np.uint8)

# Original callback for compatibility
def _parse_camera_image_cb(weak_self, image: carla.Image, sensor_key: str, image_type: str):
    """Original camera callback for backward compatibility."""
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): 
        return

    processed_data = None

    if image_type == 'rgb':
        array_bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        array_rgb_hwc = array_bgra[:, :, :3][:, :, ::-1]
        processed_data = np.transpose(array_rgb_hwc.astype(np.uint8), (2, 0, 1))
    
    elif image_type == 'depth':
        array_bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        
        b_channel = array_bgra[:,:,0].astype(np.float32)
        g_channel = array_bgra[:,:,1].astype(np.float32)
        r_channel = array_bgra[:,:,2].astype(np.float32)
        
        depth_encoded_val = r_channel + g_channel * 256.0 + b_channel * 256.0 * 256.0
        depth_meters = depth_encoded_val / (16777215.0)
        depth_meters = depth_meters * 1000.0

        max_depth_normalization = 100.0
        normalized_depth = np.clip(depth_meters / max_depth_normalization, 0.0, 1.0)
        
        processed_data = np.reshape(normalized_depth.astype(np.float32), (1, image.height, image.width))

    elif image_type == 'semantic':
        array_bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        labels_hw = array_bgra[:, :, 2]
        processed_data = np.reshape(labels_hw.astype(np.uint8), (1, image.height, image.width))
    else:
        env_instance = weak_self()
        log_target = env_instance.logger if env_instance and hasattr(env_instance, 'logger') else logging
        log_target.warning(f"CameraHandler: Unknown image type '{image_type}' for sensor '{sensor_key}'")
        return

    if processed_data is not None:
        self.latest_sensor_data[sensor_key] = processed_data
    else:
        env_instance = weak_self()
        log_target = env_instance.logger if env_instance and hasattr(env_instance, 'logger') else logging
        log_target.warning(f"CameraHandler: Processed data is None for '{sensor_key}'. This should not happen.")

# Performance monitoring
def get_performance_stats() -> dict:
    """Get performance statistics for camera processing."""
    return {
        'numba_available': NUMBA_AVAILABLE,
        'cupy_available': CUPY_AVAILABLE,
        'gpu_acceleration': USE_GPU_ACCELERATION,
        'async_processing': USE_ASYNC_PROCESSING,
        'max_worker_threads': MAX_WORKER_THREADS
    } 