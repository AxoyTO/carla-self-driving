# Handles RGB, Depth, and Semantic Segmentation cameras 

import carla
import numpy as np
import weakref
import logging
from collections import OrderedDict

from gymnasium import spaces

# Configure a logger for this handler
logger = logging.getLogger(__name__)

# Default camera parameters (can be overridden)
DEFAULT_CAMERA_IMAGE_WIDTH = 84
DEFAULT_CAMERA_IMAGE_HEIGHT = 84
DEFAULT_CAMERA_FOV = 90.0

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

def _setup_camera_blueprint(blueprint_library, sensor_type, image_width, image_height, fov, sensor_tick):
    """Helper to configure a camera blueprint."""
    camera_bp = blueprint_library.find(sensor_type)
    camera_bp.set_attribute('image_size_x', str(image_width))
    camera_bp.set_attribute('image_size_y', str(image_height))
    camera_bp.set_attribute('fov', str(fov))
    camera_bp.set_attribute('sensor_tick', str(sensor_tick))
    return camera_bp

def setup_rgb_camera(world, vehicle, carla_env_weak_ref,
                     image_width=DEFAULT_CAMERA_IMAGE_WIDTH, image_height=DEFAULT_CAMERA_IMAGE_HEIGHT,
                     fov=DEFAULT_CAMERA_FOV, sensor_tick=0.05, transform=None):
    """Spawns and configures an RGB camera."""
    camera_bp = _setup_camera_blueprint(world.get_blueprint_library(), 'sensor.camera.rgb',
                                     image_width, image_height, fov, sensor_tick)
    
    if transform is None: # Default transform if not provided
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    logger.info(f"Spawned RGB Camera: {camera.id} at {transform}")

    def callback(data):
        me = carla_env_weak_ref()
        if me:
            me.latest_sensor_data['rgb_camera'] = data
    camera.listen(callback)
    return camera

def setup_depth_camera(world, vehicle, carla_env_weak_ref,
                       image_width=DEFAULT_CAMERA_IMAGE_WIDTH, image_height=DEFAULT_CAMERA_IMAGE_HEIGHT,
                       fov=DEFAULT_CAMERA_FOV, sensor_tick=0.05, transform=None):
    """Spawns and configures a Depth camera."""
    camera_bp = _setup_camera_blueprint(world.get_blueprint_library(), 'sensor.camera.depth',
                                     image_width, image_height, fov, sensor_tick)
    if transform is None:
        transform = carla.Transform(carla.Location(x=1.5, y=0.1, z=2.4)) # Slight offset from RGB for clarity
    
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    logger.info(f"Spawned Depth Camera: {camera.id} at {transform}")

    def callback(data):
        me = carla_env_weak_ref()
        if me:
            me.latest_sensor_data['depth_camera'] = data
    camera.listen(callback)
    return camera

def setup_semantic_segmentation_camera(world, vehicle, carla_env_weak_ref,
                                       image_width=DEFAULT_CAMERA_IMAGE_WIDTH, image_height=DEFAULT_CAMERA_IMAGE_HEIGHT,
                                       fov=DEFAULT_CAMERA_FOV, sensor_tick=0.05, transform=None):
    """Spawns and configures a Semantic Segmentation camera."""
    camera_bp = _setup_camera_blueprint(world.get_blueprint_library(), 'sensor.camera.semantic_segmentation',
                                     image_width, image_height, fov, sensor_tick)
    if transform is None:
        transform = carla.Transform(carla.Location(x=1.5, y=-0.1, z=2.4)) # Slight offset
        
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    logger.info(f"Spawned Semantic Segmentation Camera: {camera.id} at {transform}")

    def callback(data):
        me = carla_env_weak_ref()
        if me:
            me.latest_sensor_data['semantic_camera'] = data
    camera.listen(callback)
    return camera

# --- Data Processing Functions ---

def process_rgb_camera_data(raw_data, target_shape_chw):
    """Processes raw RGB camera data (carla.Image) to a CHW NumPy array."""
    if raw_data is None:
        return np.zeros(target_shape_chw, dtype=np.uint8)
    
    img_bgra = np.array(raw_data.raw_data).reshape((raw_data.height, raw_data.width, 4))
    img_rgb = img_bgra[:, :, :3][:, :, ::-1]  # BGRA to BGR to RGB
    # Ensure it matches target shape if resizing was intended to be handled by camera attributes
    # For now, assume raw_data.height/width match target_shape_chw[1]/[2]
    return np.transpose(img_rgb, (2, 0, 1)).astype(np.uint8)

def process_depth_camera_data(raw_data, target_shape_chw):
    """Processes raw Depth camera data (carla.Image) to a normalized CHW NumPy array."""
    if raw_data is None:
        return np.zeros(target_shape_chw, dtype=np.float32)

    # For 'sensor.camera.depth' (non-logarithmic):
    # Raw data is BGRA. Each pixel = R + G*256 + B*256*256. Max value = 256^3 - 1.
    # Depth in meters = (R + G*256 + B*256*256) / (256^3 - 1) * 1000.0 (for 1km range)
    # To get normalized [0,1] depth:
    depth_bgra = np.array(raw_data.raw_data).reshape((raw_data.height, raw_data.width, 4))
    array = depth_bgra.astype(np.float32)
    # Extract BGR channels, then apply the formula. Carla stores depth in BGR order in the image.
    normalized_depth_map = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0]) # BGR to int
    normalized_depth_map /= 16777215.0  # (256^3 - 1), for normalization to [0,1]
    
    return normalized_depth_map[np.newaxis, :, :].astype(np.float32) # Add channel dim

def process_semantic_camera_data(raw_data, target_shape_chw):
    """Processes raw Semantic Segmentation camera data (carla.Image) to a CHW NumPy array."""
    if raw_data is None:
        return np.zeros(target_shape_chw, dtype=np.uint8)
    
    # The R channel of the BGRA image contains the semantic tag
    seg_array_bgra = np.array(raw_data.raw_data).reshape((raw_data.height, raw_data.width, 4))
    # Semantic tag is in the R channel (index 2 in BGRA)
    return seg_array_bgra[:, :, 2][np.newaxis, :, :].astype(np.uint8) # R channel, add channel dim 