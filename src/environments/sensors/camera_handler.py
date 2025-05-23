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

def _parse_image_cb(weak_self, image, sensor_key):
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): return
    # Convert CARLA image to a NumPy array
    # BGRA to RGB, and then to C, H, W format (PyTorch default)
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  # Take BGR
    array = array[:, :, ::-1] # Convert BGR to RGB
    array = np.array(array, dtype=np.uint8) # Ensure correct dtype
    # Transpose to (C, H, W) if observation space expects it, or keep as (H, W, C)
    # Assuming observation space for cameras expects (C, H, W)
    if array.shape[2] == 3: # Check if it has 3 channels
        processed_image = np.transpose(array, (2, 0, 1))
    else: # Grayscale or other, ensure it meets expected shape (e.g. (1, H, W) for depth)
        # For depth/semantic, specific conversion might be needed if not already handled by raw_data format
        if sensor_key == 'depth_camera' or sensor_key == 'semantic_camera':
            # Example: if depth is float, it might already be H,W. Add channel dim.
            # array = np.reshape(array, (1, image.height, image.width))
            # This part needs to be consistent with how get_camera_observation_spaces defines shapes.
            # For now, let's assume the initial processing in callback gives H,W,C or H,W
            # and the get_observation will handle final shaping if needed, but dtype must be correct.
            # The main point is to store a numpy array, not a carla.Image.
            if sensor_key == 'depth_camera':
                 # Depth camera raw data is often float32. Here it's being converted from BGRA like.
                 # This needs to be fixed to handle actual depth data correctly.
                 # For now, let's make it a single channel (H, W) -> (1, H, W)
                 # A proper depth conversion would be: (assuming image.convert(carla.ColorConverter.Depth) was used implicitly)
                 depth_array = np.array(image.raw_data).reshape((image.height, image.width, 4))[:,:,0] # Assuming R channel for depth
                 processed_image = depth_array.astype(np.float32) / 255.0 * 1000.0 # Example scaling
                 processed_image = np.reshape(processed_image, (1, image.height, image.width))
            elif sensor_key == 'semantic_camera':
                 # Semantic data contains the class labels directly in the raw data
                 # DO NOT convert to CityScapesPalette as that destroys the semantic labels!
                 semantic_array = np.array(image.raw_data).reshape((image.height, image.width, 4))[:,:,2] # R channel for semantic labels
                 processed_image = np.reshape(semantic_array.astype(np.uint8), (1, image.height, image.width))
            else: # Default for other single channel images if any, or fallback
                 processed_image = np.transpose(array, (2, 0, 1)) # Should not happen for depth/seg this way
        else:
            processed_image = array # Fallback if not 3 channels

    self.latest_sensor_data[sensor_key] = processed_image

def setup_rgb_camera(world, vehicle, env_ref, image_width, image_height, fov, sensor_tick, transform, sensor_key):
    """Spawns and configures an RGB camera."""
    camera_bp = _setup_camera_blueprint(world.get_blueprint_library(), 'sensor.camera.rgb',
                                     image_width, image_height, fov, sensor_tick)
    
    if transform is None: # Default transform if not provided
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    
    camera = world.try_spawn_actor(camera_bp, transform, attach_to=vehicle)
    if camera:
        camera.listen(lambda image: _parse_camera_image_cb(env_ref, image, sensor_key, 'rgb'))
    logger.debug(f"Spawned RGB Camera ({sensor_key}): {camera.id} at {transform}")
    return camera

def setup_depth_camera(world, vehicle, env_ref, image_width, image_height, fov, sensor_tick, transform, sensor_key):
    """Spawns and configures a Depth camera."""
    camera_bp = _setup_camera_blueprint(world.get_blueprint_library(), 'sensor.camera.depth',
                                     image_width, image_height, fov, sensor_tick)
    if transform is None:
        transform = carla.Transform(carla.Location(x=1.5, y=0.1, z=2.4)) # Slight offset from RGB for clarity
    
    camera = world.try_spawn_actor(camera_bp, transform, attach_to=vehicle)
    if camera:
        camera.listen(lambda image: _parse_camera_image_cb(env_ref, image, sensor_key, 'depth'))
    logger.debug(f"Spawned Depth Camera ({sensor_key}): {camera.id} at {transform}")
    return camera

def setup_semantic_segmentation_camera(world, vehicle, env_ref, image_width, image_height, fov, sensor_tick, transform, sensor_key):
    """Spawns and configures a Semantic Segmentation camera."""
    camera_bp = _setup_camera_blueprint(world.get_blueprint_library(), 'sensor.camera.semantic_segmentation',
                                     image_width, image_height, fov, sensor_tick)
    if transform is None:
        transform = carla.Transform(carla.Location(x=1.5, y=-0.1, z=2.4)) # Slight offset
        
    camera = world.try_spawn_actor(camera_bp, transform, attach_to=vehicle)
    if camera:
        # Semantic data is often converted to CityScapesPalette by the sensor itself before providing raw_data
        camera.listen(lambda image: _parse_camera_image_cb(env_ref, image, sensor_key, 'semantic'))
    logger.debug(f"Spawned Semantic Segmentation Camera ({sensor_key}): {camera.id} at {transform}")
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

# A more robust _parse_image_cb for different camera types:
def _parse_camera_image_cb(weak_self, image: carla.Image, sensor_key: str, image_type: str):
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): return

    processed_data = None

    if image_type == 'rgb':
        array_bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        array_rgb_hwc = array_bgra[:, :, :3][:, :, ::-1]
        processed_data = np.transpose(array_rgb_hwc.astype(np.uint8), (2, 0, 1))
    
    elif image_type == 'depth':
        # Process raw BGRA data from sensor.camera.depth to get linear depth in meters
        # Formula from CARLA docs: Depth = (R + G*256 + B*256*256) / (256^3 - 1) * 1000.0
        # Note: CARLA stores in BGR order in the image data, so array_bgra[:,:,0] is B, [:,:,1] is G, [:,:,2] is R.
        array_bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        
        # Extract B, G, R channels as float for calculation
        b_channel = array_bgra[:,:,0].astype(np.float32)
        g_channel = array_bgra[:,:,1].astype(np.float32)
        r_channel = array_bgra[:,:,2].astype(np.float32)
        
        depth_encoded_val = r_channel + g_channel * 256.0 + b_channel * 256.0 * 256.0
        depth_meters = depth_encoded_val / (16777215.0)  # (256**3 - 1)
        depth_meters = depth_meters * 1000.0 # Scale to meters (assuming sensor range is 1000m)

        # Adjust max_depth_normalization for better visual contrast of relevant distances
        max_depth_normalization = 100.0 # Changed from 1000.0 to 100.0 (or try 50.0)
        normalized_depth = np.clip(depth_meters / max_depth_normalization, 0.0, 1.0)
        
        processed_data = np.reshape(normalized_depth.astype(np.float32), (1, image.height, image.width))

    elif image_type == 'semantic':
        # DO NOT convert to CityScapesPalette - this destroys the semantic labels!
        # The raw semantic data from CARLA already contains the semantic class IDs
        array_bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        
        # For semantic segmentation, the labels are encoded directly in the raw data
        # The R channel contains the semantic class IDs (0-28)
        labels_hw = array_bgra[:, :, 2]  # R channel contains the semantic labels
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