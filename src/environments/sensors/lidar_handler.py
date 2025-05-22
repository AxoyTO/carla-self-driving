# Handles LIDAR sensor setup and data processing 

import carla
import numpy as np
import weakref
import logging

from gymnasium import spaces

# Configure a logger for this handler
logger = logging.getLogger(__name__)

# Default LIDAR parameters
DEFAULT_LIDAR_CHANNELS = 32
DEFAULT_LIDAR_RANGE = 50.0  # meters
DEFAULT_LIDAR_POINTS_PER_SECOND = 120000 
DEFAULT_LIDAR_ROTATION_FREQUENCY = 10.0 # Hz
DEFAULT_LIDAR_UPPER_FOV = 15.0
DEFAULT_LIDAR_LOWER_FOV = -25.0
DEFAULT_PROCESSED_LIDAR_NUM_POINTS = 360 # Number of points after processing
DEFAULT_LIDAR_SENSOR_TICK = "0.05" # Default to match typical environment timestep

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

def setup_lidar_sensor(world, vehicle, carla_env_weak_ref, lidar_config=None, transform=None,
                         sensor_key: str = 'lidar'): # Added sensor_key
    """Spawns and configures a (standard) Ray-Cast LIDAR sensor."""
    blueprint_library = world.get_blueprint_library()
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')

    # Use provided config or defaults
    cfg = lidar_config if lidar_config else {}
    channels = str(cfg.get('channels', DEFAULT_LIDAR_CHANNELS))
    lidar_range = str(cfg.get('range', DEFAULT_LIDAR_RANGE))
    points_per_second = str(cfg.get('points_per_second', DEFAULT_LIDAR_POINTS_PER_SECOND))
    rotation_frequency = str(cfg.get('rotation_frequency', DEFAULT_LIDAR_ROTATION_FREQUENCY))
    upper_fov = str(cfg.get('upper_fov', DEFAULT_LIDAR_UPPER_FOV))
    lower_fov = str(cfg.get('lower_fov', DEFAULT_LIDAR_LOWER_FOV))
    sensor_tick = str(cfg.get('sensor_tick', DEFAULT_LIDAR_SENSOR_TICK))
    # dropoff_general_rate, dropoff_intensity_limit, dropoff_zero_intensity are other params

    lidar_bp.set_attribute('channels', channels)
    lidar_bp.set_attribute('range', lidar_range)
    lidar_bp.set_attribute('points_per_second', points_per_second)
    lidar_bp.set_attribute('rotation_frequency', rotation_frequency)
    lidar_bp.set_attribute('upper_fov', upper_fov)
    lidar_bp.set_attribute('lower_fov', lower_fov)
    lidar_bp.set_attribute('sensor_tick', sensor_tick)
    
    if transform is None:
        transform = carla.Transform(carla.Location(x=0.0, z=2.5)) # Default on the roof

    lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    logger.debug(f"Spawned LIDAR Sensor: {lidar.id} at {transform} with range {lidar_range}m")

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
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast_semantic')

    cfg = semantic_lidar_config if semantic_lidar_config else {}
    # Attributes are mostly the same as standard LIDAR
    channels = str(cfg.get('channels', DEFAULT_LIDAR_CHANNELS))
    lidar_range = str(cfg.get('range', DEFAULT_LIDAR_RANGE))
    points_per_second = str(cfg.get('points_per_second', DEFAULT_LIDAR_POINTS_PER_SECOND))
    rotation_frequency = str(cfg.get('rotation_frequency', DEFAULT_LIDAR_ROTATION_FREQUENCY))
    upper_fov = str(cfg.get('upper_fov', DEFAULT_LIDAR_UPPER_FOV))
    lower_fov = str(cfg.get('lower_fov', DEFAULT_LIDAR_LOWER_FOV))
    sensor_tick = str(cfg.get('sensor_tick', DEFAULT_LIDAR_SENSOR_TICK))

    lidar_bp.set_attribute('channels', channels)
    lidar_bp.set_attribute('range', lidar_range)
    lidar_bp.set_attribute('points_per_second', points_per_second)
    lidar_bp.set_attribute('rotation_frequency', rotation_frequency)
    lidar_bp.set_attribute('upper_fov', upper_fov)
    lidar_bp.set_attribute('lower_fov', lower_fov)
    lidar_bp.set_attribute('sensor_tick', sensor_tick)
    
    if transform is None:
        transform = carla.Transform(carla.Location(x=0.0, z=2.5)) # Default on the roof

    semantic_lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    logger.debug(f"Spawned Semantic LIDAR Sensor ({sensor_key}): {semantic_lidar.id} at {transform} with range {lidar_range}m")

    def callback(data): # carla.SemanticLidarMeasurement
        me = carla_env_weak_ref()
        if me:
            me.latest_sensor_data[sensor_key] = data # Use provided sensor_key
    semantic_lidar.listen(callback)
    return semantic_lidar

def process_lidar_data(raw_data, num_target_points=DEFAULT_PROCESSED_LIDAR_NUM_POINTS):
    """Processes raw Ray-Cast LIDAR data (carla.LidarMeasurement) to a fixed-size NumPy array of (x,y,z)."""
    target_shape = (num_target_points, 3)
    if raw_data is None:
        return np.zeros(target_shape, dtype=np.float32)

    num_raw_points = len(raw_data)
    if num_raw_points == 0:
        return np.zeros(target_shape, dtype=np.float32)

    # Pre-allocate NumPy array
    points = np.empty((num_raw_points, 3), dtype=np.float32)
    for i, detection in enumerate(raw_data):
        points[i, 0] = detection.point.x
        points[i, 1] = detection.point.y
        points[i, 2] = detection.point.z
    
    # points = np.array(points_list, dtype=np.float32) # Old way

    # Subsample or pad as before, operating on the 'points' NumPy array
    if num_raw_points > num_target_points:
        indices = np.random.choice(num_raw_points, num_target_points, replace=False)
        processed_points = points[indices, :]
    elif num_raw_points < num_target_points:
        processed_points = np.zeros(target_shape, dtype=np.float32)
        processed_points[:num_raw_points, :] = points
    else: # len(points) == num_target_points
        processed_points = points
    
    return processed_points.astype(np.float32)

def process_semantic_lidar_data(raw_data: carla.SemanticLidarMeasurement, num_target_points=DEFAULT_PROCESSED_LIDAR_NUM_POINTS):
    """Processes raw Semantic LIDAR data to a fixed-size NumPy array of (x,y,z,object_tag)."""
    target_shape = (num_target_points, 4) # x,y,z,tag
    if raw_data is None:
        return np.zeros(target_shape, dtype=np.float32)

    num_raw_points = len(raw_data)
    if num_raw_points == 0:
        return np.zeros(target_shape, dtype=np.float32)

    # Pre-allocate NumPy array
    data_array = np.empty((num_raw_points, 4), dtype=np.float32)
    for i, detection in enumerate(raw_data):
        data_array[i, 0] = detection.point.x
        data_array[i, 1] = detection.point.y
        data_array[i, 2] = detection.point.z
        data_array[i, 3] = float(detection.object_tag)

    # Subsample or pad as before, operating on the 'data_array' NumPy array
    if num_raw_points > num_target_points:
        indices = np.random.choice(num_raw_points, num_target_points, replace=False)
        processed_data = data_array[indices, :]
    elif num_raw_points < num_target_points:
        processed_data = np.zeros(target_shape, dtype=np.float32)
        processed_data[:num_raw_points, :] = data_array
    else: 
        processed_data = data_array
    
    return processed_data.astype(np.float32) 