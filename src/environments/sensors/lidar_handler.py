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
DEFAULT_PROCESSED_LIDAR_NUM_POINTS = 720 # Number of points after processing
DEFAULT_LIDAR_SENSOR_TICK = "0.05" # Default to match typical environment timestep

def get_lidar_observation_space(num_points=DEFAULT_PROCESSED_LIDAR_NUM_POINTS):
    """Returns the gymnasium.spaces.Box for processed LIDAR data."""
    return spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(num_points, 3),  # N_points x (x,y,z)
        dtype=np.float32
    )

def setup_lidar_sensor(world, vehicle, carla_env_weak_ref, lidar_config=None, transform=None):
    """Spawns and configures a LIDAR sensor."""
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
    logger.info(f"Spawned LIDAR Sensor: {lidar.id} at {transform} with range {lidar_range}m")

    def callback(data): # carla.LidarMeasurement
        me = carla_env_weak_ref()
        if me:
            me.latest_sensor_data['lidar'] = data
    lidar.listen(callback)
    return lidar

def process_lidar_data(raw_data, num_target_points=DEFAULT_PROCESSED_LIDAR_NUM_POINTS):
    """Processes raw LIDAR data (carla.LidarMeasurement) to a fixed-size NumPy array."""
    target_shape = (num_target_points, 3)
    if raw_data is None:
        return np.zeros(target_shape, dtype=np.float32)

    # Each point in raw_data is a carla.LidarDetection, not carla.Location directly for new versions.
    # carla.LidarDetection has point, intensity.
    # For older versions, it might be an iterable of carla.Location.
    # Assuming carla.LidarMeasurement is iterable and yields objects with a .point attribute (carla.Location)
    # or directly carla.LidarDetection objects.

    points_list = []
    for detection in raw_data:
        points_list.append([detection.point.x, detection.point.y, detection.point.z])
    
    points = np.array(points_list, dtype=np.float32)

    if not points.size: # Empty point cloud
        return np.zeros(target_shape, dtype=np.float32)

    if len(points) > num_target_points:
        # Subsample (randomly or first N). Random is often better.
        indices = np.random.choice(len(points), num_target_points, replace=False)
        processed_points = points[indices, :]
    elif len(points) < num_target_points:
        processed_points = np.zeros(target_shape, dtype=np.float32)
        processed_points[:len(points), :] = points # Pad with zeros
    else: # len(points) == num_target_points
        processed_points = points
    
    return processed_points.astype(np.float32) 