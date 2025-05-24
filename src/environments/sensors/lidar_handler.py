# Handles LIDAR sensor setup and data processing 

import carla
import numpy as np
import weakref
import logging

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
    
    # Subsample or pad to ensure a fixed number of points
    if points_xyz.shape[0] > num_points_to_keep:
        # Subsample: take a random subset of points
        indices = np.random.choice(points_xyz.shape[0], num_points_to_keep, replace=False)
        processed_points = points_xyz[indices, :]
    elif points_xyz.shape[0] < num_points_to_keep:
        # Pad with zeros if fewer points than expected
        pad_width = num_points_to_keep - points_xyz.shape[0]
        processed_points = np.pad(points_xyz, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
    else:
        processed_points = points_xyz
        
    return processed_points.astype(np.float32)

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

    if points_xyz_tag.shape[0] > num_points_to_keep:
        indices = np.random.choice(points_xyz_tag.shape[0], num_points_to_keep, replace=False)
        processed_points = points_xyz_tag[indices, :]
    elif points_xyz_tag.shape[0] < num_points_to_keep:
        pad_width = num_points_to_keep - points_xyz_tag.shape[0]
        processed_points = np.pad(points_xyz_tag, ((0, pad_width), (0, 0)), 'constant', constant_values=0)
    else:
        processed_points = points_xyz_tag
        
    return processed_points.astype(np.float32) # ObjTag will be float, agent needs to handle

def _parse_lidar_cb(weak_self, lidar_data: carla.LidarMeasurement, sensor_key: str, num_points_processed: int):
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): return
    # Store the raw data for visualizers that might need it
    self.latest_sensor_data[sensor_key + '_raw'] = lidar_data 
    processed_numpy_array = process_lidar_data(lidar_data, num_points_processed)
    self.latest_sensor_data[sensor_key] = processed_numpy_array

def _parse_semantic_lidar_cb(weak_self, sem_lidar_data: carla.SemanticLidarMeasurement, sensor_key: str, num_points_processed: int):
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): return
    # Store the raw data for visualizers
    self.latest_sensor_data[sensor_key + '_raw'] = sem_lidar_data
    processed_numpy_array = process_semantic_lidar_data(sem_lidar_data, num_points_processed)
    self.latest_sensor_data[sensor_key] = processed_numpy_array

def setup_lidar_sensor(world, vehicle, env_ref, lidar_config, transform, sensor_key='lidar'):
    # ... (blueprint setup using lidar_config) ...
    lidar_bp = _setup_lidar_blueprint(world.get_blueprint_library(), lidar_config, is_semantic=False)
    lidar_sensor = world.try_spawn_actor(lidar_bp, transform, attach_to=vehicle)
    if lidar_sensor:
        num_processed = lidar_config.get('num_points_processed', DEFAULT_LIDAR_NUM_POINTS_PROCESSED)
        lidar_sensor.listen(lambda data: _parse_lidar_cb(env_ref, data, sensor_key, num_processed))
    return lidar_sensor

def setup_semantic_lidar_sensor(world, vehicle, env_ref, semantic_lidar_config, transform, sensor_key='semantic_lidar'):
    # ... (blueprint setup using semantic_lidar_config) ...
    lidar_bp = _setup_lidar_blueprint(world.get_blueprint_library(), semantic_lidar_config, is_semantic=True)
    lidar_sensor = world.try_spawn_actor(lidar_bp, transform, attach_to=vehicle)
    if lidar_sensor:
        num_processed = semantic_lidar_config.get('num_points_processed', DEFAULT_LIDAR_NUM_POINTS_PROCESSED)
        lidar_sensor.listen(lambda data: _parse_semantic_lidar_cb(env_ref, data, sensor_key, num_processed))
    return lidar_sensor 