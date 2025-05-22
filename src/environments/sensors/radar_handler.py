# Handles RADAR sensor setup and data processing 

import carla
import numpy as np
import weakref
import logging

from gymnasium import spaces

# Configure a logger for this handler
logger = logging.getLogger(__name__)

# Default RADAR parameters
DEFAULT_RADAR_RANGE = 70.0  # meters
DEFAULT_RADAR_HORIZONTAL_FOV = 30.0  # degrees
DEFAULT_RADAR_VERTICAL_FOV = 10.0  # degrees
DEFAULT_RADAR_POINTS_PER_SECOND = "1500" # String for set_attribute
DEFAULT_PROCESSED_RADAR_MAX_DETECTIONS = 20
DEFAULT_RADAR_SENSOR_TICK = "0.05"

def get_radar_observation_space(max_detections=DEFAULT_PROCESSED_RADAR_MAX_DETECTIONS):
    """Returns the gymnasium.spaces.Box for processed RADAR data."""
    return spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(max_detections, 4),  # M_detections x (range, azimuth, altitude_angle, velocity)
        dtype=np.float32
    )

def setup_radar_sensor(world, vehicle, carla_env_weak_ref, radar_config=None, transform=None):
    """Spawns and configures a RADAR sensor."""
    blueprint_library = world.get_blueprint_library()
    radar_bp = blueprint_library.find('sensor.other.radar')

    cfg = radar_config if radar_config else {}
    horizontal_fov = str(cfg.get('horizontal_fov', DEFAULT_RADAR_HORIZONTAL_FOV))
    vertical_fov = str(cfg.get('vertical_fov', DEFAULT_RADAR_VERTICAL_FOV))
    radar_range = str(cfg.get('range', DEFAULT_RADAR_RANGE))
    points_per_second = str(cfg.get('points_per_second', DEFAULT_RADAR_POINTS_PER_SECOND))
    sensor_tick = str(cfg.get('sensor_tick', DEFAULT_RADAR_SENSOR_TICK))

    radar_bp.set_attribute('horizontal_fov', horizontal_fov)
    radar_bp.set_attribute('vertical_fov', vertical_fov)
    radar_bp.set_attribute('range', radar_range)
    radar_bp.set_attribute('points_per_second', points_per_second) # Affects density/update rate
    radar_bp.set_attribute('sensor_tick', sensor_tick)

    if transform is None:
        transform = carla.Transform(carla.Location(x=2.0, z=1.0))  # Default front bumper

    radar = world.spawn_actor(radar_bp, transform, attach_to=vehicle)
    logger.debug(f"Spawned RADAR Sensor: {radar.id} at {transform} with range {radar_range}m")

    def callback(data):  # carla.RadarMeasurement
        me = carla_env_weak_ref()
        if me:
            me.latest_sensor_data['radar'] = data
    radar.listen(callback)
    return radar

def process_radar_data(raw_data, max_target_detections=DEFAULT_PROCESSED_RADAR_MAX_DETECTIONS):
    """Processes raw RADAR data (carla.RadarMeasurement) to a fixed-size NumPy array."""
    target_shape = (max_target_detections, 4)
    if raw_data is None:
        return np.zeros(target_shape, dtype=np.float32)

    detections_list = []
    for detection in raw_data:  # Iterate carla.RadarDetection
        # Each detection has: detection.depth, detection.azimuth, detection.altitude, detection.velocity
        detections_list.append([
            detection.depth,      # Range in meters
            detection.azimuth,    # Horizontal angle in radians
            detection.altitude,   # Vertical angle in radians
            detection.velocity    # Velocity of the detected object towards the sensor in m/s
        ])
    
    # Ensure fixed size (take first N or pad)
    processed_detections = np.zeros(target_shape, dtype=np.float32)
    num_to_take = min(len(detections_list), max_target_detections)
    if num_to_take > 0:
        processed_detections[:num_to_take, :] = np.array(detections_list[:num_to_take], dtype=np.float32)
    
    return processed_detections 