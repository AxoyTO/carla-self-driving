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

def process_radar_data(radar_measurement: carla.RadarMeasurement, max_detections: int) -> np.ndarray:
    """Processes raw RADAR data into a fixed-size NumPy array.
       Returns [depth, azimuth, altitude, velocity] for each detection, padded/truncated to max_detections.
    """
    detections = []
    for detection in radar_measurement:
        detections.append([
            detection.depth,      # float (meters)
            detection.azimuth,    # float (radians)
            detection.altitude,   # float (radians)
            detection.velocity    # float (m/s, towards sensor is negative)
        ])
    
    num_detected = len(detections)
    processed_detections = np.zeros((max_detections, 4), dtype=np.float32)

    if num_detected > 0:
        if num_detected > max_detections:
            # Simple truncation if more detections than max_detections
            # Could be smarter (e.g., sort by depth/velocity and take closest/fastest)
            processed_detections[:, :] = np.array(detections[:max_detections])
        else:
            # Pad with zeros if fewer detections
            processed_detections[:num_detected, :] = np.array(detections)
            
    return processed_detections

def _radar_callback(weak_self, radar_data: carla.RadarMeasurement, sensor_key: str, max_detections_processed: int):
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): return
    processed_numpy_array = process_radar_data(radar_data, max_detections_processed)
    self.latest_sensor_data[sensor_key] = processed_numpy_array

def setup_radar_sensor(world, vehicle, env_ref, radar_config, transform, sensor_key='radar'):
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

    radar_sensor = world.try_spawn_actor(radar_bp, transform, attach_to=vehicle)
    if radar_sensor:
        max_detections = radar_config.get('max_detections_processed', DEFAULT_PROCESSED_RADAR_MAX_DETECTIONS)
        radar_sensor.listen(lambda data: _radar_callback(env_ref, data, sensor_key, max_detections))
    logger.debug(f"Spawned RADAR Sensor: {radar_sensor.id} at {transform} with range {radar_range}m")
    return radar_sensor 