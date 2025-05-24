# Handles GNSS and IMU sensor setup and data processing 

import carla
import numpy as np
import weakref
import logging

from gymnasium import spaces

# Configure a logger for this handler
logger = logging.getLogger(__name__)

# Default sensor parameters
DEFAULT_SENSOR_TICK = "0.05" # Default to match typical environment timestep

# --- GNSS Sensor --- 
def get_gnss_observation_space():
    """Returns the gymnasium.spaces.Box for GNSS data."""
    return spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)  # lat, lon, alt

def process_gnss_data(gnss_measurement: carla.GnssMeasurement) -> np.ndarray:
    """Processes GnssMeasurement into a NumPy array [lat, lon, alt]."""
    return np.array([
        gnss_measurement.latitude,
        gnss_measurement.longitude,
        gnss_measurement.altitude
    ], dtype=np.float32)

def setup_gnss_sensor(world, vehicle, env_ref, sensor_tick, transform):
    """Spawns and configures a GNSS sensor."""
    blueprint_library = world.get_blueprint_library()
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_bp.set_attribute('sensor_tick', str(sensor_tick))

    if transform is None:
        transform = carla.Transform(carla.Location(x=0.0, z=2.0))  # Default on the roof

    gnss_sensor = world.try_spawn_actor(gnss_bp, transform, attach_to=vehicle)
    if gnss_sensor:
        gnss_sensor.listen(lambda data: _gnss_callback(env_ref, data))
        logger.debug(f"Spawned GNSS Sensor: {gnss_sensor.id} at {transform}")
    return gnss_sensor

def _gnss_callback(weak_self, data: carla.GnssMeasurement):
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): return
    processed_data = process_gnss_data(data)
    self.latest_sensor_data['gnss'] = processed_data

# --- IMU Sensor --- 
def get_imu_observation_space():
    """Returns the gymnasium.spaces.Box for IMU data."""
    # accel(3), gyro(3)
    return spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

def process_imu_data(imu_measurement: carla.IMUMeasurement) -> np.ndarray:
    """Processes IMUMeasurement into a NumPy array [accel_x,y,z, gyro_x,y,z]."""
    accel = imu_measurement.accelerometer
    gyro = imu_measurement.gyroscope
    # Compass (imu_measurement.compass) is orientation wrt North; not typically used directly as 6-DOF IMU state vector for RL
    # The 6-DOF is usually linear acceleration and angular velocity.
    return np.array([
        accel.x, accel.y, accel.z,
        gyro.x, gyro.y, gyro.z
    ], dtype=np.float32)

def setup_imu_sensor(world, vehicle, env_ref, sensor_tick, transform):
    """Spawns and configures an IMU sensor."""
    blueprint_library = world.get_blueprint_library()
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', str(sensor_tick))
    # Add other IMU attributes if needed (e.g., noise for accelerometer/gyroscope)
    # imu_bp.set_attribute('noise_accel_stddev_x', '0.0')
    # ... and for y, z, and gyro noise attributes

    if transform is None:
        transform = carla.Transform(carla.Location(x=0.0, z=1.5))  # Approx center of mass

    imu_sensor = world.try_spawn_actor(imu_bp, transform, attach_to=vehicle)
    if imu_sensor:
        imu_sensor.listen(lambda data: _imu_callback(env_ref, data))
        logger.debug(f"Spawned IMU Sensor: {imu_sensor.id} at {transform}")
    return imu_sensor

def _imu_callback(weak_self, data: carla.IMUMeasurement):
    self = weak_self()
    if not self or not hasattr(self, 'latest_sensor_data'): return
    processed_data = process_imu_data(data)
    self.latest_sensor_data['imu'] = processed_data 