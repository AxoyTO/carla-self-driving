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

def setup_gnss_sensor(world, vehicle, carla_env_weak_ref, sensor_tick=DEFAULT_SENSOR_TICK, transform=None):
    """Spawns and configures a GNSS sensor."""
    blueprint_library = world.get_blueprint_library()
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_bp.set_attribute('sensor_tick', str(sensor_tick))

    if transform is None:
        transform = carla.Transform(carla.Location(x=0.0, z=2.0))  # Default on the roof

    gnss = world.spawn_actor(gnss_bp, transform, attach_to=vehicle)
    logger.info(f"Spawned GNSS Sensor: {gnss.id} at {transform}")

    def callback(data):  # carla.GnssMeasurement
        me = carla_env_weak_ref()
        if me:
            me.latest_sensor_data['gnss'] = data
    gnss.listen(callback)
    return gnss

def process_gnss_data(raw_data):
    """Processes raw GNSS data (carla.GnssMeasurement) to a NumPy array."""
    if raw_data is None:
        return np.zeros(get_gnss_observation_space().shape, dtype=np.float32)
    return np.array([raw_data.latitude, raw_data.longitude, raw_data.altitude], dtype=np.float32)

# --- IMU Sensor --- 
def get_imu_observation_space():
    """Returns the gymnasium.spaces.Box for IMU data."""
    # accel(3), gyro(3)
    return spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

def setup_imu_sensor(world, vehicle, carla_env_weak_ref, sensor_tick=DEFAULT_SENSOR_TICK, transform=None):
    """Spawns and configures an IMU sensor."""
    blueprint_library = world.get_blueprint_library()
    imu_bp = blueprint_library.find('sensor.other.imu')
    imu_bp.set_attribute('sensor_tick', str(sensor_tick))
    # Add other IMU attributes if needed (e.g., noise for accelerometer/gyroscope)
    # imu_bp.set_attribute('noise_accel_stddev_x', '0.0')
    # ... and for y, z, and gyro noise attributes

    if transform is None:
        transform = carla.Transform(carla.Location(x=0.0, z=1.5))  # Approx center of mass

    imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)
    logger.info(f"Spawned IMU Sensor: {imu.id} at {transform}")

    def callback(data):  # carla.IMUMeasurement
        me = carla_env_weak_ref()
        if me:
            me.latest_sensor_data['imu'] = data
    imu.listen(callback)
    return imu

def process_imu_data(raw_data):
    """Processes raw IMU data (carla.IMUMeasurement) to a NumPy array."""
    if raw_data is None:
        return np.zeros(get_imu_observation_space().shape, dtype=np.float32)
    return np.array([
        raw_data.accelerometer.x, raw_data.accelerometer.y, raw_data.accelerometer.z,
        raw_data.gyroscope.x, raw_data.gyroscope.y, raw_data.gyroscope.z
    ], dtype=np.float32) 