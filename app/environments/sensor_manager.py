import carla
import weakref
import logging
import numpy as np
import time
from typing import Optional, Tuple, List, Dict, Any, Union
from gymnasium import spaces # Assuming gymnasium is used, adjust if it's gym
from collections import OrderedDict

# Assuming these handlers are in the same directory or accessible via PYTHONPATH
from .sensors import camera_handler
from .sensors import lidar_handler
from .sensors import radar_handler
from .sensors import gnss_imu_handler
from .sensors import collision_handler

class SensorManager:
    """Manages all sensors attached to the vehicle."""

    def __init__(self, world: carla.World, vehicle: carla.Actor, env_ref: weakref.ref,
                 image_size: Tuple[int, int], fov: int, timestep: float, time_scale: float,
                 enable_pygame_display: bool = False, pygame_window_size: Tuple[int, int] = (1920, 1080),
                 lidar_config: Optional[dict] = None, radar_config: Optional[dict] = None,
                 observation_space: Optional[spaces.Dict] = None, logger: Optional[logging.Logger] = None):
        """Initialize the sensor manager.
        Args:
            world: CARLA world instance
            vehicle: Vehicle actor to attach sensors to
            env_ref: Weak reference to the environment
            image_size: Tuple of (width, height) for camera images
            fov: Field of view for cameras
            timestep: Simulation timestep
            time_scale: Time scale factor
            enable_pygame_display: Whether to enable pygame display
            pygame_window_size: Tuple of (width, height) for pygame window
            lidar_config: Optional LIDAR configuration
            radar_config: Optional RADAR configuration
            observation_space: Optional observation space definition
            logger: Optional logger instance
        """
        self.world = world
        self.vehicle = vehicle
        self.env_ref = env_ref
        self.image_width, self.image_height = image_size
        self.fov = fov
        self.timestep = timestep
        self.time_scale = time_scale
        self.enable_pygame_display = enable_pygame_display
        self.pygame_window_width, self.pygame_window_height = pygame_window_size
        self.lidar_config = lidar_config if lidar_config is not None else {}
        self.radar_config = radar_config if radar_config is not None else {}
        self.observation_space = observation_space # This should be a spaces.Dict
        self.logger = logger if logger else logging.getLogger(__name__ + ".SensorManager")

        self.sensor_list: List[Dict[str, any]] = []
        self.latest_sensor_data: Dict[str, any] = {} # This can be removed if SensorManager doesn't hold state

        self._init_sensor_data_keys_for_env() # Initialize keys expected by environment

    def _init_sensor_data_keys_for_env(self):
        """Defines the keys that the environment will expect in its latest_sensor_data dictionary.
           Sensor callbacks should update the environment's dictionary directly using these keys.
        """
        # This method is mostly for clarity or if SensorManager needed to track its own subset.
        # The actual latest_sensor_data dictionary is in CarlaEnv and populated by callbacks.
        # However, SensorManager needs to know these keys if it were to, for example, provide a zeroed version.
        # For now, this list ensures SensorManager is aware of what it helps manage.
        expected_keys = [
            'rgb_camera', 'left_rgb_camera', 'right_rgb_camera', 'rear_rgb_camera',
            'display_rgb_camera', 'display_left_rgb_camera', 'display_right_rgb_camera', 'display_rear_rgb_camera',
            'spectator_camera', 'depth_camera', 'semantic_camera',
            'display_depth_camera', 'display_semantic_camera',
            'lidar', 'semantic_lidar',
            'collision', 'gnss', 'imu', 'radar', 'lane_invasion_event'
        ]
        # Initialize SensorManager's own placeholder if it were to manage this data directly.
        # for key in expected_keys:
        #     self.latest_sensor_data[key] = None 
        pass # No direct data storage in SensorManager for now, env holds it.

    def _get_scaled_sensor_tick(self) -> str:
        if self.time_scale > 0:
            return str(self.timestep * self.time_scale)
        return str(self.timestep)

    def setup_all_sensors(self):
        self._cleanup_existing_sensors()
        if not self.vehicle or not self.vehicle.is_alive:
            self.logger.error("SensorManager: Vehicle not spawned or not alive before setting up sensors.")
            return

        self._setup_camera_sensors()
        self._setup_lidar_sensors()
        self._setup_other_sensors()

        if self.enable_pygame_display:
            self._setup_spectator_camera()

        if self.world and self.world.get_settings().synchronous_mode:
            self.world.tick()
        self.logger.debug(f"SensorManager: All sensors set up. Total sensors: {len(self.sensor_list)}")

    def _cleanup_existing_sensors(self):
        """Clean up any existing sensors before setting up new ones."""
        actors_to_destroy_ids = []
        for sensor_entry in self.sensor_list:
            sensor_actor = sensor_entry.get('actor')
            if sensor_actor and sensor_actor.is_alive:
                if hasattr(sensor_actor, 'is_listening') and sensor_actor.is_listening:
                    try:
                        sensor_actor.stop()
                    except RuntimeError as e:
                        self.logger.warning(f"SensorManager: Error stopping sensor {sensor_actor.id} ({sensor_actor.type_id}): {e}. It might already be invalid.")
                actors_to_destroy_ids.append(sensor_actor.id) # Store IDs for batch destruction
        
        if actors_to_destroy_ids:
            self.logger.debug(f"SensorManager: Attempting to destroy {len(actors_to_destroy_ids)} sensor actors by ID: {actors_to_destroy_ids}")
            # Using individual destruction with ticks for more robust cleanup in sync mode, as batch sometimes times out.
            for actor_id in actors_to_destroy_ids:
                actor_to_destroy = self.world.get_actor(actor_id) # Get a fresh reference
                if actor_to_destroy and actor_to_destroy.is_alive:
                    try:
                        actor_to_destroy.destroy()
                        # self.logger.debug(f"SensorManager: Destroyed sensor {actor_id}")
                        if self.world.get_settings().synchronous_mode:
                            self.world.tick() # Tick after each actor destruction
                    except RuntimeError as e:
                        self.logger.error(f"SensorManager: RuntimeError destroying sensor {actor_id}: {e}")
                # else:
                    # self.logger.debug(f"SensorManager: Sensor {actor_id} was already None or not alive before explicit destroy call.")
            
            # One final tick after all individual destructions have been attempted.
            # if self.world.get_settings().synchronous_mode:
            #     self.world.tick()
            self.logger.debug(f"SensorManager: Finished attempting to destroy {len(actors_to_destroy_ids)} sensors individually.")
                        
        self.sensor_list = []
        # self.logger.debug("SensorManager: Sensor list cleared after cleanup.") # Redundant with above info log

    def _add_sensor_to_list(self, actor: Optional[carla.Actor], purpose: str):
        if actor:
            self.sensor_list.append({'actor': actor, 'purpose': purpose})
            self.logger.debug(f"SensorManager: Added {purpose} sensor: {actor.type_id} ({actor.id})")
        else:
            self.logger.warning(f"SensorManager: Attempted to add a None sensor for purpose {purpose}")

    def _setup_camera_sensors(self):
        cam_configs = [
            {'key': 'rgb_camera', 'transform': carla.Transform(carla.Location(x=1.5, z=2.4))},
            {'key': 'left_rgb_camera', 'transform': carla.Transform(carla.Location(x=0.75, y=-0.9, z=1.3), carla.Rotation(yaw=-90))},
            {'key': 'right_rgb_camera', 'transform': carla.Transform(carla.Location(x=0.75, y=0.9, z=1.3), carla.Rotation(yaw=90))},
            {'key': 'rear_rgb_camera', 'transform': carla.Transform(carla.Location(x=-1.8, z=1.6), carla.Rotation(yaw=180))}
        ]
        depth_transform = carla.Transform(carla.Location(x=1.5, y=0.1, z=2.4))
        seg_transform = carla.Transform(carla.Location(x=1.5, y=-0.1, z=2.4))

        for cfg in cam_configs:
            # Check if the base sensor key is in observation space (e.g., 'rgb_camera')
            if self.observation_space and cfg['key'] in self.observation_space.spaces:
                 self._create_camera_sensor(cfg['key'], cfg['transform'], camera_handler.setup_rgb_camera)
            elif cfg['key'] == 'rgb_camera': # Always setup main rgb_camera for agent, even if not in obs for some reason (though it should be)
                 self._create_camera_sensor(cfg['key'], cfg['transform'], camera_handler.setup_rgb_camera)

        if self.observation_space and 'depth_camera' in self.observation_space.spaces:
            self._create_camera_sensor('depth_camera', depth_transform, camera_handler.setup_depth_camera)

        if self.observation_space and 'semantic_camera' in self.observation_space.spaces:
            self._create_camera_sensor('semantic_camera', seg_transform, camera_handler.setup_semantic_segmentation_camera)

    def _create_camera_sensor(self, sensor_key: str, transform: carla.Transform, setup_func):
        # Use optimized camera setup with performance enhancements
        agent_cam = setup_func(
            self.world, self.vehicle, self.env_ref, self.image_width, self.image_height,
            self.fov, self._get_scaled_sensor_tick(), transform, sensor_key
        )
        self._add_sensor_to_list(agent_cam, 'agent')

        if self.enable_pygame_display:
            display_key = f'display_{sensor_key}'
            display_cam = setup_func(
                self.world, self.vehicle, self.env_ref, 
                self.pygame_window_width // 2, self.pygame_window_height // 2,
                self.fov, self._get_scaled_sensor_tick(), transform, display_key
            )
            self._add_sensor_to_list(display_cam, 'display')

    def _setup_lidar_sensors(self):
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        if self.observation_space and 'lidar' in self.observation_space.spaces:
            # Use optimized LIDAR setup with processed point count
            lidar_config_with_processing = self.lidar_config.copy()
            lidar_config_with_processing['num_points_processed'] = lidar_handler.DEFAULT_LIDAR_NUM_POINTS_PROCESSED
            
            lidar_actor = lidar_handler.setup_lidar_sensor_optimized(
                self.world, self.vehicle, self.env_ref, lidar_config_with_processing,
                lidar_transform, 'lidar'
            )
            self._add_sensor_to_list(lidar_actor, 'agent')

        if self.observation_space and 'semantic_lidar' in self.observation_space.spaces:
            # Use optimized semantic LIDAR setup with processed point count
            semantic_lidar_config_with_processing = self.lidar_config.copy()
            semantic_lidar_config_with_processing['num_points_processed'] = lidar_handler.DEFAULT_LIDAR_NUM_POINTS_PROCESSED
            
            semantic_lidar_actor = lidar_handler.setup_semantic_lidar_sensor_optimized(
                self.world, self.vehicle, self.env_ref, semantic_lidar_config_with_processing, 
                lidar_transform, 'semantic_lidar'
            )
            self._add_sensor_to_list(semantic_lidar_actor, 'agent')

    def _setup_other_sensors(self):
        gnss_actor = gnss_imu_handler.setup_gnss_sensor(
            self.world, self.vehicle, self.env_ref, self._get_scaled_sensor_tick(),
            carla.Transform(carla.Location(x=0.0, z=2.0))
        )
        self._add_sensor_to_list(gnss_actor, 'agent')

        imu_actor = gnss_imu_handler.setup_imu_sensor(
            self.world, self.vehicle, self.env_ref, self._get_scaled_sensor_tick(),
            carla.Transform(carla.Location(x=0.0, z=1.5))
        )
        self._add_sensor_to_list(imu_actor, 'agent')

        env = self.env_ref()
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0)) # Default
        if env and hasattr(env, 'radar_to_vehicle_transform'):
             radar_transform = env.radar_to_vehicle_transform
        else:
            self.logger.warning("SensorManager: radar_to_vehicle_transform not found on env. Using default.")
             
        radar_actor = radar_handler.setup_radar_sensor(
            self.world, self.vehicle, self.env_ref, self.radar_config, radar_transform
        )
        self._add_sensor_to_list(radar_actor, 'agent')

        collision_actor = collision_handler.setup_collision_sensor(
            self.world, self.vehicle, self.env_ref, carla.Transform(carla.Location(x=0.0, z=0.0))
        )
        self._add_sensor_to_list(collision_actor, 'agent')

        lane_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_sensor = self.world.spawn_actor(lane_bp, carla.Transform(), attach_to=self.vehicle)
        def lane_cb(event):
            cb_env = self.env_ref()
            if cb_env:
                cb_env.latest_sensor_data['lane_invasion_event'] = event
            else:
                self.logger.warning("Lane invasion sensor: Environment reference is None")
        lane_sensor.listen(lane_cb)
        self._add_sensor_to_list(lane_sensor, 'agent')
        self.logger.debug("Lane invasion sensor setup completed.")

    def _setup_spectator_camera(self):
        bp = camera_handler._setup_camera_blueprint(
            self.world.get_blueprint_library(), 'sensor.camera.rgb',
            self.pygame_window_width, self.pygame_window_height, self.fov, self._get_scaled_sensor_tick()
        )
        env = self.env_ref()
        spectator_transform = carla.Transform(carla.Location(x=-5.5, z=3.5), carla.Rotation(pitch=-15)) # Default
        if env and hasattr(env, 'pygame_display_camera_transform'):
            spectator_transform = env.pygame_display_camera_transform
        else:
            self.logger.warning("SensorManager: pygame_display_camera_transform not found on env. Using default.")

        actor = self.world.spawn_actor(bp, spectator_transform, attach_to=self.vehicle)
        def spec_cb(data):
            cb_env = self.env_ref()
            if cb_env:
                cb_env.latest_sensor_data['spectator_camera'] = data
        actor.listen(spec_cb)
        self._add_sensor_to_list(actor, 'spectator')

    def get_observation_data(self) -> OrderedDict:
        """Gets the current observation data based on the observation space.
           Relies on the environment's latest_sensor_data being up-to-date from callbacks.
        Returns:
            OrderedDict containing sensor observations, ensuring no None values for defined spaces.
        """
        obs = OrderedDict()
        env = self.env_ref() # Get the actual environment instance
        
        if not env:
            self.logger.warning("SensorManager: Env reference lost. Cannot get observation data.")
            # If observation_space is available, return zeroed based on it, else empty.
            if self.observation_space:
                for key, space_def in self.observation_space.spaces.items():
                    obs[key] = np.zeros(space_def.shape, dtype=space_def.dtype)
            return obs

        if not self.observation_space:
            self.logger.warning("SensorManager: Observation space not available. Returning empty observation dict.")
            return obs

        for key, space_def in self.observation_space.spaces.items():
            data = env.latest_sensor_data.get(key)
            if data is None:
                # self.logger.debug(f"SensorManager: No data for key '{key}'. Using zeroed array.")
                obs[key] = np.zeros(space_def.shape, dtype=space_def.dtype)
            elif not isinstance(data, np.ndarray):
                # This case should ideally not happen if sensor handlers are correct.
                # It means data is present but not a numpy array as expected.
                self.logger.warning(f"SensorManager: Data for key '{key}' is not a numpy array (type: {type(data)}). Using zeroed array.")
                obs[key] = np.zeros(space_def.shape, dtype=space_def.dtype)
            elif data.shape != space_def.shape:
                self.logger.warning(f"SensorManager: Data for key '{key}' has mismatched shape {data.shape} vs expected {space_def.shape}. Using zeroed array.")
                obs[key] = np.zeros(space_def.shape, dtype=space_def.dtype)
            elif data.dtype != space_def.dtype:
                # Attempt to cast, or use zeroed array if cast fails or is inappropriate
                try:
                    obs[key] = data.astype(space_def.dtype)
                    # self.logger.debug(f"SensorManager: Data for key '{key}' dtype {data.dtype} cast to {space_def.dtype}.")
                except Exception as e:
                    self.logger.warning(f"SensorManager: Failed to cast data for key '{key}' from {data.dtype} to {space_def.dtype} (Error: {e}). Using zeroed array.")
                    obs[key] = np.zeros(space_def.shape, dtype=space_def.dtype)
            else:
                obs[key] = data
        return obs

    def get_all_sensor_actors(self) -> List[carla.Actor]:
        return [s_entry['actor'] for s_entry in self.sensor_list if s_entry.get('actor') and s_entry['actor'].is_alive]

    def cleanup(self):
        """Public method to cleanup sensors, usually called by the environment."""
        self._cleanup_existing_sensors()
        # self.latest_sensor_data = {} # Don't clear this if env owns the data store
        # self._sensor_save_dirs = {} # If sensor manager handled saving 