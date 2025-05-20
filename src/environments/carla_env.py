import carla
import random
import numpy as np
import time # For sensor data synchronization
import weakref # For sensor callbacks
import logging # Import logging module
from collections import OrderedDict # For Dict space if needed, though gymnasium handles it

from .base_env import BaseEnv
import gymnasium as gym # Changed from # import gymnasium as gym
from gymnasium import spaces # Changed from # from gymnasium import spaces

from .sensors import camera_handler # Import the new camera handler
from .sensors import lidar_handler # Import the new lidar handler
from .sensors import radar_handler # Import the new radar handler
from .sensors import gnss_imu_handler # Import the new gnss_imu_handler
from .sensors import collision_handler # Import the new collision handler

# Define a class to store sensor data temporarily for synchronization
# class SensorData: # This class is no longer used for multiple sensors
#     def __init__(self):
#         self.image = None
#         self.timestamp = None
#         self.frame = None

# Default sensor parameters (can be overridden in __init__ or config)
DEFAULT_IMAGE_WIDTH = 84
DEFAULT_IMAGE_HEIGHT = 84
DEFAULT_LIDAR_CHANNELS = 32
DEFAULT_LIDAR_RANGE = 50.0  # meters
DEFAULT_LIDAR_POINTS_PER_SECOND = 120000 
DEFAULT_LIDAR_ROTATION_FREQUENCY = 10.0 # Hz
DEFAULT_LIDAR_UPPER_FOV = 15.0
DEFAULT_LIDAR_LOWER_FOV = -25.0
PROCESSED_LIDAR_NUM_POINTS = 720 # Example: 360 azimuth steps * 2 beams (simplified)

RADAR_RANGE = 70.0 # meters
RADAR_HORIZONTAL_FOV = 30.0 # degrees
RADAR_VERTICAL_FOV = 10.0 # degrees
RADAR_POINTS_PER_SECOND = 1500 # For configuring the sensor, not directly for fixed array size
PROCESSED_RADAR_MAX_DETECTIONS = 20


class CarlaEnv(BaseEnv):
    def __init__(self, host='localhost', port=2000, town='Town03', timestep=0.05, \
                 image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT), fov=90, discrete_actions=True, num_actions=5, \
                 log_level=logging.INFO,
                 lidar_params=None, radar_params=None): # Added params for new sensors
        super().__init__()
        
        # Initialize logger for this class instance
        self.logger = logging.getLogger(f"CarlaEnv.{town}") # Specific logger name
        self.logger.setLevel(log_level)
        # Note: Handler configuration (e.g., to print to console) is usually done globally in main.py

        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.sensor_list = [] # To keep track of sensors to destroy them later
        self.target_waypoint = None # For navigation task
        self.previous_location = None # To calculate distance change

        self.host = host
        self.port = port
        self.town = town
        self.timestep = timestep # CARLA simulation time step
        
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.fov = fov
        self.discrete_actions = discrete_actions
        self.num_actions = num_actions # Used if discrete_actions is True

        # For storing raw sensor data from callbacks
        # self.sensor_data = SensorData() # Replaced by latest_sensor_data
        self.latest_sensor_data = {
            'rgb_camera': None,
            'depth_camera': None,
            'semantic_camera': None,
            'lidar': None,
            'collision': None, # Store raw collision event for processing
            'gnss': None,
            'imu': None,
            'radar': None
        }
        # To store processed collision info (count, intensity)
        self.collision_info = {'count': 0, 'last_intensity': 0.0, 'last_other_actor': None}

        # Store available spawn points
        self._spawn_points = []

        # Sensor specific configurations
        self.lidar_config = {
            'channels': lidar_params.get('channels', DEFAULT_LIDAR_CHANNELS) if lidar_params else DEFAULT_LIDAR_CHANNELS,
            'range': lidar_params.get('range', DEFAULT_LIDAR_RANGE) if lidar_params else DEFAULT_LIDAR_RANGE,
            'points_per_second': lidar_params.get('points_per_second', DEFAULT_LIDAR_POINTS_PER_SECOND) if lidar_params else DEFAULT_LIDAR_POINTS_PER_SECOND,
            'rotation_frequency': lidar_params.get('rotation_frequency', DEFAULT_LIDAR_ROTATION_FREQUENCY) if lidar_params else DEFAULT_LIDAR_ROTATION_FREQUENCY,
            'upper_fov': lidar_params.get('upper_fov', DEFAULT_LIDAR_UPPER_FOV) if lidar_params else DEFAULT_LIDAR_UPPER_FOV,
            'lower_fov': lidar_params.get('lower_fov', DEFAULT_LIDAR_LOWER_FOV) if lidar_params else DEFAULT_LIDAR_LOWER_FOV,
            'num_points_processed': lidar_params.get('num_points_processed', PROCESSED_LIDAR_NUM_POINTS) if lidar_params else PROCESSED_LIDAR_NUM_POINTS,
            'sensor_tick': str(self.timestep) # Match simulation timestep
        }
        self.radar_config = {
            'range': radar_params.get('range', RADAR_RANGE) if radar_params else RADAR_RANGE,
            'horizontal_fov': radar_params.get('horizontal_fov', RADAR_HORIZONTAL_FOV) if radar_params else RADAR_HORIZONTAL_FOV,
            'vertical_fov': radar_params.get('vertical_fov', RADAR_VERTICAL_FOV) if radar_params else RADAR_VERTICAL_FOV,
            'points_per_second': radar_params.get('points_per_second', RADAR_POINTS_PER_SECOND) if radar_params else RADAR_POINTS_PER_SECOND,
            'max_detections_processed': radar_params.get('max_detections_processed', PROCESSED_RADAR_MAX_DETECTIONS) if radar_params else PROCESSED_RADAR_MAX_DETECTIONS,
            'sensor_tick': str(self.timestep)
        }

        # Define action and observation spaces
        if self.discrete_actions:
            # Example: 0: Straight, 1: Left, 2: Right, 3: Straight+Brake, 4: No-Op (Coast)
            self._action_space = spaces.Discrete(self.num_actions) 
        else:
            # Example: Continuous actions [throttle_brake, steer]
            # throttle_brake: -1 (full brake) to 1 (full throttle)
            # steer: -1 (left) to 1 (right)
            self._action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # New Observation Space using gym.spaces.Dict
        obs_spaces = OrderedDict()
        # Get camera observation spaces from the handler
        camera_obs_spaces = camera_handler.get_camera_observation_spaces(self.image_width, self.image_height)
        obs_spaces.update(camera_obs_spaces)
        
        # LIDAR - using lidar_handler
        obs_spaces['lidar'] = lidar_handler.get_lidar_observation_space(
            num_points=self.lidar_config['num_points_processed']
        )
        # GNSS & IMU - using gnss_imu_handler
        obs_spaces['gnss'] = gnss_imu_handler.get_gnss_observation_space()
        obs_spaces['imu'] = gnss_imu_handler.get_imu_observation_space()
        # RADAR - using radar_handler
        obs_spaces['radar'] = radar_handler.get_radar_observation_space(
            max_detections=self.radar_config['max_detections_processed']
        )
        # Adding collision data to observation space (optional, can also just be used for reward/done)
        # For simplicity, let's not add collision to observation for now, but use self.collision_info directly.

        self._observation_space = spaces.Dict(obs_spaces)

        self.connect()
        if self.world:
            self._spawn_points = self.world.get_map().get_spawn_points()
            if not self._spawn_points:
                self.logger.warning("No spawn points found in the map during init!") # Changed from print

    def connect(self):
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0) # seconds
            self.world = self.client.load_world(self.town)
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.timestep
            self.world.apply_settings(settings)

            self.map = self.world.get_map()
            self.logger.info(f"Connected to CARLA server at {self.host}:{self.port} and loaded {self.town}") # Changed from print
        except Exception as e:
            self.logger.error(f"Error connecting to CARLA or loading world: {e}", exc_info=True) # Changed from print, added exc_info
            raise

    def reset(self):
        # Destroy existing actors
        time.sleep(0.1) # Brief pause before attempting destruction
        self._destroy_actors() 
        time.sleep(0.2) # Increased delay after destruction to allow server to process fully
        
        # Initialize/clear sensor data stores
        for key in self.latest_sensor_data:
            self.latest_sensor_data[key] = None
        self.collision_info = {'count': 0, 'last_intensity': 0.0, 'last_other_actor': None}

        # Select start and target spawn points
        if not self._spawn_points or len(self._spawn_points) < 2:
            # Attempt to refetch spawn points if empty
            if self.world and self.map:
                self._spawn_points = self.map.get_spawn_points()
            if not self._spawn_points or len(self._spawn_points) < 2:
                 raise RuntimeError("Not enough spawn points available in the map to set start and target.")

        start_spawn_point = random.choice(self._spawn_points)
        possible_targets = [sp for sp in self._spawn_points if sp != start_spawn_point]
        if not possible_targets:
            # This case should ideally not be reached if len(self._spawn_points) >= 2
            self.target_waypoint = random.choice(self._spawn_points) 
        else:
            self.target_waypoint = random.choice(possible_targets)
        
        self.logger.debug(f"New target waypoint: {self.target_waypoint.location}") # Changed from print & made debug

        # Spawn vehicle and sensors
        self._spawn_vehicle(spawn_point_transform=start_spawn_point) 
        self._setup_sensors() 
        
        # Position spectator camera near the spawned vehicle
        if self.world and self.vehicle:
            spectator = self.world.get_spectator()
            vehicle_transform = self.vehicle.get_transform()
            self.logger.debug(f"Vehicle spawned at transform: {vehicle_transform}") # Changed from print
            # Position spectator a bit behind and above the vehicle, looking in the same direction
            # Adjust X, Y, Z offsets as needed for desired view
            offset_x = -10  # meters behind the car
            offset_z = 5   # meters above the car
            # Calculate new spectator transform relative to vehicle's transform
            # spectator_location = vehicle_transform.location + vehicle_transform.rotation.get_forward_vector() * offset_x + carla.Location(z=offset_z)
            # A simpler way: transform a point from local vehicle space to world space
            # Define the offset in vehicle's local coordinate system
            local_offset = carla.Location(x=offset_x, y=0, z=offset_z) 
            # Transform this local offset to world coordinates based on vehicle's current transform
            spectator_location = vehicle_transform.transform(local_offset)

            spectator_transform = carla.Transform(spectator_location, vehicle_transform.rotation)
            # print(f"[DEBUG] Setting spectator transform to: {spectator_transform}") # DEBUG PRINT
            self.logger.debug(f"Setting spectator transform to: {spectator_transform}") # Changed from print
            spectator.set_transform(spectator_transform)

        # Ensure the world ticks at least once after sensor setup for them to register and send first data
        if self.world and self.vehicle: # Ensure world and vehicle exist
            self.world.tick() 
        else:
            raise RuntimeError("World or Vehicle not initialized before first tick in reset.")

        # Tick the world multiple times to ensure sensors are initialized and sending data
        # Camera might need a few frames to be ready.
        for i in range(10): # Increased retry ticks slightly, with a small delay
            observation = self._get_observation()
            if observation is not None:
                # print(f"Observation obtained after {i+1} initial ticks.")
                break
            # print(f"Still no observation after {i+1} initial ticks. Ticking again.")
            self.world.tick() # Tick again if no observation yet
            time.sleep(self.timestep) # Allow time for server to process
        
        if observation is None:
            self.logger.warning("Initial observation is None, trying a few more ticks...") # Changed from print
            for i in range(10):
                self.world.tick()
                time.sleep(self.timestep * 2) # Longer sleep
                observation = self._get_observation()
                if observation is not None:
                    self.logger.info(f"Observation obtained after {i+1} extra ticks.") # Changed from print
                    break
            if observation is None:
                 self.logger.error("Failed to get initial observation from sensors even after retries.")
                 raise RuntimeError("Failed to get initial observation from sensors even after retries.")

        self.previous_location = self.vehicle.get_location() if self.vehicle else None

        self._position_spectator_camera() # Position camera after reset and vehicle spawn

        return observation, {} # Return observation and an empty info dict

    def step(self, action):
        if self.vehicle is None:
            raise RuntimeError("Vehicle not spawned. Call reset() first.")

        # Apply action to the vehicle
        self._apply_action(action)

        # Tick the world
        self.world.tick()

        # Get observation, reward, done, info
        current_location = self.vehicle.get_location() if self.vehicle else None
        observation = self._get_observation()
        reward = self._calculate_reward(current_location, self.previous_location)
        terminated, term_info = self._check_done(current_location)
        self.previous_location = current_location # Update previous location for next step

        self._position_spectator_camera() # Update spectator camera position each step

        truncated = False # For time limits, not implemented yet by env step (handled by main loop)
        info = term_info # Pass termination info from _check_done

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action):
        if self.vehicle is None:
            self.logger.warning("Cannot apply action: vehicle is not spawned or is None.") # Changed from print
            return

        control = carla.VehicleControl()
        
        if self.discrete_actions:
            # Example discrete action mapping:
            # Action 0: Full Throttle, Straight
            # Action 1: Full Throttle, Steer Left
            # Action 2: Full Throttle, Steer Right
            # Action 3: Half Throttle, Straight (Coasting/Maintain)
            # Action 4: Brake
            if action == 0: # Straight
                control.throttle = 0.75
                control.steer = 0.0
            elif action == 1: # Left
                control.throttle = 0.5
                control.steer = -0.5
            elif action == 2: # Right
                control.throttle = 0.5
                control.steer = 0.5
            elif action == 3: # Brake
                control.throttle = 0.0
                control.brake = 1.0
            elif action == 4: # Coast (slight throttle to maintain speed)
                control.throttle = 0.3
                control.steer = 0.0
            # Add more actions as needed up to self.num_actions
            else:
                self.logger.warning(f"Unknown discrete action {action}") # Changed from print
                control.throttle = 0.0 # Default to no action
                control.steer = 0.0
        else: # Continuous actions
            control.throttle = float(max(0, action[0]))  # action[0] is throttle (0 to 1)
            control.brake = float(max(0, -action[0])) # action[0] is brake if negative (-1 to 0)
            control.steer = float(action[1])          # action[1] is steer (-1 to 1)
            
        self.vehicle.apply_control(control)
        # print(f"Applied action: {action}, Control: t={control.throttle}, s={control.steer}, b={control.brake}")

    def _get_observation(self):
        if self.world is None:
            self.logger.warning("World is None in _get_observation. Returning zeroed observation.")
            return self._get_zeroed_observation()

        # Ensure world tick has happened if in sync mode (usually handled by step/reset)
        # current_snapshot = self.world.get_snapshot()
        # current_frame = current_snapshot.frame
        
        obs_data = OrderedDict()

        # --- Camera Data Processing using Camera Handler ---
        obs_data['rgb_camera'] = camera_handler.process_rgb_camera_data(
            self.latest_sensor_data.get('rgb_camera'), 
            self._observation_space['rgb_camera'].shape
        )
        obs_data['depth_camera'] = camera_handler.process_depth_camera_data(
            self.latest_sensor_data.get('depth_camera'), 
            self._observation_space['depth_camera'].shape
        )
        obs_data['semantic_camera'] = camera_handler.process_semantic_camera_data(
            self.latest_sensor_data.get('semantic_camera'), 
            self._observation_space['semantic_camera'].shape
        )
            
        # LIDAR - using lidar_handler
        obs_data['lidar'] = lidar_handler.process_lidar_data(
            self.latest_sensor_data.get('lidar'),
            num_target_points=self.lidar_config['num_points_processed']
        )

        # GNSS & IMU - using gnss_imu_handler
        obs_data['gnss'] = gnss_imu_handler.process_gnss_data(
            self.latest_sensor_data.get('gnss')
        )
        obs_data['imu'] = gnss_imu_handler.process_imu_data(
            self.latest_sensor_data.get('imu')
        )

        # RADAR - using radar_handler
        radar_data = self.latest_sensor_data.get('radar')
        if radar_data is not None: 
            obs_data['radar'] = radar_handler.process_radar_data(
                radar_data,
                max_target_detections=self.radar_config['max_detections_processed']
            )
        else:
            obs_data['radar'] = np.zeros(self._observation_space['radar'].shape, dtype=np.float32)

        return obs_data

    def _get_zeroed_observation(self):
        """Returns a dictionary of zeroed observations matching the observation_space structure."""
        obs = OrderedDict()
        for key, space in self._observation_space.spaces.items():
            obs[key] = np.zeros(space.shape, dtype=space.dtype)
        return obs

    def _calculate_reward(self, current_location, previous_location):
        reward = 0.0

        # Constants for reward shaping
        REWARD_SPEED_MULTIPLIER = 0.05 # Reduced slightly
        PENALTY_COLLISION = -200.0
        PENALTY_PER_STEP = -0.1 
        PENALTY_STUCK_OR_REVERSING = -0.5 
        MIN_FORWARD_SPEED_THRESHOLD = 0.1 
        REWARD_GOAL_REACHED = 300.0
        REWARD_DISTANCE_FACTOR = 1.0 # Multiplier for reward based on distance change to target
        WAYPOINT_REACHED_THRESHOLD = 5.0 # meters, threshold to consider waypoint reached

        if self.vehicle is None or not self.vehicle.is_alive or current_location is None:
            return PENALTY_COLLISION 

        # 1. Reward for forward speed (as before)
        velocity_vector = self.vehicle.get_velocity()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_forward_vector = vehicle_transform.get_forward_vector()
        forward_speed = np.dot(
            [velocity_vector.x, velocity_vector.y, velocity_vector.z],
            [vehicle_forward_vector.x, vehicle_forward_vector.y, vehicle_forward_vector.z]
        )
        if forward_speed > MIN_FORWARD_SPEED_THRESHOLD:
            reward += forward_speed * REWARD_SPEED_MULTIPLIER
        elif forward_speed < 0: 
            reward += PENALTY_STUCK_OR_REVERSING * 2 
        else: 
            reward += PENALTY_STUCK_OR_REVERSING

        # 2. Penalty for collision (as before)
        if hasattr(self, 'collision_info') and self.collision_info.get('count', 0) > 0:
            reward += PENALTY_COLLISION 

        # 3. Small penalty per step (as before)
        reward += PENALTY_PER_STEP

        # 4. Navigation Reward (distance to target)
        if self.target_waypoint and previous_location:
            current_dist_to_target = current_location.distance(self.target_waypoint.location)
            prev_dist_to_target = previous_location.distance(self.target_waypoint.location)
            
            # Reward for getting closer
            distance_reduction = prev_dist_to_target - current_dist_to_target
            reward += distance_reduction * REWARD_DISTANCE_FACTOR

            # Check if goal is reached (within threshold)
            if current_dist_to_target < WAYPOINT_REACHED_THRESHOLD:
                reward += REWARD_GOAL_REACHED
                self.logger.info(f"Goal reached! Target: {self.target_waypoint.location}, Current: {current_location}") # Changed from print

        self.logger.debug(f"Calculated reward: {reward:.2f} (Speed: {forward_speed:.2f} m/s, Dist: {current_dist_to_target if self.target_waypoint and current_location else -1:.2f} m)") # Changed from print, made debug
        return reward

    def _check_done(self, current_location):
        info = {}
        WAYPOINT_REACHED_THRESHOLD = 5.0 # meters

        if self.vehicle is not None:
            if not self.vehicle.is_alive: 
                 self.logger.info("Vehicle is not alive (destroyed), episode done.") # Changed from print
                 info["termination_reason"] = "vehicle_destroyed"
                 return True, info
        else: # No vehicle
            self.logger.info("No vehicle found, episode done.") # Added log for this case
            info["termination_reason"] = "no_vehicle"
            return True, info
        
        if hasattr(self, 'collision_info') and self.collision_info.get('count', 0) > 0:
            self.logger.info(f"Collision detected! Episode done. Collision intensity: {self.collision_info.get('last_intensity', 0)}") # Changed from print
            self.collision_info['count'] = 0 
            info["termination_reason"] = "collision"
            return True, info

        # Check if goal reached
        if self.target_waypoint and current_location:
            dist_to_target = current_location.distance(self.target_waypoint.location)
            if dist_to_target < WAYPOINT_REACHED_THRESHOLD:
                info["termination_reason"] = "goal_reached"
                return True, info

        return False, info 

    def _spawn_vehicle(self, spawn_point_transform=None): # Added spawn_point_transform argument
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3') 
        if vehicle_bp.has_attribute('color'):
            vehicle_bp.set_attribute('color', '0,0,255') 

        if spawn_point_transform is None: # If no specific spawn point, choose randomly or a default
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                self.logger.error("No spawn points found in the map for _spawn_vehicle during random selection!") # Changed from print
                raise RuntimeError("No spawn points available.")
            spawn_point_transform = random.choice(spawn_points)

        # Ensure previous vehicle is destroyed before spawning a new one
        if self.vehicle is not None:
            # print(f"Destroying existing vehicle {self.vehicle.id} before spawning new one.")
            self.vehicle.destroy()
            self.vehicle = None
            if self.world.get_settings().synchronous_mode:
                 self.world.tick()

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point_transform) # Use the provided or chosen transform
        
        if self.vehicle is None:
            # Fallback if the specific spawn_point_transform failed (should be rare if it came from get_spawn_points)
            self.logger.warning(f"Failed to spawn vehicle at specified transform {spawn_point_transform.location}. Trying other spawn points...") # Changed from print
            all_spawn_points = self.map.get_spawn_points()
            for sp in all_spawn_points: 
                self.vehicle = self.world.try_spawn_actor(vehicle_bp, sp)
                if self.vehicle is not None:
                    self.logger.info(f"Spawned vehicle {self.vehicle.id} at {sp.location} (fallback)") # Changed from print
                    break
            if self.vehicle is None:
                 self.logger.error("Could not spawn vehicle after trying all spawn points (fallback).") # Changed from print
                 raise RuntimeError("Could not spawn vehicle after trying all spawn points (fallback).")
        else:
            self.logger.debug(f"Spawned vehicle {self.vehicle.id} at {self.vehicle.get_location()}") # Changed from print, made debug
            # pass # Original pass was here, logger.debug serves a similar purpose if enabled

    def _setup_sensors(self):
        """
        Set up all sensors attached to the vehicle.
        This now calls the individual setup functions from the sensor handlers.
        """
        # Clean up existing sensors first, just in case
        # This is important if reset() is called multiple times
        for sensor in self.sensor_list:
            if sensor and sensor.is_alive: # Check if sensor object exists and is alive
                sensor.destroy()
                self.logger.debug(f"Destroyed sensor: {sensor.id}")
        self.sensor_list = []
        self.logger.info("Destroyed existing sensors before setting up new ones.")

        if not self.vehicle:
            self.logger.error("Vehicle not spawned before setting up sensors.")
            return

        # blueprint_library = self.world.get_blueprint_library() # No longer needed here, handlers get it
        world_weak_ref = weakref.ref(self) # Pass weakref of carla_env instance

        # Define sensor transforms (relative to the vehicle)
        # These can be customized as needed
        rgb_cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        depth_cam_transform = carla.Transform(carla.Location(x=1.5, y=0.1, z=2.4)) # Small offset for clarity
        seg_cam_transform = carla.Transform(carla.Location(x=1.5, y=-0.1, z=2.4))   # Small offset
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))      # On the roof
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))      # Front bumper
        gnss_transform = carla.Transform(carla.Location(x=0.0, z=2.0))       # Roof
        imu_transform = carla.Transform(carla.Location(x=0.0, z=1.5))        # Approx CoM
        collision_transform = carla.Transform(carla.Location(x=0.0, z=0.0))  # At vehicle origin

        # RGB Camera
        rgb_camera_actor = camera_handler.setup_rgb_camera(
            self.world, self.vehicle, world_weak_ref,
            image_width=self.image_width, image_height=self.image_height, 
            fov=self.fov, sensor_tick=str(self.timestep), transform=rgb_cam_transform
        )
        if rgb_camera_actor: self.sensor_list.append(rgb_camera_actor)
        
        # Depth Camera
        depth_camera_actor = camera_handler.setup_depth_camera(
            self.world, self.vehicle, world_weak_ref,
            image_width=self.image_width, image_height=self.image_height,
            fov=self.fov, sensor_tick=str(self.timestep), transform=depth_cam_transform
        )
        if depth_camera_actor: self.sensor_list.append(depth_camera_actor)

        # Semantic Segmentation Camera
        seg_camera_actor = camera_handler.setup_segmentation_camera(
            self.world, self.vehicle, world_weak_ref,
            image_width=self.image_width, image_height=self.image_height,
            fov=self.fov, sensor_tick=str(self.timestep), transform=seg_cam_transform
        )
        if seg_camera_actor: self.sensor_list.append(seg_camera_actor)
        
        # LIDAR Sensor
        lidar_actor = lidar_handler.setup_lidar_sensor(
            self.world, self.vehicle, world_weak_ref, 
            lidar_config=self.lidar_config, transform=lidar_transform
        )
        if lidar_actor: self.sensor_list.append(lidar_actor)
        
        # GNSS Sensor
        gnss_actor = gnss_imu_handler.setup_gnss_sensor(
            self.world, self.vehicle, world_weak_ref, 
            sensor_tick=str(self.timestep), transform=gnss_transform
        )
        if gnss_actor: self.sensor_list.append(gnss_actor)
        
        # IMU Sensor
        imu_actor = gnss_imu_handler.setup_imu_sensor(
            self.world, self.vehicle, world_weak_ref, 
            sensor_tick=str(self.timestep), transform=imu_transform
        )
        if imu_actor: self.sensor_list.append(imu_actor)
        
        # RADAR Sensor
        radar_actor = radar_handler.setup_radar_sensor(
            self.world, self.vehicle, world_weak_ref, 
            radar_config=self.radar_config, transform=radar_transform
        )
        if radar_actor: self.sensor_list.append(radar_actor)
        
        # Collision Sensor
        collision_actor = collision_handler.setup_collision_sensor(
            self.world, self.vehicle, world_weak_ref, transform=collision_transform
        )
        if collision_actor: self.sensor_list.append(collision_actor)
        
        # Tick the world once to let sensors register
        if self.world and self.world.get_settings().synchronous_mode:
            self.world.tick()
            # time.sleep(0.1) # Small delay to ensure registration - tick should be enough in sync mode
        self.logger.info(f"All sensors set up. Total sensors attached: {len(self.sensor_list)}")

    def _position_spectator_camera(self):
        """Positions the spectator camera behind and above the vehicle."""
        if self.world and self.vehicle and self.vehicle.is_alive:
            spectator = self.world.get_spectator()
            vehicle_transform = self.vehicle.get_transform()
            # print(f"[DEBUG] Vehicle current transform for spectator: {vehicle_transform}") 
            
            offset_x = -10  # meters behind the car
            offset_z = 5    # meters above the car
            local_offset = carla.Location(x=offset_x, y=0, z=offset_z) 
            spectator_location = vehicle_transform.transform(local_offset)

            spectator_transform = carla.Transform(spectator_location, vehicle_transform.rotation)
            # print(f"[DEBUG] Setting spectator transform to: {spectator_transform}") 
            spectator.set_transform(spectator_transform)

    def _destroy_actors(self):
        self.logger.debug(f"Attempting to destroy {len(self.sensor_list)} sensors and vehicle (if any).")
        
        # Stop and collect sensor actors
        sensors_to_destroy = []
        if self.client:
            for sensor in self.sensor_list:
                if sensor is not None and sensor.is_alive:
                    if sensor.is_listening:
                        sensor.stop()
                        # self.world.tick() # Optional: Tick after stopping each sensor
                    sensors_to_destroy.append(sensor.id)
        
        # Destroy vehicle first (if it exists)
        # This can sometimes help if sensors are rigidly attached and cause issues if vehicle disappears first
        # However, common wisdom is often sensors first. Let's stick to sensors first for now unless issues persist.
        if self.vehicle is not None and self.vehicle.is_alive:
            self.logger.debug(f"Destroying vehicle: {self.vehicle.id}")
            self.vehicle.destroy()
            self.vehicle = None
            if self.world and self.world.get_settings().synchronous_mode:
                 self.world.tick() # Tick to process vehicle destruction
                 time.sleep(0.05) # Small delay after vehicle destruction tick

        # Destroy sensor actors in a batch
        if self.client and sensors_to_destroy:
            self.logger.debug(f"Destroying sensor actors by ID: {sensors_to_destroy}")
            try:
                self.client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in sensors_to_destroy], True)
                if self.world and self.world.get_settings().synchronous_mode:
                    self.world.tick() # Tick to process sensor destruction
                    time.sleep(0.05) # Small delay after sensor destruction tick
            except RuntimeError as e:
                self.logger.error(f"RuntimeError during sensor destruction: {e}", exc_info=True)
                # Optionally, try to destroy one by one as a fallback
                # for actor_id in sensors_to_destroy:
                #     try:
                #         actor_to_destroy = self.world.get_actor(actor_id)
                #         if actor_to_destroy and actor_to_destroy.is_alive:
                #             actor_to_destroy.destroy()
                #             if self.world and self.world.get_settings().synchronous_mode:
                #                 self.world.tick()
                #     except Exception as e_ind:
                #         self.logger.error(f"Failed to destroy individual sensor {actor_id}: {e_ind}")
                pass # Continue cleanup

        self.sensor_list = [] 
        # Reset sensor data stores
        for key in self.latest_sensor_data:
            self.latest_sensor_data[key] = None
        self.collision_info = {'count': 0, 'last_intensity': 0.0, 'last_other_actor': None} # Reset collision info

        # Vehicle should be None already, but as a safeguard:
        if self.vehicle is not None:
            if self.vehicle.is_alive:
                 self.logger.warning("Vehicle was not properly destroyed, attempting again.")
                 self.vehicle.destroy()
            self.vehicle = None
        
        # Final tick to ensure server is in a clean state
        if self.world and self.world.get_settings().synchronous_mode:
             self.world.tick() 

    def render(self, mode='human'):
        # CARLA rendering is usually handled by the spectator or sensor data.
        # This method could be used to display sensor data using Pygame or OpenCV if needed.
        print(f"Render called with mode: {mode}. (Not implemented yet)")
        pass

    def close(self):
        self._destroy_actors()
        # Revert to asynchronous mode if it was changed
        if self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            self.world = None # Nullify world object
        
        # self.client = None # Client object does not need to be explicitly closed or nullified
        self.logger.info("Closed CARLA environment and destroyed actors.") # Changed from print

    @property
    def action_space(self):
        if self._action_space is None:
            # Define a default action space if not set. This should be configured.
            # Example: Discrete action space (e.g., 0: straight, 1: left, 2: right)
            # from gymnasium.spaces import Discrete
            # self._action_space = Discrete(3) 
            raise NotImplementedError("Action space not defined yet.")
        return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is None:
            # Define a default observation space if not set. This should be configured.
            # Example: Image observation (e.g., 84x84 RGB image)
            # from gymnasium.spaces import Box
            # self._observation_space = Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
            raise NotImplementedError("Observation space not defined yet.")
        return self._observation_space

# Example usage (for testing purposes, typically in main.py or a test script)
# if __name__ == '__main__':
#     try:
#         env = CarlaEnv()
#         obs, _ = env.reset()
#         print("Initial observation shape:", obs.shape if hasattr(obs, 'shape') else type(obs))
#         for _ in range(10):
#             action = env.action_space.sample() # Assuming action_space is defined and has sample()
#             obs, reward, terminated, truncated, info = env.step(action)
#             print(f"Step: Reward={reward}, Done={terminated}")
#             if terminated:
#                 print("Episode finished.")
#                 obs, _ = env.reset()
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         if 'env' in locals() and env is not None:
#             env.close() 