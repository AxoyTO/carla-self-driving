import carla
import random
import numpy as np
import time # For sensor data synchronization
import weakref # For sensor callbacks
import logging # Import logging module
from collections import OrderedDict # For Dict space if needed, though gymnasium handles it
from collections import Counter # For counting sensor types
# import pygame # No longer directly used here for init/font/etc.
import math # For compass
import os
from datetime import datetime # For timestamped save directory

from .base_env import BaseEnv
import gymnasium as gym # Changed from # import gymnasium as gym
from gymnasium import spaces # Changed from # from gymnasium import spaces

from .sensors import camera_handler # Import the new camera handler
from .sensors import lidar_handler # Import the new lidar handler
from .sensors import radar_handler # Import the new radar handler
from .sensors import gnss_imu_handler # Import the new gnss_imu_handler
from .sensors import collision_handler # Import the new collision handler
from utils.pygame_visualizer import PygameVisualizer # Corrected import path based on previous fix

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
                 image_size=(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT), fov=90, discrete_actions=True, num_actions=6, \
                 log_level=logging.INFO,
                 lidar_params=None, radar_params=None,
                 enable_pygame_display=False, # Default to False as per user request
                 pygame_window_width=1920, pygame_window_height=1080,
                 initial_curriculum_episodes=50,
                 save_sensor_data=False,        # New: Main toggle for saving sensor data
                 sensor_save_base_path="./sensor_capture", # New: Base path
                 sensor_save_interval=100):      # New: Save every N steps
        super().__init__()
        
        self.logger = logging.getLogger(f"CarlaEnv.{town}")
        self.logger.setLevel(log_level)

        self.pygame_window_width = pygame_window_width
        self.pygame_window_height = pygame_window_height

        self.client = None
        self.world = None
        self.map = None
        self.vehicle = None
        self.sensor_list = [] 
        self.target_waypoint = None 
        self.previous_location = None

        self.host = host
        self.port = port
        self.town = town
        self.timestep = timestep 
        
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.fov = fov
        self.discrete_actions = discrete_actions
        self.num_actions = num_actions

        self.latest_sensor_data = {
            'rgb_camera': None,
            'pygame_rgb_camera': None, 
            'depth_camera': None,
            'semantic_camera': None,
            'lidar': None,
            'collision': None, 
            'gnss': None, # Will be removed from HUD display
            'imu': None,
            'radar': None,
            'lane_invasion_event': None # For lane invasion detector
        }
        self.collision_info = {
            'count': 0, 
            'last_intensity': 0.0, 
            'last_other_actor_id': "N/A", 
            'last_other_actor_type': "N/A"
        }

        self._spawn_points = []

        self.relevant_traffic_light_state_debug = None # Stores the carla.TrafficLightState enum
        self.traffic_light_stop_line_waypoint = None 

        # Debug attributes (will be used for Pygame display)
        self.episode_count_debug = 0
        self.step_count_debug = 0
        self.current_action_debug = "N/A"
        self.step_reward_debug = 0.0
        self.episode_score_debug = 0.0
        self.forward_speed_debug = 0.0
        self.dist_to_goal_debug = float('inf')
        self.collision_flag_debug = False # This will be passed directly as boolean
        self.proximity_penalty_flag_debug = False # This will be passed directly as boolean
        self.last_termination_reason_debug = "N/A"
        
        self.enable_pygame_display = enable_pygame_display
        self.visualizer = None 
        if self.enable_pygame_display:
            self.visualizer = PygameVisualizer(
                window_width=self.pygame_window_width, 
                window_height=self.pygame_window_height,
                caption=f"CARLA RL Agent View",
                carla_env_ref=weakref.ref(self) # Pass weak reference to self
            )

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

        # Define a transform for the Pygame display camera (chase cam)
        self.pygame_display_camera_transform = carla.Transform(carla.Location(x=-5.5, y=0, z=3.5), carla.Rotation(pitch=-15))

        # Define action and observation spaces
        if self.discrete_actions:
            # Example: 0: Straight, 1: Left, 2: Right, 3: Straight+Brake, 4: No-Op (Coast)
            # Updated: 0: Forward-HighThrottle, 1: Forward-Left, 2: Forward-Right, 3: Brake, 4: Coast, 5: Reverse
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

        # Reward shaping parameters
        self.target_speed_kmh = 40.0 # Target speed for reward
        self.current_action_for_reward = None # To store action for reward calculation

        # Curriculum learning state
        self.initial_curriculum_episodes = initial_curriculum_episodes
        self.current_episode_for_curriculum = 0 # Will be incremented by main training loop via a setter or by reset itself

        # Define fixed easy spawn and target for initial curriculum (example for Town03)
        # These indices might need adjustment based on your map's spawn points.
        # You can get spawn points via: world.get_map().get_spawn_points()
        # And then print them to choose suitable ones.
        self.curriculum_spawn_point_idx = 10 # Example spawn index
        self.curriculum_target_point_idx = 50 # Example target index (different from spawn)

        # Sensor data saving attributes
        self.save_sensor_data_enabled = save_sensor_data
        self.sensor_save_base_path = sensor_save_base_path
        self.sensor_save_interval = sensor_save_interval
        self.current_run_sensor_save_path = ""
        self._sensor_save_dirs = {} # To store paths like {'rgb_camera': 'path/to/rgb'}

        if self.save_sensor_data_enabled:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.current_run_sensor_save_path = os.path.join(self.sensor_save_base_path, f"run_{timestamp}")
            try:
                os.makedirs(self.current_run_sensor_save_path, exist_ok=True)
                self.logger.info(f"Sensor data will be saved in: {self.current_run_sensor_save_path}")
                # Pre-create subdirectories for sensors we intend to save
                # Note: These keys MUST match the keys used in self.latest_sensor_data for easy lookup
                sensors_to_log_individually = ['rgb_camera', 'depth_camera', 'semantic_camera', 'lidar']
                for sensor_key in sensors_to_log_individually:
                    s_path = os.path.join(self.current_run_sensor_save_path, sensor_key)
                    os.makedirs(s_path, exist_ok=True)
                    self._sensor_save_dirs[sensor_key] = s_path
            except Exception as e:
                self.logger.error(f"Could not create sensor save directories: {e}")
                self.save_sensor_data_enabled = False # Disable if cant create dirs

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
        self.current_episode_for_curriculum += 1
        time.sleep(0.1) 
        self._destroy_actors() 
        time.sleep(0.2) 
        for key in self.latest_sensor_data: self.latest_sensor_data[key] = None
        self.latest_sensor_data['lane_invasion_event'] = None
        self.collision_info = {'count': 0, 'last_intensity': 0.0, 'last_other_actor_id': "N/A", 'last_other_actor_type': "N/A"}

        if not self._spawn_points:
            if self.world and self.map:
                self._spawn_points = self.map.get_spawn_points()
            if not self._spawn_points:
                 raise RuntimeError("No spawn points available in the map.")

        start_spawn_point = None # Initialize to ensure it's assigned
        if self.current_episode_for_curriculum <= self.initial_curriculum_episodes and len(self._spawn_points) > max(self.curriculum_spawn_point_idx, self.curriculum_target_point_idx):
            self.logger.info(f"Curriculum active (Episode {self.current_episode_for_curriculum}/{self.initial_curriculum_episodes}): Using fixed easy route.")
            start_spawn_point = self._spawn_points[self.curriculum_spawn_point_idx]
            self.target_waypoint = self._spawn_points[self.curriculum_target_point_idx]
            if start_spawn_point == self.target_waypoint: 
                self.logger.warning("Curriculum spawn and target are the same, adjusting target.")
                new_target_idx = (self.curriculum_target_point_idx + 1) % len(self._spawn_points)
                if new_target_idx == self.curriculum_spawn_point_idx and len(self._spawn_points) > 1:
                    new_target_idx = (new_target_idx + 1) % len(self._spawn_points) # Try one more hop
                if self._spawn_points[new_target_idx] != start_spawn_point:
                    self.target_waypoint = self._spawn_points[new_target_idx]
                else: # Still same or only one spawn point, log and proceed (will be very short episode)
                    self.logger.error("Could not find a different target point for curriculum with >1 spawn points. Or only 1 spawn point exists.")
        else:
            if self.current_episode_for_curriculum == self.initial_curriculum_episodes + 1 and self.initial_curriculum_episodes > 0:
                 self.logger.info(f"Curriculum finished. Switching to random spawn/target points.")
                 # No need to set self.initial_curriculum_episodes = 0, the check current_episode > initial_curriculum_episodes handles it.
            
            if len(self._spawn_points) < 1 : # Need at least one for start, and ideally >1 for target
                raise RuntimeError("Not enough spawn points for random selection.")
            start_spawn_point = random.choice(self._spawn_points)
            if len(self._spawn_points) < 2:
                self.target_waypoint = start_spawn_point # Target self if only one point
                self.logger.warning("Only one spawn point available. Target set to start point.")
            else:
                possible_targets = [sp for sp in self._spawn_points if sp != start_spawn_point]
                self.target_waypoint = random.choice(possible_targets) 
        
        self.logger.debug(f"New target waypoint: {self.target_waypoint.location if self.target_waypoint else 'None'}") 
        self.logger.debug(f"Start spawn point: {start_spawn_point.location if start_spawn_point else 'None'}")

        # Ensure start_spawn_point is valid before proceeding
        if start_spawn_point is None:
            raise RuntimeError("Start spawn point was not set. Check curriculum/random logic and spawn point availability.")

        self._spawn_vehicle(spawn_point_transform=start_spawn_point) 
        self._setup_sensors() 
        self._position_spectator_camera()
        
        if self.world and self.vehicle: self.world.tick() 
        else: raise RuntimeError("World or Vehicle not initialized properly before first tick in reset.")
        
        observation = None; agent_rgb_ready = False; pygame_cam_ready = not self.enable_pygame_display
        for i in range(20): # Increased initial ticks slightly
            observation = self._get_observation()
            agent_rgb_ready = observation is not None and observation.get('rgb_camera') is not None
            if self.enable_pygame_display: pygame_cam_ready = (self.latest_sensor_data.get('pygame_rgb_camera') is not None)
            if agent_rgb_ready and pygame_cam_ready: break
            if self.world: self.world.tick(); 
            time.sleep(self.timestep if i < 10 else self.timestep * 2) # Progressively longer sleep if needed
        
        if not(agent_rgb_ready and pygame_cam_ready):
            self.logger.error("Failed to get initial observation/display data even after extended retries.")
            if not agent_rgb_ready: observation = self._get_zeroed_observation()
        else:
            self.logger.info("Initial observation and display camera data obtained.")

        self.previous_location = self.vehicle.get_location() if self.vehicle else None
        self.episode_count_debug += 1; self.step_count_debug = 0; self.episode_score_debug = 0.0
        self.current_action_debug = "N/A (Reset)"; self.step_reward_debug = 0.0
        if self.vehicle:
            self.forward_speed_debug = 0.0 
            current_loc = self.vehicle.get_location()
            if self.target_waypoint and current_loc:
                 self.dist_to_goal_debug = current_loc.distance(self.target_waypoint.location)
            else:
                 self.dist_to_goal_debug = float('inf')
        else:
            self.dist_to_goal_debug = float('inf')
            self.forward_speed_debug = 0.0
        
        self.relevant_traffic_light_state_debug = None
        self.collision_flag_debug = False
        self.proximity_penalty_flag_debug = False
        
        if self.enable_pygame_display:
            self._render_pygame()

        return observation, {}

    def step(self, action):
        if self.vehicle is None:
            raise RuntimeError("Vehicle not spawned. Call reset() first.")

        self.current_action_debug = str(action) 
        self.step_count_debug += 1
        self.current_action_for_reward = action # Store action for use in _calculate_reward
        self._apply_action(action)
        self.world.tick()

        # Process sensor data that arrived from the tick
        # The collision callback (if a collision happened) would have updated self.latest_sensor_data['collision']
        raw_collision_event = self.latest_sensor_data.get('collision')
        if raw_collision_event:
            self._process_collision_event(raw_collision_event)
            self.latest_sensor_data['collision'] = None # Consume the event

        # ---- Call to save sensor data ----
        self._save_specific_sensor_data() # Call before _get_observation if using latest_sensor_data directly from callbacks
                                        # Or after if _get_observation populates something that should be saved.
                                        # Given save_to_disk uses raw carla objects, before is fine.

        current_location = self.vehicle.get_location() if self.vehicle else None
        observation = self._get_observation()
        
        self.collision_flag_debug = False 
        self.proximity_penalty_flag_debug = False

        reward = self._calculate_reward(current_location, self.previous_location)
        
        # Trigger collision notification - uses self.collision_flag_debug set in _calculate_reward
        if self.collision_flag_debug and self.visualizer and self.enable_pygame_display:
            notif_text = f"COLLISION! Int: {self.collision_info['last_intensity']:.2f}"
            if self.collision_info['last_other_actor_type'] != "Environment":
                notif_text += f" | Hit: {self.collision_info['last_other_actor_type']} (ID: {self.collision_info['last_other_actor_id']})"
            else:
                notif_text += f" | Hit: {self.collision_info['last_other_actor_type']}"
            self.visualizer.add_notification(notif_text, duration_seconds=4.0, color=(255, 20, 20)) # Bright Red

        self.step_reward_debug = reward 
        self.episode_score_debug += reward 

        terminated, term_info = self._check_done(current_location)
        self.last_termination_reason_debug = term_info.get("termination_reason", "terminated") if terminated else "Running"

        lane_event = self.latest_sensor_data.get('lane_invasion_event')
        if lane_event and self.visualizer and self.enable_pygame_display:
            crossed_lanes_text = []
            for marking in lane_event.crossed_lane_markings:
                crossed_lanes_text.append(str(marking.type).upper())
            if crossed_lanes_text:
                self.visualizer.add_notification(f"Lane Invasion: {', '.join(crossed_lanes_text)}", duration_seconds=2.0, color=(255, 165, 0))
            self.latest_sensor_data['lane_invasion_event'] = None 

        self.previous_location = current_location 
        self._position_spectator_camera() 
        
        if self.enable_pygame_display:
            self._render_pygame()

        truncated = False 
        info = term_info 

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action):
        if self.vehicle is None:
            self.logger.warning("Cannot apply action: vehicle is not spawned or is None.") # Changed from print
            return

        control = carla.VehicleControl()
        
        if self.discrete_actions:
            # Example discrete action mapping:
            # Action 0: Forward-HighThrottle, 1: Forward-Left, 2: Forward-Right, 3: Brake, 4: Coast, 5: Reverse
            if action == 0: # Straight high throttle
                control.throttle = 0.75
                control.steer = 0.0
                control.brake = 0.0
                control.reverse = False
            elif action == 1: # Left
                control.throttle = 0.5
                control.steer = -0.5
                control.brake = 0.0
                control.reverse = False
            elif action == 2: # Right
                control.throttle = 0.5
                control.steer = 0.5
                control.brake = 0.0
                control.reverse = False
            elif action == 3: # Brake
                control.throttle = 0.0
                control.steer = 0.0
                control.brake = 1.0
                control.reverse = False
            elif action == 4: # Coast (slight throttle to maintain speed)
                control.throttle = 0.3 # Gentle forward
                control.steer = 0.0
                control.brake = 0.0
                control.reverse = False
            elif action == 5: # Reverse
                control.throttle = 0.3 # Gentle reverse
                control.steer = 0.0
                control.brake = 0.0
                control.reverse = True
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
        
        obs_data = OrderedDict()
        debug_sensor_data_once = not hasattr(self, '_sensor_debug_printed_this_step') # Print once per step
        if debug_sensor_data_once:
            self._sensor_debug_printed_this_step = True

        # --- Camera Data Processing ---
        raw_rgb = self.latest_sensor_data.get('rgb_camera')
        obs_data['rgb_camera'] = camera_handler.process_rgb_camera_data(
            raw_rgb, 
            self._observation_space['rgb_camera'].shape
        )
        if debug_sensor_data_once and raw_rgb:
            self.logger.debug(f"Sensor Verify - RGB: Processed shape={obs_data['rgb_camera'].shape}, dtype={obs_data['rgb_camera'].dtype}, Raw frame={raw_rgb.frame}")

        raw_depth = self.latest_sensor_data.get('depth_camera')
        obs_data['depth_camera'] = camera_handler.process_depth_camera_data(
            raw_depth, 
            self._observation_space['depth_camera'].shape
        )
        if debug_sensor_data_once and raw_depth:
            self.logger.debug(f"Sensor Verify - Depth: Processed shape={obs_data['depth_camera'].shape}, dtype={obs_data['depth_camera'].dtype}, Raw frame={raw_depth.frame}, Sample val={obs_data['depth_camera'][0,0,0] if obs_data['depth_camera'].size > 0 else 'N/A'}")
        
        raw_semantic = self.latest_sensor_data.get('semantic_camera')
        obs_data['semantic_camera'] = camera_handler.process_semantic_camera_data(
            raw_semantic, 
            self._observation_space['semantic_camera'].shape
        )
        if debug_sensor_data_once and raw_semantic:
             self.logger.debug(f"Sensor Verify - Semantic: Processed shape={obs_data['semantic_camera'].shape}, dtype={obs_data['semantic_camera'].dtype}, Raw frame={raw_semantic.frame}")
            
        # LIDAR
        raw_lidar = self.latest_sensor_data.get('lidar')
        obs_data['lidar'] = lidar_handler.process_lidar_data(
            raw_lidar,
            num_target_points=self.lidar_config['num_points_processed']
        )
        if debug_sensor_data_once and raw_lidar:
            self.logger.debug(f"Sensor Verify - LIDAR: Processed shape={obs_data['lidar'].shape}, dtype={obs_data['lidar'].dtype}, Raw points={raw_lidar.get_point_count(0) if raw_lidar else 'N/A'}")

        # GNSS & IMU
        raw_gnss = self.latest_sensor_data.get('gnss')
        obs_data['gnss'] = gnss_imu_handler.process_gnss_data(raw_gnss)
        if debug_sensor_data_once and raw_gnss:
            self.logger.debug(f"Sensor Verify - GNSS: Processed val={obs_data['gnss']}, Raw: ({raw_gnss.latitude:.5f}, {raw_gnss.longitude:.5f}, {raw_gnss.altitude:.2f})")
        
        raw_imu = self.latest_sensor_data.get('imu')
        obs_data['imu'] = gnss_imu_handler.process_imu_data(raw_imu)
        if debug_sensor_data_once and raw_imu:
            self.logger.debug(f"Sensor Verify - IMU: Processed val={obs_data['imu']}, AccelRaw=({raw_imu.accelerometer.x:.2f}, {raw_imu.accelerometer.y:.2f}), GyroRaw=({raw_imu.gyroscope.x:.2f}, {raw_imu.gyroscope.y:.2f})")

        # RADAR
        raw_radar = self.latest_sensor_data.get('radar')
        if raw_radar is not None: 
            obs_data['radar'] = radar_handler.process_radar_data(
                raw_radar,
                max_target_detections=self.radar_config['max_detections_processed']
            )
        else:
            obs_data['radar'] = np.zeros(self._observation_space['radar'].shape, dtype=np.float32)
        if debug_sensor_data_once and raw_radar:
            self.logger.debug(f"Sensor Verify - RADAR: Processed shape={obs_data['radar'].shape}, dtype={obs_data['radar'].dtype}, Raw Detections={len(raw_radar) if raw_radar else 0}")
        
        # Reset the flag after printing once for this step
        if hasattr(self, '_sensor_debug_printed_this_step'):
            del self._sensor_debug_printed_this_step

        return obs_data

    def _get_zeroed_observation(self):
        """Returns a dictionary of zeroed observations matching the observation_space structure."""
        obs = OrderedDict()
        for key, space in self._observation_space.spaces.items():
            obs[key] = np.zeros(space.shape, dtype=space.dtype)
        return obs

    def _format_vehicle_model_name(self, type_id_str: str) -> str:
        if not type_id_str:
            return "Unknown Vehicle"
        name_parts = type_id_str.split('.')
        if name_parts[0] == 'vehicle':
            name_parts = name_parts[1:] # Remove 'vehicle' prefix
        
        formatted_name = ' '.join([part.capitalize() for part in name_parts])
        return formatted_name

    def _update_relevant_traffic_light(self):
        self.relevant_traffic_light_state_debug = None # Store the enum/state itself
        # self.tl_state_debug = "N/A" # String version for logging can be separate if needed

        if self.vehicle and self.map:
            tl_actor = self.vehicle.get_traffic_light()
            if tl_actor and isinstance(tl_actor, carla.TrafficLight):
                self.relevant_traffic_light_state_debug = tl_actor.get_state()
                # self.tl_state_debug = str(self.relevant_traffic_light_state_debug) # For logging
            # else:
                # self.tl_state_debug = "N/A"

    def _process_collision_event(self, event: carla.CollisionEvent):
        """Helper to process a raw collision event and update self.collision_info."""
        self.collision_info['count'] += 1
        self.collision_info['last_intensity'] = math.sqrt(event.normal_impulse.x**2 + event.normal_impulse.y**2 + event.normal_impulse.z**2)
        other_actor = event.other_actor
        if other_actor:
            self.collision_info['last_other_actor_id'] = str(other_actor.id)
            self.collision_info['last_other_actor_type'] = self._format_vehicle_model_name(other_actor.type_id) # Reuse formatter
        else:
            self.collision_info['last_other_actor_id'] = "Static/Unknown"
            self.collision_info['last_other_actor_type'] = "Environment"
        
        # This method is called by the collision sensor callback.
        # The reward calculation and done check will use self.collision_info.
        # self.collision_flag_debug will be set in _calculate_reward if count > 0.

    def _calculate_reward(self, current_location, previous_location):
        reward = 0.0
        if self.vehicle is None or not self.vehicle.is_alive or current_location is None:
            return PENALTY_COLLISION # Use defined constant

        # Constants for reward shaping
        REWARD_SPEED_MULTIPLIER = 0.05
        PENALTY_COLLISION = -200.0
        PENALTY_PER_STEP = -0.1 
        PENALTY_STUCK_OR_REVERSING_BASE = -0.5 
        MIN_FORWARD_SPEED_THRESHOLD = 0.1 
        REWARD_GOAL_REACHED = 300.0
        REWARD_DISTANCE_FACTOR = 1.0
        WAYPOINT_REACHED_THRESHOLD = 5.0

        PENALTY_TRAFFIC_LIGHT_RED_MOVING = -75.0
        REWARD_TRAFFIC_LIGHT_GREEN_PROCEED = 5.0
        REWARD_TRAFFIC_LIGHT_STOPPED_AT_RED = 15.0 
        VEHICLE_STOPPED_SPEED_THRESHOLD = 0.1 # m/s

        PROXIMITY_THRESHOLD_VEHICLE = 4.0 # meters, closer than this is penalized
        PENALTY_PROXIMITY_VEHICLE_FRONT = -15.0 # Penalty for being too close to a vehicle in front

        # Reward shaping parameters
        TARGET_SPEED_REWARD_FACTOR = 0.5  # Multiplier for target speed adherence reward
        TARGET_SPEED_STD_DEV_KMH = 10.0   # How sharply the reward drops off from target speed
        LANE_CENTERING_REWARD_FACTOR = 0.2 # Multiplier for lane centering
        LANE_ORIENTATION_PENALTY_FACTOR = 0.1 # Multiplier for lane orientation penalty
        PENALTY_OFFROAD = -50.0           # Penalty for driving off designated lanes

        self._update_relevant_traffic_light() 
        vehicle_transform = self.vehicle.get_transform()
        current_speed_mps = self.forward_speed_debug # This is already calculated and stored in m/s
        current_speed_kmh = current_speed_mps * 3.6
        is_reversing_action = self.vehicle.get_control().reverse 

        # 1. Penalty per step
        reward += PENALTY_PER_STEP

        # 2. Navigation Reward (distance to target)
        current_dist_to_target = float('inf')
        if self.target_waypoint and previous_location:
            current_dist_to_target = current_location.distance(self.target_waypoint.location)
            prev_dist_to_target = previous_location.distance(self.target_waypoint.location)
            distance_reduction = prev_dist_to_target - current_dist_to_target
            reward += distance_reduction * REWARD_DISTANCE_FACTOR
            if current_dist_to_target < WAYPOINT_REACHED_THRESHOLD:
                reward += REWARD_GOAL_REACHED
                self.logger.info(f"Goal reached! Target: {self.target_waypoint.location}, Current: {current_location}")

        # 3. Speed Adherence Reward (Gaussian-like)
        speed_diff_kmh = current_speed_kmh - self.target_speed_kmh
        speed_reward = TARGET_SPEED_REWARD_FACTOR * math.exp(-0.5 * (speed_diff_kmh / TARGET_SPEED_STD_DEV_KMH)**2)
        # Only apply if not braking hard or in reverse (unless target speed is 0 for stop)
        if not (self.vehicle.get_control().brake > 0.5 and current_speed_kmh < 5) and not is_reversing_action:
             reward += speed_reward
        # Penalize excessive speeding beyond a threshold (e.g. target_speed + 2*std_dev)
        if current_speed_kmh > self.target_speed_kmh + 2 * TARGET_SPEED_STD_DEV_KMH:
            reward -= (current_speed_kmh - (self.target_speed_kmh + 2 * TARGET_SPEED_STD_DEV_KMH)) * 0.1 # Linear penalty for overspeeding

        # 4. Lane Keeping Rewards/Penalties
        if self.map:
            current_waypoint = self.map.get_waypoint(current_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if current_waypoint:
                # Off-road penalty
                if current_waypoint.lane_type != carla.LaneType.Driving:
                    reward += PENALTY_OFFROAD
                else:
                    # Lane Centering (based on distance to waypoint, which is lane center)
                    # vehicle_location is current_location
                    waypoint_location = current_waypoint.transform.location
                    lateral_distance = math.sqrt((current_location.x - waypoint_location.x)**2 + \
                                               (current_location.y - waypoint_location.y)**2 + \
                                               (current_location.z - waypoint_location.z)**2) # Using 3D distance to waypoint for simplicity
                    # A simpler lateral distance: distance from current_location to line defined by waypoint
                    # For now, distance to waypoint itself can act as a proxy if waypoints are dense enough.
                    # Reward for being close to center, penalize for distance.
                    # Max reward when lateral_distance is 0, decreases. Max distance could be half lane_width.
                    # Max offset approx current_waypoint.lane_width / 2.0
                    max_dev = current_waypoint.lane_width / 1.8 # Slightly less than half to encourage really good centering
                    centering_reward = LANE_CENTERING_REWARD_FACTOR * (1.0 - min(lateral_distance / max_dev, 1.0)**2)
                    reward += centering_reward

                    # Lane Orientation (angle between vehicle forward and lane direction)
                    vehicle_fwd = vehicle_transform.get_forward_vector()
                    lane_fwd = current_waypoint.transform.get_forward_vector()
                    angle_diff_rad = math.acos(np.clip(np.dot([vehicle_fwd.x, vehicle_fwd.y], [lane_fwd.x, lane_fwd.y]) / 
                                               (math.sqrt(vehicle_fwd.x**2+vehicle_fwd.y**2)*math.sqrt(lane_fwd.x**2+lane_fwd.y**2) + 1e-4), -1.0, 1.0))
                    angle_diff_deg = math.degrees(angle_diff_rad)
                    # Penalize large deviations (e.g. > 15-20 degrees)
                    if angle_diff_deg > 20.0:
                        reward -= LANE_ORIENTATION_PENALTY_FACTOR * (angle_diff_deg / 90.0) # Normalize by 90 deg
            else: # Could not get a waypoint on a driving lane
                 reward += PENALTY_OFFROAD / 2 # Still penalize if can't find a driving waypoint (likely off-road)

        # 5. Penalty for stuck/reversing (Adjusted to consider intended reverse from action)
        # Action 5 is Reverse. self.current_action_for_reward stores this.
        intended_reverse = (self.current_action_for_reward == 5)
        if is_reversing_action: # If vehicle is in reverse gear
            if not intended_reverse: # Unintended reverse gear (e.g. due to previous action not finishing)
                reward += PENALTY_STUCK_OR_REVERSING_BASE * 1.5 # Penalize if in reverse but not by choice
            elif intended_reverse and current_speed_mps > -MIN_FORWARD_SPEED_THRESHOLD: # Chosen reverse but not moving backward effectively
                 reward += PENALTY_STUCK_OR_REVERSING_BASE / 2 # Penalty for being stuck in chosen reverse
            # If intended_reverse and moving backward: no penalty from this section, speed_reward might be negative if target_speed is positive.
        elif not is_reversing_action: # Not in reverse gear
            if current_speed_mps < MIN_FORWARD_SPEED_THRESHOLD and current_speed_mps > -MIN_FORWARD_SPEED_THRESHOLD: # Stuck or very slow forward
                reward += PENALTY_STUCK_OR_REVERSING_BASE
            elif current_speed_mps < -MIN_FORWARD_SPEED_THRESHOLD: # Moving backward without reverse gear (bad)
                reward += PENALTY_STUCK_OR_REVERSING_BASE * 3
        
        # 6. Collision Penalty (uses self.collision_info updated by _process_collision_event)
        if self.collision_info.get('count', 0) > 0:
            reward += PENALTY_COLLISION 
            self.collision_flag_debug = True 
            self.collision_info['count'] = 0 # Reset count for this step after penalizing, so next step is fresh

        # 7. Traffic Light Adherence
        if self.relevant_traffic_light_state_debug and self.vehicle.is_at_traffic_light():
            tl_state = self.relevant_traffic_light_state_debug
            if tl_state == carla.TrafficLightState.Red:
                if current_speed_mps > VEHICLE_STOPPED_SPEED_THRESHOLD: reward += PENALTY_TRAFFIC_LIGHT_RED_MOVING
                else: reward += REWARD_TRAFFIC_LIGHT_STOPPED_AT_RED
            elif tl_state == carla.TrafficLightState.Green:
                if current_speed_mps > MIN_FORWARD_SPEED_THRESHOLD: reward += REWARD_TRAFFIC_LIGHT_GREEN_PROCEED
        
        # 8. Proximity Penalty (as before)
        if self.world and self.vehicle: # Simplified proximity check
            vehicle_forward_vector = vehicle_transform.get_forward_vector()
            vehicle_forward_vec_2d = np.array([vehicle_forward_vector.x, vehicle_forward_vector.y])
            norm_fwd = np.linalg.norm(vehicle_forward_vec_2d); 
            if norm_fwd > 1e-4: vehicle_forward_vec_2d /= norm_fwd
            for other_vehicle in self.world.get_actors().filter('vehicle.*'):
                if other_vehicle.id == self.vehicle.id: continue
                other_location = other_vehicle.get_location()
                dist_xyz = current_location.distance(other_location)
                if dist_xyz < PROXIMITY_THRESHOLD_VEHICLE + 5.0:
                    vec_to_other = other_location - current_location
                    vec_to_other_2d = np.array([vec_to_other.x, vec_to_other.y])
                    dist_2d = np.linalg.norm(vec_to_other_2d)
                    if dist_2d < PROXIMITY_THRESHOLD_VEHICLE and dist_2d > 0.1:
                        if norm_fwd > 1e-4: vec_to_other_2d /= (dist_2d + 1e-4)
                        dot_product = np.dot(vehicle_forward_vec_2d, vec_to_other_2d)
                        if dot_product > 0.707:
                            reward += PENALTY_PROXIMITY_VEHICLE_FRONT
                            self.proximity_penalty_flag_debug = True; break

        # More detailed debug log for reward components:
        log_msg = (
            f"R: {reward:.2f} | "
            f"FSpd: {self.forward_speed_debug:.2f}, R_spd: {self.forward_speed_debug * REWARD_SPEED_MULTIPLIER if self.forward_speed_debug > MIN_FORWARD_SPEED_THRESHOLD and not is_reversing_action else 0.0:.2f} | "
            f"DSG: {current_dist_to_target:.1f}, R_nav: {(previous_location.distance(self.target_waypoint.location) - current_dist_to_target) * REWARD_DISTANCE_FACTOR if self.target_waypoint and previous_location else 0.0:.2f} | "
            f"TL: {self.relevant_traffic_light_state_debug}, R_tl: <see_logic> | " # TL reward is conditional
            f"Coll: {'Y' if self.collision_flag_debug else 'N'} ({PENALTY_COLLISION if self.collision_flag_debug else 0}) | "
            f"Prox: {'Y' if self.proximity_penalty_flag_debug else 'N'} ({PENALTY_PROXIMITY_VEHICLE_FRONT if self.proximity_penalty_flag_debug else 0}) | "
            f"StepP: {PENALTY_PER_STEP}"
        )
        self.logger.debug(log_msg)
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
        
        if self.collision_info.get('count', 0) > 0:
            self.logger.info(f"Collision detected! Episode done. Intensity: {self.collision_info['last_intensity']:.2f}, Other: {self.collision_info['last_other_actor_type']}") 
            # self.collision_info['count'] = 0 # Reset here or after processing in step. Resetting in step is better.
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
        if 'semantic_camera' in self.observation_space.spaces:
            seg_transform = carla.Transform(carla.Location(x=1.5, y=-0.1, z=2.4)) # Default, can be customized
            seg_camera_actor = camera_handler.setup_semantic_segmentation_camera(
                self.world, self.vehicle, weakref.ref(self),
            image_width=self.image_width, image_height=self.image_height,
                fov=self.fov, sensor_tick=str(self.timestep), transform=seg_transform
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
        
        # Lane Invasion Sensor
        lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        def lane_invasion_callback(event):
            me = world_weak_ref()
            if me:
                me.latest_sensor_data['lane_invasion_event'] = event
        lane_invasion_sensor.listen(lane_invasion_callback)
        self.sensor_list.append(lane_invasion_sensor)
        self.logger.info(f"Spawned Lane Invasion Sensor: {lane_invasion_sensor.id}")

        # Dedicated RGB Camera for Pygame Display (if enabled)
        if self.enable_pygame_display:
            pygame_cam_bp = camera_handler._setup_camera_blueprint( 
                self.world.get_blueprint_library(), 'sensor.camera.rgb',
                self.pygame_window_width, self.pygame_window_height, self.fov, str(self.timestep) # Use instance attributes
            )
            pygame_display_camera_actor = self.world.spawn_actor(
                pygame_cam_bp, 
                self.pygame_display_camera_transform, 
                attach_to=self.vehicle
            )
            def pygame_camera_callback(data):
                me = world_weak_ref()
                if me:
                    me.latest_sensor_data['pygame_rgb_camera'] = data
            pygame_display_camera_actor.listen(pygame_camera_callback)
            self.sensor_list.append(pygame_display_camera_actor)
            self.logger.info(f"Spawned Pygame Display RGB Camera: {pygame_display_camera_actor.id} at {self.pygame_display_camera_transform}")
        
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
        self.collision_info = {'count': 0, 'last_intensity': 0.0, 'last_other_actor_id': "N/A", 'last_other_actor_type': "N/A"} # Reset collision info

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
        if self.world is not None and hasattr(self.world, 'get_settings'): 
            settings = self.world.get_settings()
            if settings.synchronous_mode: # Only revert if it was in sync mode
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
        
        self.world = None 
        self.logger.info("Closed CARLA environment and destroyed actors.")

        if self.visualizer:
            self.visualizer.close()
            self.logger.info("Pygame visualizer closed.")
            self.visualizer = None

    @property
    def action_space(self):
        if self._action_space is None:
            raise NotImplementedError("Action space not defined yet.")
        return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is None:
            raise NotImplementedError("Observation space not defined yet.")
        return self._observation_space

    def get_sensor_summary(self) -> OrderedDict:
        summary = OrderedDict()
        if not self.sensor_list:
            summary["Sensor Info"] = "(No sensors attached)"
            return summary

        sensor_counts = Counter()
        for sensor_actor in self.sensor_list:
            if sensor_actor and hasattr(sensor_actor, 'type_id'):
                sensor_counts[sensor_actor.type_id] += 1
            else:
                sensor_counts["unknown_or_None"] += 1
        
        if not sensor_counts:
            summary["Sensor Info"] = "(Sensor list processed, no types found)"
            return summary

        total_sensors = sum(sensor_counts.values())
        summary[f"Total Sensors"] = f"{total_sensors}" # Include total count in header

        for type_id, count in sorted(sensor_counts.items()):
            parts = type_id.split('.')
            raw_category = parts[1] if len(parts) > 1 else ""
            raw_specific_type = " ".join(parts[2:]) if len(parts) > 2 else ""

            # Prioritize specific, known pretty names
            if type_id == 'sensor.camera.rgb':
                formatted_name = "RGB Camera"
            elif type_id == 'sensor.camera.depth':
                formatted_name = "Depth Camera"
            elif type_id == 'sensor.camera.semantic_segmentation':
                formatted_name = "Semantic Seg. Camera"
            elif type_id == 'sensor.lidar.ray_cast':
                formatted_name = "LIDAR"
            elif type_id == 'sensor.other.gnss':
                formatted_name = "GNSS Sensor"
            elif type_id == 'sensor.other.imu':
                formatted_name = "IMU Sensor"
            elif type_id == 'sensor.other.radar':
                formatted_name = "RADAR Sensor"
            elif type_id == 'sensor.other.lane_invasion':
                formatted_name = "Lane Invasion Detector"
            elif type_id == 'sensor.other.collision':
                formatted_name = "Collision Detector"
            # Generic fallback formatting if no specific match
            elif raw_category and raw_specific_type:
                formatted_name = f"{raw_category.capitalize()} {raw_specific_type.replace('_', ' ').capitalize()}"
            elif raw_category:
                formatted_name = raw_category.capitalize()
            else:
                formatted_name = type_id # Absolute fallback
            
            summary[formatted_name] = count
        return summary

    def _format_time(self, seconds: float) -> str:
        """Formats seconds into hh:mm:ss string."""
        if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
            return "00:00:00"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _render_pygame(self): 
        if not self.visualizer or not self.visualizer.is_active:
            return

        pygame_cam_image = self.latest_sensor_data.get('pygame_rgb_camera')
        debug_info_dict = OrderedDict()

        debug_info_dict["Server FPS"] = "N/A (Sync Mode)"

        if self.vehicle:
            debug_info_dict["Vehicle Model"] = self._format_vehicle_model_name(self.vehicle.type_id)
            if self.map:
                 debug_info_dict["Map"] = self.map.name.split('/')[-1]
            else:
                 debug_info_dict["Map"] = "N/A"
            
            elapsed_seconds = self.world.get_snapshot().timestamp.elapsed_seconds
            debug_info_dict["Simulation Time"] = self._format_time(elapsed_seconds)
            debug_info_dict["Speed (km/h)"] = f"{self.forward_speed_debug * 3.6:.2f}"
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            debug_info_dict["Location (X,Y,Z)"] = f"({vehicle_location.x:.2f}, {vehicle_location.y:.2f}, {vehicle_location.z:.2f})"
            fwd_vec = vehicle_transform.get_forward_vector()
            heading_rad = math.atan2(fwd_vec.y, fwd_vec.x)
            heading_deg = math.degrees(heading_rad)
            debug_info_dict["Compass (Deg)"] = f"{heading_deg:.1f}"
            imu_data = self.latest_sensor_data.get('imu')
            if imu_data:
                accel = imu_data.accelerometer; gyro = imu_data.gyroscope
                debug_info_dict["Acceleration"] = f"({accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f})"
                debug_info_dict["Gyroscope"] = f"({gyro.x:.2f}, {gyro.y:.2f}, {gyro.z:.2f})"
            else:
                debug_info_dict["Acceleration"] = "N/A"; debug_info_dict["Gyroscope"] = "N/A"
            control = self.vehicle.get_control()
            debug_info_dict["Throttle"] = f"{control.throttle:.2f}"
            debug_info_dict["Steer"] = f"{control.steer:.2f}"
            debug_info_dict["Brake"] = f"{control.brake:.2f}"
            gear = "N"; 
            if control.reverse: gear = "R"
            elif self.forward_speed_debug > 0.1 or control.throttle > 0.1: gear = "D"
            debug_info_dict["Gear"] = gear

        else: # If no vehicle
            debug_info_dict["Vehicle Model"] = "N/A"
            debug_info_dict["Map"] = self.map.name.split('/')[-1] if self.map else "N/A"
            elapsed_seconds = self.world.get_snapshot().timestamp.elapsed_seconds if self.world else 0
            debug_info_dict["Simulation Time"] = self._format_time(elapsed_seconds) 
            for k in ["Speed (km/h)", "Location (X,Y,Z)", "Compass (Deg)", "Acceleration", "Gyroscope", "Throttle", "Steer", "Brake", "Gear"]:
                debug_info_dict[k] = "N/A"

        debug_info_dict["Episode | Step"] = f"{self.episode_count_debug} | {self.step_count_debug}"
        debug_info_dict["RL Action"] = self.current_action_debug
        debug_info_dict["Step Reward"] = f"{self.step_reward_debug:.2f}"
        debug_info_dict["Episode Score"] = f"{self.episode_score_debug:.2f}"
        debug_info_dict["Dist to Goal (m)"] = f"{self.dist_to_goal_debug:.2f}"
        debug_info_dict["Traffic Light"] = self.relevant_traffic_light_state_debug 
        debug_info_dict["Collision"] = self.collision_flag_debug 
        debug_info_dict["Proximity Penalty"] = self.proximity_penalty_flag_debug 
        debug_info_dict["Term Reason"] = self.last_termination_reason_debug

        if not self.visualizer.render(pygame_cam_image, debug_info_dict):
            self.enable_pygame_display = False 
            self.visualizer.close() 
            self.visualizer = None
            self.logger.info("Pygame display has been disabled.")

    def _save_specific_sensor_data(self):
        """Saves data for specified sensors to disk if interval is met."""
        if not self.save_sensor_data_enabled or not self.current_run_sensor_save_path: 
            return
        if self.step_count_debug == 0 or self.step_count_debug % self.sensor_save_interval != 0:
            return

        self.logger.debug(f"Saving sensor data for Ep: {self.episode_count_debug}, Step: {self.step_count_debug}")

        for sensor_key, data in self.latest_sensor_data.items():
            if data is None or sensor_key not in self._sensor_save_dirs:
                continue

            save_dir = self._sensor_save_dirs[sensor_key]
            filename_base = f"ep{self.episode_count_debug:04d}_step{self.step_count_debug:05d}"

            try:
                if isinstance(data, carla.Image) and sensor_key in ['rgb_camera', 'depth_camera', 'semantic_camera']:
                    data.save_to_disk(os.path.join(save_dir, f"{filename_base}.png"))
                    # For depth, consider saving raw data or converting appropriately if png is not ideal
                    # if sensor_key == 'depth_camera': data.save_to_disk(os.path.join(save_dir, f"{filename_base}.raw"), carla.ColorConverter.Raw)
                elif isinstance(data, carla.LidarMeasurement) and sensor_key == 'lidar':
                    data.save_to_disk(os.path.join(save_dir, f"{filename_base}.ply"))
                # Add more sensor types here (IMU, GNSS, RADAR to CSV/JSON)
                # elif sensor_key == 'imu': ...
            except Exception as e:
                self.logger.error(f"Error saving data for sensor {sensor_key}: {e}")

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