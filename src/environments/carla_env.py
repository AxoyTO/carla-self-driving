import carla
import random
import numpy as np
import time # For sensor data synchronization
import weakref # For sensor callbacks
import logging # Import logging module
from collections import OrderedDict # For Dict space if needed, though gymnasium handles it
from collections import Counter # For counting sensor types
import math # For compass
import os
from datetime import datetime, timedelta # For timestamped save directory and timing tracking
from typing import Optional, Tuple

import config

from .base_env import BaseEnv
import gymnasium as gym 
from gymnasium import spaces 

from .sensors import camera_handler 
from .sensors import lidar_handler 
from .sensors import radar_handler 
from .sensors import gnss_imu_handler 
from .sensors import collision_handler 
from utils.pygame_visualizer import PygameVisualizer 
from .reward_calculator import RewardCalculator 
from utils.open3d_visualizer import Open3DLidarVisualizer
from .traffic_manager import TrafficManager

class CarlaEnv(BaseEnv):
    def __init__(self, host='localhost', port=2000, town='Town03', timestep=0.05, time_scale=1.0, \
                 image_size=(config.CARLA_DEFAULT_IMAGE_WIDTH, config.CARLA_DEFAULT_IMAGE_HEIGHT), 
                 fov=90, discrete_actions=True, num_actions=6, \
                 log_level=logging.INFO,
                 lidar_params=None, radar_params=None,
                 enable_pygame_display=False, 
                 pygame_window_width=1920, pygame_window_height=1080,
                 save_sensor_data=False,      
                 sensor_save_base_path="./sensor_capture", 
                 sensor_save_interval=100, 
                 curriculum_phases: list = None ): 
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

        # Timing tracking variables
        self.episode_start_time = None
        self.phase_start_time = None
        self.total_phase_time = timedelta(0)
        self.total_episode_time = timedelta(0)
        self.episode_count = 0

        self.host = host
        self.port = port
        self.town = town
        self.timestep = timestep 
        self.time_scale = time_scale
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.fov = fov
        self.discrete_actions = discrete_actions
        self.num_actions = num_actions

        self.latest_sensor_data = {
            'rgb_camera': None,      
            'left_rgb_camera': None,  
            'right_rgb_camera': None, 
            'rear_rgb_camera': None,  
            
            'display_rgb_camera': None,   
            'display_left_rgb_camera': None,
            'display_right_rgb_camera': None,
            'display_rear_rgb_camera': None, 

            'spectator_camera': None, 
            'depth_camera': None,      
            'semantic_camera': None, 
            'display_depth_camera': None,
            'display_semantic_camera': None,

            'lidar': None,
            'semantic_lidar': None, # New key for Semantic LIDAR data
            'collision': None, 
            'gnss': None, 
            'imu': None,
            'radar': None,
            'lane_invasion_event': None 
        }
        self.collision_info = {
            'count': 0, 
            'last_intensity': 0.0, 
            'last_other_actor_id': "N/A", 
            'last_other_actor_type': "N/A"
        }
        self._spawn_points = []
        self.relevant_traffic_light_state_debug = None
        self.traffic_light_stop_line_waypoint = None 
        self.episode_count_debug = 0
        self.step_count_debug = 0
        self.current_action_debug = "N/A (Reset)"
        self.step_reward_debug = 0.0
        self.episode_score_debug = 0.0
        self.forward_speed_debug = 0.0
        self.dist_to_goal_debug = float('inf')
        self.collision_flag_debug = False 
        self.proximity_penalty_flag_debug = False 
        self.last_termination_reason_debug = "N/A"
        self.on_sidewalk_debug_flag = False # New flag for sidewalk state
        self.enable_pygame_display = enable_pygame_display
        self.visualizer = None 
        if self.enable_pygame_display:
            self.visualizer = PygameVisualizer(
                window_width=self.pygame_window_width, 
                window_height=self.pygame_window_height,
                caption=f"CARLA RL Agent View",
                carla_env_ref=weakref.ref(self) 
            )
        self.radar_to_vehicle_transform = carla.Transform(carla.Location(x=2.0, z=1.0)) # Store this fixed relative transform
        self.lidar_config = self._create_lidar_config(lidar_params)
        self.semantic_lidar_config = self.lidar_config 
        self.radar_config = self._create_radar_config(radar_params)
        self.pygame_display_camera_transform = carla.Transform(carla.Location(x=-5.5, y=0, z=3.5), carla.Rotation(pitch=-15))
        if self.discrete_actions:
            self._action_space = spaces.Discrete(self.num_actions) 
        else:
            self._action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        obs_spaces = OrderedDict()
        camera_obs_spaces = camera_handler.get_camera_observation_spaces(self.image_width, self.image_height)
        obs_spaces.update(camera_obs_spaces) 
        rgb_camera_shape = (3, self.image_height, self.image_width)
        if 'left_rgb_camera' not in obs_spaces:
            obs_spaces['left_rgb_camera'] = spaces.Box(low=0, high=255, shape=rgb_camera_shape, dtype=np.uint8)
        if 'right_rgb_camera' not in obs_spaces:
            obs_spaces['right_rgb_camera'] = spaces.Box(low=0, high=255, shape=rgb_camera_shape, dtype=np.uint8)
        if 'rear_rgb_camera' not in obs_spaces:
            obs_spaces['rear_rgb_camera'] = spaces.Box(low=0, high=255, shape=rgb_camera_shape, dtype=np.uint8)
        obs_spaces['lidar'] = lidar_handler.get_lidar_observation_space(
            num_points=self.lidar_config['num_points_processed']
        )
        obs_spaces['semantic_lidar'] = lidar_handler.get_semantic_lidar_observation_space(
            num_points=self.semantic_lidar_config.get('num_points_processed', config.CARLA_PROCESSED_LIDAR_NUM_POINTS) 
        )
        obs_spaces['gnss'] = gnss_imu_handler.get_gnss_observation_space()
        obs_spaces['imu'] = gnss_imu_handler.get_imu_observation_space()
        obs_spaces['radar'] = radar_handler.get_radar_observation_space(
            max_detections=self.radar_config['max_detections_processed']
        )
        self._observation_space = spaces.Dict(obs_spaces)
        self.connect()
        if self.world:
            self._spawn_points = self.world.get_map().get_spawn_points()
            if not self._spawn_points:
                self.logger.warning("No spawn points found in the map during init!")
        self.target_speed_kmh = 40.0 
        self.current_action_for_reward = None 
        if curriculum_phases is None:
            self.curriculum_phases = config.CARLA_DEFAULT_CURRICULUM_PHASES
        else:
            self.curriculum_phases = curriculum_phases
        self.current_curriculum_phase_idx = 0
        self.episode_in_current_phase = 0
        self.total_episodes_for_curriculum_tracking = 0
        self.phase0_spawn_point_idx = 41 
        self.phase0_target_distance_m = 50.0 
        self.save_sensor_data_enabled = save_sensor_data
        self.sensor_save_base_path = sensor_save_base_path
        self.sensor_save_interval = sensor_save_interval
        self.current_run_sensor_save_path = ""
        self._sensor_save_dirs = {}
        if self.save_sensor_data_enabled:
            self._setup_sensor_saving_directories()
        self.reward_calculator = RewardCalculator(
            reward_configs={},
            target_speed_kmh=self.target_speed_kmh,
            curriculum_phases=self.curriculum_phases, 
            carla_env_ref=weakref.ref(self) 
        )
        self.o3d_lidar_vis = None
        if self.enable_pygame_display: 
            try:
                self.o3d_lidar_vis = Open3DLidarVisualizer()
            except Exception as e:
                self.logger.error(f"Failed to initialize Open3D Lidar Visualizer: {e}. Open3D features will be unavailable.")
                self.o3d_lidar_vis = None

        # --- Traffic Manager Initialization ---
        try:
            self.traffic_manager = TrafficManager(self.client, self.world, weakref.ref(self), log_level=log_level, time_scale=time_scale)
        except Exception as e:
            self.logger.error(f"Failed to initialize TrafficManager: {e}. Traffic features will be unavailable.")
            self.traffic_manager = None

    def connect(self):
        try:
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0) # seconds
            self.world = self.client.load_world(self.town)
            
            # Set synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.timestep
            
            # Apply time scale factor
            if self.time_scale != 1.0:
                self.logger.info(f"Setting simulation time scale to {self.time_scale}x")
                try:
                    # In CARLA, time scale is controlled by adjusting fixed_delta_seconds
                    # A larger value makes simulation run faster (more simulation time per real second)
                    # Adjust fixed_delta_seconds directly proportional to time_scale
                    if self.time_scale > 0:
                        settings.fixed_delta_seconds = self.timestep * self.time_scale
                except Exception as e:
                    self.logger.warning(f"Failed to apply time scale: {e}")
            
            self.world.apply_settings(settings)

            # Log effective simulation speed
            if self.time_scale != 1.0:
                effective_timestep = settings.fixed_delta_seconds
                self.logger.info(f"Effective simulation timestep: {effective_timestep:.4f}s (normal: {self.timestep:.4f}s)")
                self.logger.info(f"Simulation running at {self.time_scale:.1f}x speed")

            self.map = self.world.get_map()
            self.logger.info(f"Connected to CARLA server at {self.host}:{self.port} and loaded {self.town}") # Changed from print
        except Exception as e:
            self.logger.error(f"Error connecting to CARLA or loading world: {e}", exc_info=True) # Changed from print, added exc_info
            raise

    def reset(self):
        self._advance_curriculum_phase()
        current_phase_config = self.curriculum_phases[self.current_curriculum_phase_idx]
        phase_name = current_phase_config["name"]
        spawn_config_type = current_phase_config["spawn_config"]
        self.logger.info(f"Phase '{phase_name}' (Ep {self.episode_in_current_phase}/{current_phase_config['episodes']}), Spawn: {spawn_config_type}, Total Episodes: {self.total_episodes_for_curriculum_tracking}, Total Time: {self.total_episode_time}")

        # Record episode start time
        self.episode_start_time = datetime.now()
        self.episode_count += 1

        # Destroy any existing NPCs and ego vehicle/sensors
        if self.traffic_manager:
            self.traffic_manager.destroy_npcs()
        time.sleep(0.1 / max(1.0, self.time_scale)); self._destroy_actors(); time.sleep(0.2 / max(1.0, self.time_scale))
        for key in self.latest_sensor_data: self.latest_sensor_data[key] = None
        self.latest_sensor_data['lane_invasion_event'] = None
        self.collision_info = {'count': 0, 'last_intensity': 0.0, 'last_other_actor_id': "N/A", 'last_other_actor_type': "N/A"}

        if not self._spawn_points:
            if self.world and self.map: self._spawn_points = self.map.get_spawn_points()
            if not self._spawn_points: raise RuntimeError("No spawn points in map for reset.")

        start_spawn_point_transform, self.target_waypoint = self._determine_spawn_and_target(spawn_config_type)
        
        if start_spawn_point_transform is None or self.target_waypoint is None or self.target_waypoint.transform is None:
            self.logger.critical("CRITICAL FALLBACK: Failed to set a valid spawn or target. Defaulting to first spawn point for both.")
            if not self._spawn_points: raise RuntimeError("No spawn points available for critical fallback.")
            start_spawn_point_transform = self._spawn_points[0]
            self.target_waypoint = self.map.get_waypoint(start_spawn_point_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if self.target_waypoint is None: 
                self.target_waypoint = self.map.get_waypoint(start_spawn_point_transform.location) # Any waypoint
            if self.target_waypoint is None: 
                raise RuntimeError("Cannot get any waypoint for the first spawn point in critical fallback.")

        self.logger.debug(f"Reset Target: {self.target_waypoint.transform.location}, Start: {start_spawn_point_transform.location}")

        self._spawn_vehicle(spawn_point_transform=start_spawn_point_transform)
        self._setup_sensors() 
        self._position_spectator_camera()
        
        if self.world and self.vehicle: self.world.tick() 
        else: raise RuntimeError("World or Vehicle not initialized properly before first tick in reset.")
        
        observation = self._synchronize_initial_observation()

        self.previous_location = self.vehicle.get_location() if self.vehicle else None
        self.episode_count_debug += 1; self.step_count_debug = 0; self.episode_score_debug = 0.0
        self.current_action_debug = "N/A (Reset)"; self.step_reward_debug = 0.0
        if self.vehicle:
            self.forward_speed_debug = 0.0 
            current_loc = self.vehicle.get_location()
            if self.target_waypoint and current_loc: # target_waypoint is now a waypoint object
                 self.dist_to_goal_debug = current_loc.distance(self.target_waypoint.transform.location)
            else: self.dist_to_goal_debug = float('inf')
        else:
            self.dist_to_goal_debug = float('inf'); self.forward_speed_debug = 0.0
        
        self.relevant_traffic_light_state_debug = None
        self.collision_flag_debug = False
        self.proximity_penalty_flag_debug = False
        self.on_sidewalk_debug_flag = False # Reset sidewalk flag (ensure this is the only reset for this flag in reset())
        
        if self.enable_pygame_display: self._render_pygame()

        # Spawn traffic NPCs for current curriculum phase
        if self.traffic_manager:
            traffic_config = current_phase_config.get("traffic_config", {"type": "none"})
            try:
                self.traffic_manager.spawn_npcs(traffic_config)
            except Exception as e:
                self.logger.error(f"TrafficManager.spawn_npcs failed: {e}")

        return observation, {}

    def step(self, action):
        if self.vehicle is None:
            raise RuntimeError("Vehicle not spawned. Call reset() first.")

        self.current_action_debug = str(action) 
        self.step_count_debug += 1
        self.current_action_for_reward = action 
        self._apply_action(action)
        self.world.tick()

        raw_collision_event = self.latest_sensor_data.get('collision')
        if raw_collision_event:
            self._process_collision_event(raw_collision_event)
            self.latest_sensor_data['collision'] = None

        self._save_specific_sensor_data()

        current_location = self.vehicle.get_location() if self.vehicle else None
        observation = self._get_observation()
        
        # Update dist_to_goal_debug every step for HUD
        if self.target_waypoint and current_location and self.target_waypoint.transform:
            self.dist_to_goal_debug = current_location.distance(self.target_waypoint.transform.location)
        elif self.target_waypoint: # If no current_location but target exists
            self.dist_to_goal_debug = float('inf') # Or some other indicator
        # If no target_waypoint, dist_to_goal_debug remains as set in reset (likely inf or previous value)

        self.collision_flag_debug = False 
        self.proximity_penalty_flag_debug = False
        self.on_sidewalk_debug_flag = False # Reset sidewalk flag

        reward = self._calculate_reward(current_location, self.previous_location)
        
        # Trigger collision notification - uses self.collision_flag_debug set in _calculate_reward
        if self.collision_flag_debug and self.visualizer and self.enable_pygame_display:
            notif_text = f"COLLISION! Int: {self.collision_info['last_intensity']:.2f}"
            if self.collision_info['last_other_actor_type'] != "Environment":
                notif_text += f" | Hit: {self.collision_info['last_other_actor_type']} (ID: {self.collision_info['last_other_actor_id']})"
            else:
                notif_text += f" | Hit: {self.collision_info['last_other_actor_type']}"
            self.visualizer.add_notification(notif_text, duration_seconds=4.0, color=(255, 20, 20)) # Bright Red

        # New: Trigger sidewalk notification
        if self.on_sidewalk_debug_flag and self.visualizer and self.enable_pygame_display:
            self.visualizer.add_notification("SIDEWALK! Critical Event.", duration_seconds=4.0, color=(255, 0, 255)) # Magenta color

        self.step_reward_debug = reward 
        self.episode_score_debug += reward 

        terminated, term_info = self._check_done(current_location)
        self.last_termination_reason_debug = term_info.get("termination_reason", "terminated") if terminated else "Running"

        # Track episode end time if the episode is terminated
        if terminated and self.episode_start_time:
            episode_duration = datetime.now() - self.episode_start_time
            self.total_episode_time += episode_duration
            avg_episode_time = self.total_episode_time / self.episode_count if self.episode_count else timedelta(0)
            self.logger.info(f"Episode {self.episode_count} completed in {episode_duration.total_seconds():.2f}s "
                           f"(Reason: {self.last_termination_reason_debug}, "
                           f"Score: {self.episode_score_debug:.2f}, "
                           f"Avg time: {avg_episode_time.total_seconds():.2f}s)")

        # ADDED: Reset collision_info count here, after it has been used by _check_done()
        if self.collision_info.get('count', 0) > 0:
            self.collision_info['count'] = 0
            self.logger.debug("Collision count reset in step() after _check_done.")

        # Update Open3D Lidar Visualizer if active
        if self.o3d_lidar_vis and self.o3d_lidar_vis.is_active():
            if self.vehicle and self.latest_sensor_data.get('lidar'):
                if not self.o3d_lidar_vis.update_data(self.latest_sensor_data['lidar'], self.vehicle.get_transform()):
                    # Window was closed by user, deactivate it
                    self.logger.info("Open3D Lidar window closed by user.")
                    self.o3d_lidar_vis.close() # Ensure it's marked as inactive internally
            # else: # No vehicle or no lidar data, Open3D window will just show last state or be empty
                # self.o3d_lidar_vis.update_data(None, None) # Optionally clear it

        lane_event = self.latest_sensor_data.get('lane_invasion_event')
        if lane_event and self.visualizer and self.enable_pygame_display:
            crossed_lanes_text = []
            for marking in lane_event.crossed_lane_markings:
                crossed_lanes_text.append(str(marking.type).upper())
            if crossed_lanes_text:
                self.visualizer.add_notification(f"Lane Invasion: {', '.join(crossed_lanes_text)}", duration_seconds=2.0, color=(255, 165, 0))
            self.latest_sensor_data['lane_invasion_event'] = None 

        self.previous_location = current_location 
        # self._position_spectator_camera() # No longer call in step
        
        if self.enable_pygame_display:
            self._render_pygame()

        truncated = False 
        info = term_info 

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action):
        if self.vehicle is None:
            self.logger.warning("Cannot apply action: vehicle is not spawned or is None.")
            self.current_action_debug = "No Vehicle"
            return

        control = carla.VehicleControl()
        action_name = "Unknown"
        action_num_str = str(action) # Default for continuous or unmapped
        
        if self.discrete_actions:
            action_num_str = str(action) # Keep original action number for display
            if action == 0: 
                control.throttle = 0.75; control.steer = 0.0; control.brake = 0.0; control.reverse = False
                action_name = "Fwd-Fast"
            elif action == 1: 
                control.throttle = 0.5; control.steer = -0.5; control.brake = 0.0; control.reverse = False
                action_name = "Fwd-Left"
            elif action == 2: 
                control.throttle = 0.5; control.steer = 0.5; control.brake = 0.0; control.reverse = False
                action_name = "Fwd-Right"
            elif action == 3: 
                control.throttle = 0.0; control.steer = 0.0; control.brake = 1.0; control.reverse = False
                action_name = "Brake"
            elif action == 4: 
                control.throttle = 0.3; control.steer = 0.0; control.brake = 0.0; control.reverse = False
                action_name = "Coast"
            elif action == 5: 
                control.throttle = 0.3; control.steer = 0.0; control.brake = 0.0; control.reverse = True
                action_name = "Reverse"
            else:
                self.logger.warning(f"Unknown discrete action: {action}")
                control.throttle = 0.0; control.steer = 0.0 # Default to no motor action
                action_name = f"UnknownDiscrete"
            self.current_action_debug = f"{action_name} ({action_num_str})"
        else: # Continuous actions
            control.throttle = float(max(0, action[0]))
            control.brake = float(max(0, -action[0]))
            control.steer = float(action[1])
            # For continuous, self.current_action_debug can show the raw values
            self.current_action_debug = f"T:{control.throttle:.2f} S:{control.steer:.2f} B:{control.brake:.2f}"
            
        self.vehicle.apply_control(control)

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

        # Process new cameras
        for cam_key in ['left_rgb_camera', 'right_rgb_camera', 'rear_rgb_camera']:
            if cam_key in self._observation_space.spaces:
                raw_cam_data = self.latest_sensor_data.get(cam_key)
                obs_data[cam_key] = camera_handler.process_rgb_camera_data(
                    raw_cam_data,
                    self._observation_space[cam_key].shape
                )
                if debug_sensor_data_once and raw_cam_data:
                    self.logger.debug(f"Sensor Verify - {cam_key}: Processed shape={obs_data[cam_key].shape}, dtype={obs_data[cam_key].dtype}, Raw frame={raw_cam_data.frame}")
            else: # Should not happen if space is defined
                obs_data[cam_key] = np.zeros(self._observation_space[cam_key].shape, dtype=self._observation_space[cam_key].dtype)

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
            
        # LIDAR (Standard)
        raw_lidar = self.latest_sensor_data.get('lidar')
        obs_data['lidar'] = lidar_handler.process_lidar_data(
            raw_lidar,
            num_target_points=self.lidar_config['num_points_processed']
        )
        if debug_sensor_data_once and raw_lidar:
            self.logger.debug(f"Sensor Verify - LIDAR: Processed shape={obs_data['lidar'].shape}, dtype={obs_data['lidar'].dtype}, Raw points={raw_lidar.get_point_count(0) if raw_lidar else 'N/A'}")

        # New: Semantic LIDAR processing
        if 'semantic_lidar' in self.observation_space.spaces:
            raw_semantic_lidar = self.latest_sensor_data.get('semantic_lidar')
            obs_data['semantic_lidar'] = lidar_handler.process_semantic_lidar_data(
                raw_semantic_lidar,
                num_target_points=self.semantic_lidar_config.get('num_points_processed', config.CARLA_PROCESSED_LIDAR_NUM_POINTS)
            )
            if debug_sensor_data_once and raw_semantic_lidar:
                 self.logger.debug(f"Sensor Verify - SemanticLIDAR: Processed shape={obs_data['semantic_lidar'].shape}, dtype={obs_data['semantic_lidar'].dtype}, Raw points={len(raw_semantic_lidar) if raw_semantic_lidar else 'N/A'}")
        
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
        # This method is now a wrapper around the RewardCalculator instance
        
        # Ensure forward_speed_debug is up-to-date before calling
        if self.vehicle and self.vehicle.is_alive:
            velocity_vector = self.vehicle.get_velocity()
            vehicle_transform = self.vehicle.get_transform() 
            vehicle_forward_vector = vehicle_transform.get_forward_vector()
            self.forward_speed_debug = np.dot(
                [velocity_vector.x, velocity_vector.y, velocity_vector.z],
                [vehicle_forward_vector.x, vehicle_forward_vector.y, vehicle_forward_vector.z]
            )
        else:
            self.forward_speed_debug = 0.0

        # Call the external reward calculator
        reward_val, coll_flag, prox_flag, sidewalk_flag = self.reward_calculator.calculate_reward(
            vehicle=self.vehicle,
            current_location=current_location,
            previous_location=previous_location,
            collision_info=self.collision_info, # Pass the dict
            relevant_traffic_light_state=self.relevant_traffic_light_state_debug,
            current_action_for_reward=self.current_action_for_reward,
            forward_speed_debug=self.forward_speed_debug,
            carla_map=self.map,
            target_waypoint=self.target_waypoint
        )

        # Update HUD flags based on calculator's return
        self.collision_flag_debug = coll_flag
        self.proximity_penalty_flag_debug = prox_flag
        self.on_sidewalk_debug_flag = sidewalk_flag # Store sidewalk flag
        
        # Detailed logging can still happen here or be moved into RewardCalculator if preferred
        # For now, let's assume RewardCalculator might do its own internal logging if needed,
        # or CarlaEnv can log the final reward_val.
        # self.logger.debug(f"Calculated Reward: {reward_val:.2f} ...") 
        return reward_val

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
        
        # Check for sidewalk first as it's a critical event
        if self.on_sidewalk_debug_flag:
            self.logger.info("Vehicle on sidewalk! Episode done.")
            info["termination_reason"] = "on_sidewalk"
            return True, info
        
        if self.collision_info.get('count', 0) > 0:
            self.logger.info(f"Collision detected! Episode done. Intensity: {self.collision_info['last_intensity']:.2f}, Other: {self.collision_info['last_other_actor_type']}") 
            # self.collision_info['count'] = 0 # Reset here or after processing in step. Resetting in step is better.
            info["termination_reason"] = "collision"
            return True, info

        # Check if goal reached
        if self.target_waypoint and current_location:
            dist_to_target = current_location.distance(self.target_waypoint.transform.location)
            if dist_to_target < WAYPOINT_REACHED_THRESHOLD:
                # Check if phase requires stopping at goal
                require_stop = False
                try:
                    current_phase_config = self.curriculum_phases[self.current_curriculum_phase_idx]
                    require_stop = current_phase_config.get("require_stop_at_goal", False)
                except Exception:
                    pass

                if not require_stop:
                    info["termination_reason"] = "goal_reached"
                    return True, info
                else:
                    # Need to be nearly stopped
                    speed_mps = abs(self.forward_speed_debug)
                    if speed_mps <= config.STOP_AT_GOAL_SPEED_THRESHOLD:
                        info["termination_reason"] = "goal_reached_and_stopped"
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
        self.logger.debug("Destroyed existing sensors before setting up new ones.")

        if not self.vehicle:
            self.logger.error("Vehicle not spawned before setting up sensors.")
            return

        # blueprint_library = self.world.get_blueprint_library() # No longer needed here, handlers get it
        world_weak_ref = weakref.ref(self) # Pass weakref of carla_env instance

        # Define sensor transforms (relative to the vehicle)
        # These can be customized as needed
        rgb_cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        left_cam_transform = carla.Transform(carla.Location(x=0.75, y=-0.9, z=1.3), carla.Rotation(yaw=-90))
        right_cam_transform = carla.Transform(carla.Location(x=0.75, y=0.9, z=1.3), carla.Rotation(yaw=90))
        rear_cam_transform = carla.Transform(carla.Location(x=-1.8, z=1.6), carla.Rotation(yaw=180))

        # Dimensions for display cameras (e.g., half of pygame window for balance)
        display_cam_width = self.pygame_window_width // 2
        display_cam_height = self.pygame_window_height // 2

        depth_cam_transform = carla.Transform(carla.Location(x=1.5, y=0.1, z=2.4))
        seg_cam_transform = carla.Transform(carla.Location(x=1.5, y=-0.1, z=2.4))
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))      # On the roof
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))      # Front bumper
        gnss_transform = carla.Transform(carla.Location(x=0.0, z=2.0))       # Roof
        imu_transform = carla.Transform(carla.Location(x=0.0, z=1.5))        # Approx CoM
        collision_transform = carla.Transform(carla.Location(x=0.0, z=0.0))  # At vehicle origin

        # RGB Camera (Agent's low-res)
        rgb_camera_actor = camera_handler.setup_rgb_camera(
            self.world, self.vehicle, world_weak_ref,
            image_width=self.image_width, image_height=self.image_height, 
            fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=rgb_cam_transform,
            sensor_key='rgb_camera' 
        )
        if rgb_camera_actor: self.sensor_list.append({'actor': rgb_camera_actor, 'purpose': 'agent'})
        
        # High-res display version of front RGB camera
        if self.enable_pygame_display:
            display_rgb_camera_actor = camera_handler.setup_rgb_camera(
                self.world, self.vehicle, world_weak_ref,
                image_width=display_cam_width, image_height=display_cam_height, 
                fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=rgb_cam_transform,
                sensor_key='display_rgb_camera'
            )
            if display_rgb_camera_actor: self.sensor_list.append({'actor': display_rgb_camera_actor, 'purpose': 'display'})

        # Agent's low-res Left Camera
        left_rgb_camera_actor = camera_handler.setup_rgb_camera(
            self.world, self.vehicle, world_weak_ref, 
            image_width=self.image_width, image_height=self.image_height, 
            fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=left_cam_transform, 
            sensor_key='left_rgb_camera' 
        )
        if left_rgb_camera_actor: self.sensor_list.append({'actor': left_rgb_camera_actor, 'purpose': 'agent'})
        # High-res display version of left RGB camera
        if self.enable_pygame_display:
            display_left_rgb_camera_actor = camera_handler.setup_rgb_camera(
                self.world, self.vehicle, world_weak_ref, 
                image_width=display_cam_width, image_height=display_cam_height, 
                fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=left_cam_transform, 
                sensor_key='display_left_rgb_camera'
            )
            if display_left_rgb_camera_actor: self.sensor_list.append({'actor': display_left_rgb_camera_actor, 'purpose': 'display'})

        # Agent's low-res Right Camera
        right_rgb_camera_actor = camera_handler.setup_rgb_camera(
            self.world, self.vehicle, world_weak_ref, 
            image_width=self.image_width, image_height=self.image_height, 
            fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=right_cam_transform, 
            sensor_key='right_rgb_camera'
        )
        if right_rgb_camera_actor: self.sensor_list.append({'actor': right_rgb_camera_actor, 'purpose': 'agent'})
        # High-res display version of right RGB camera
        if self.enable_pygame_display:
            display_right_rgb_camera_actor = camera_handler.setup_rgb_camera(
                self.world, self.vehicle, world_weak_ref, 
                image_width=display_cam_width, image_height=display_cam_height, 
                fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=right_cam_transform, 
                sensor_key='display_right_rgb_camera'
            )
            if display_right_rgb_camera_actor: self.sensor_list.append({'actor': display_right_rgb_camera_actor, 'purpose': 'display'})

        # Agent's low-res Rear Camera
        rear_rgb_camera_actor = camera_handler.setup_rgb_camera(
            self.world, self.vehicle, world_weak_ref, 
            image_width=self.image_width, image_height=self.image_height, 
            fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=rear_cam_transform, 
            sensor_key='rear_rgb_camera'
        )
        if rear_rgb_camera_actor: self.sensor_list.append({'actor': rear_rgb_camera_actor, 'purpose': 'agent'})
        # High-res display version of rear RGB camera
        if self.enable_pygame_display:
            display_rear_rgb_camera_actor = camera_handler.setup_rgb_camera(
                self.world, self.vehicle, world_weak_ref, 
                image_width=display_cam_width, image_height=display_cam_height, 
                fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=rear_cam_transform, 
                sensor_key='display_rear_rgb_camera'
            )
            if display_rear_rgb_camera_actor: self.sensor_list.append({'actor': display_rear_rgb_camera_actor, 'purpose': 'display'})

        # Depth Camera (agent's low-res version)
        if 'depth_camera' in self.observation_space.spaces:
            depth_camera_actor = camera_handler.setup_depth_camera(
                self.world, self.vehicle, world_weak_ref,
                image_width=self.image_width, image_height=self.image_height,
                fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=depth_cam_transform,
                sensor_key='depth_camera' 
            )
            if depth_camera_actor: self.sensor_list.append({'actor': depth_camera_actor, 'purpose': 'agent'})

            # High-res display version of depth camera
            if self.enable_pygame_display:
                display_depth_camera_actor = camera_handler.setup_depth_camera(
                    self.world, self.vehicle, world_weak_ref,
                    image_width=display_cam_width, image_height=display_cam_height, 
                    fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=depth_cam_transform,
                    sensor_key='display_depth_camera'
                )
                if display_depth_camera_actor: self.sensor_list.append({'actor': display_depth_camera_actor, 'purpose': 'display'})

        # Semantic Segmentation Camera (agent's low-res version)
        if 'semantic_camera' in self.observation_space.spaces:
            seg_camera_actor = camera_handler.setup_semantic_segmentation_camera(
                self.world, self.vehicle, world_weak_ref, 
                image_width=self.image_width, image_height=self.image_height,
                fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=seg_cam_transform,
                sensor_key='semantic_camera' 
            )
            if seg_camera_actor: self.sensor_list.append({'actor': seg_camera_actor, 'purpose': 'agent'})

            # High-res display version of semantic camera
            if self.enable_pygame_display:
                display_semantic_camera_actor = camera_handler.setup_semantic_segmentation_camera(
                    self.world, self.vehicle, world_weak_ref, 
                    image_width=display_cam_width, image_height=display_cam_height, 
                    fov=self.fov, sensor_tick=self._get_scaled_sensor_tick(), transform=seg_cam_transform,
                    sensor_key='display_semantic_camera'
                )
                if display_semantic_camera_actor: self.sensor_list.append({'actor': display_semantic_camera_actor, 'purpose': 'display'})
        
        # LIDAR Sensor
        lidar_actor = lidar_handler.setup_lidar_sensor(
            self.world, self.vehicle, world_weak_ref, 
            lidar_config=self.lidar_config, transform=lidar_transform, sensor_key='lidar'
        )
        if lidar_actor: self.sensor_list.append({'actor': lidar_actor, 'purpose': 'agent'})
        
        # GNSS Sensor
        gnss_actor = gnss_imu_handler.setup_gnss_sensor(
            self.world, self.vehicle, world_weak_ref, 
            sensor_tick=self._get_scaled_sensor_tick(), transform=gnss_transform
        )
        if gnss_actor: self.sensor_list.append({'actor': gnss_actor, 'purpose': 'agent'})
        
        # IMU Sensor
        imu_actor = gnss_imu_handler.setup_imu_sensor(
            self.world, self.vehicle, world_weak_ref, 
            sensor_tick=self._get_scaled_sensor_tick(), transform=imu_transform
        )
        if imu_actor: self.sensor_list.append({'actor': imu_actor, 'purpose': 'agent'})
        
        # RADAR Sensor
        radar_actor = radar_handler.setup_radar_sensor(
            self.world, self.vehicle, world_weak_ref, 
            radar_config=self.radar_config, transform=self.radar_to_vehicle_transform # Use stored relative transform
        )
        if radar_actor: self.sensor_list.append({'actor': radar_actor, 'purpose': 'agent'})
        
        # Collision Sensor
        collision_actor = collision_handler.setup_collision_sensor(
            self.world, self.vehicle, world_weak_ref, transform=collision_transform
        )
        if collision_actor: self.sensor_list.append({'actor': collision_actor, 'purpose': 'agent'})
        
        # Lane Invasion Sensor
        lane_invasion_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        def lane_invasion_callback(event):
            me = world_weak_ref()
            if me:
                me.latest_sensor_data['lane_invasion_event'] = event
        lane_invasion_sensor.listen(lane_invasion_callback)
        self.sensor_list.append({'actor': lane_invasion_sensor, 'purpose': 'agent'})
        self.logger.debug(f"Spawned Lane Invasion Sensor: {lane_invasion_sensor.id}") # Changed to DEBUG

        # Dedicated Spectator Camera for Pygame Display (if enabled) (RENAMED)
        if self.enable_pygame_display:
            spectator_cam_bp = camera_handler._setup_camera_blueprint(
                self.world.get_blueprint_library(), 'sensor.camera.rgb',
                self.pygame_window_width, self.pygame_window_height, self.fov, self._get_scaled_sensor_tick()
            )
            spectator_camera_actor = self.world.spawn_actor(
                spectator_cam_bp, 
                self.pygame_display_camera_transform, # Using the existing transform variable
                attach_to=self.vehicle
            )
            def spectator_camera_callback(data): # This callback is specific to spectator cam
                me = world_weak_ref()
                if me:
                    me.latest_sensor_data['spectator_camera'] = data 
            spectator_camera_actor.listen(spectator_camera_callback)
            if spectator_camera_actor: self.sensor_list.append({'actor': spectator_camera_actor, 'purpose': 'spectator'})
            self.logger.debug(f"Spawned Spectator Camera (for Pygame): {spectator_camera_actor.id} at {self.pygame_display_camera_transform}") # Changed to DEBUG
        
        # New: Semantic LIDAR Sensor
        if 'semantic_lidar' in self.observation_space.spaces:
            semantic_lidar_actor = lidar_handler.setup_semantic_lidar_sensor(
                self.world, self.vehicle, world_weak_ref,
                semantic_lidar_config=self.semantic_lidar_config, # Use its own config if created, else same as lidar_config
                transform=lidar_transform, # Can use the same transform or a different one
                sensor_key='semantic_lidar'
            )
            if semantic_lidar_actor: self.sensor_list.append({'actor': semantic_lidar_actor, 'purpose': 'agent'})
        
        # Tick the world once to let sensors register
        if self.world and self.world.get_settings().synchronous_mode:
            self.world.tick()
        self.logger.debug(f"All sensors set up. Total sensors attached: {len(self.sensor_list)}")

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
            for sensor_entry in self.sensor_list:
                sensor = sensor_entry['actor'] # Get the actual sensor actor
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
                 time.sleep(0.05 / max(1.0, self.time_scale)) # Small delay after vehicle destruction tick, adjusted for time scale

        # Destroy sensor actors in a batch
        if self.client and sensors_to_destroy:
            self.logger.debug(f"Destroying sensor actors by ID: {sensors_to_destroy}")
            try:
                self.client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in sensors_to_destroy], True)
                if self.world and self.world.get_settings().synchronous_mode:
                    self.world.tick() # Tick to process sensor destruction
                    time.sleep(0.05 / max(1.0, self.time_scale)) # Small delay after sensor destruction tick, adjusted for time scale
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

        # Log timing statistics at environment close
        if self.episode_count > 0:
            avg_episode_time = self.total_episode_time / self.episode_count 
            self.logger.info(f"Timing Statistics:")
            self.logger.info(f"- Total episodes: {self.episode_count}")
            self.logger.info(f"- Total episode time: {self.total_episode_time.total_seconds():.2f}s")
            self.logger.info(f"- Average episode time: {avg_episode_time.total_seconds():.2f}s")
            if self.total_phase_time.total_seconds() > 0:
                self.logger.info(f"- Total curriculum phase time: {self.total_phase_time.total_seconds():.2f}s")

        if self.visualizer:
            self.visualizer.close()
            self.logger.info("Pygame visualizer closed.")
            self.visualizer = None
        
        if self.o3d_lidar_vis: # New: Close Open3D visualizer
            self.o3d_lidar_vis.close()
            self.logger.info("Open3D Lidar visualizer closed.")
            self.o3d_lidar_vis = None

        # Clean up traffic NPCs
        if self.traffic_manager:
            self.traffic_manager.destroy_npcs()

    def toggle_o3d_lidar_visualization(self):
        """Toggles the Open3D LIDAR visualization window."""
        if not self.o3d_lidar_vis:
            self.logger.warning("Open3D Lidar Visualizer was not initialized. Cannot toggle.")
            # Optionally try to initialize it here if it failed before and pygame is enabled
            if self.enable_pygame_display:
                try:
                    self.logger.info("Attempting to re-initialize Open3D Lidar Visualizer for toggle.")
                    self.o3d_lidar_vis = Open3DLidarVisualizer()
                except Exception as e:
                    self.logger.error(f"Failed to re-initialize Open3D Lidar Visualizer: {e}")
                    self.o3d_lidar_vis = None
                    return
            else:
                return

        if self.o3d_lidar_vis.is_active():
            self.o3d_lidar_vis.close()
        else:
            if not self.o3d_lidar_vis.activate():
                self.logger.error("Failed to activate Open3D Lidar window.")
                # If activation fails, ensure it's fully closed/reset if needed
                self.o3d_lidar_vis.close() 

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
        agent_sensor_count = 0 # Keep track of agent sensors for the total count
        for sensor_entry in self.sensor_list:
            sensor_actor = sensor_entry['actor']
            sensor_purpose = sensor_entry.get('purpose', 'agent') # Default to agent if no purpose

            if sensor_actor and hasattr(sensor_actor, 'type_id'):
                if sensor_purpose == 'agent':
                    sensor_counts[sensor_actor.type_id] += 1
                    agent_sensor_count +=1
                # Non-agent sensors (display, spectator) are now skipped for counting
            elif sensor_purpose == 'agent': # Only count unknown/None if it was intended as an agent sensor
                sensor_counts["unknown_or_None"] += 1
                agent_sensor_count +=1
        
        if not sensor_counts: # This means no agent sensors were found
            summary["Sensor Info"] = "(No agent sensors to display)"
            # If you still want to show display/spectator sensors are present even if not counted,
            # you could add a generic line here, e.g., if len(self.sensor_list) > 0: summary["Other Sensors"] = "Present"
            return summary

        summary[f"Total Agent Sensors"] = f"{agent_sensor_count}" # Use the dedicated agent sensor count

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
                formatted_name = "Semantic Camera"
            elif type_id == 'sensor.lidar.ray_cast':
                formatted_name = "LIDAR"
            elif type_id == 'sensor.lidar.ray_cast_semantic':
                formatted_name = "Semantic LIDAR"
            elif type_id == 'sensor.other.gnss':
                formatted_name = "GNSS Sensor"
            elif type_id == 'sensor.other.imu':
                formatted_name = "IMU Sensor"
            elif type_id == 'sensor.other.radar':
                formatted_name = "RADAR"
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

    def _get_pygame_debug_info(self) -> OrderedDict:
        """Prepares the dictionary of debug information for Pygame display."""
        debug_info = OrderedDict()
        
        # Block 0: Current View
        current_view_display_name = self.visualizer.get_current_view_display_name() if self.visualizer else "N/A"
        debug_info["Current View"] = current_view_display_name
        
        # Block 1: FPS Info (Client FPS handled by HUD, Server FPS here)
        debug_info["Server FPS"] = "N/A (Sync Mode)" # Or calculate if possible, but typically fixed in sync
        debug_info["Episode | Step"] = f"{self.episode_count_debug} | {self.step_count_debug}"

        # Block 2: Vehicle Details
        if self.vehicle and self.vehicle.is_alive:
            debug_info["Vehicle Model"] = self._format_vehicle_model_name(self.vehicle.type_id)
            debug_info["Map"] = self.map.name.split('/')[-1] if self.map else self.town
            debug_info["Time Scale"] = f"{self.time_scale:.1f}x"
            elapsed_seconds = self.world.get_snapshot().timestamp.elapsed_seconds if self.world else 0.0
            debug_info["Simulation Time"] = self._format_time(elapsed_seconds)
            debug_info["Speed (km/h)"] = f"{int(self.forward_speed_debug * 3.6)}"
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            debug_info["Location (X,Y,Z)"] = f"({vehicle_location.x:.2f}, {vehicle_location.y:.2f}, {vehicle_location.z:.2f})"
            debug_info["_vehicle_world_x"] = vehicle_location.x
            debug_info["_vehicle_world_y"] = vehicle_location.y
            debug_info["_vehicle_world_z"] = vehicle_location.z
            raw_yaw_deg = vehicle_transform.rotation.yaw 
            standard_compass_deg = (90 - raw_yaw_deg + 360) % 360
            cardinal_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
            cardinal_idx = round(standard_compass_deg / 45) % 8 
            cardinal_dir_str = cardinal_directions[cardinal_idx]
            debug_info["Compass"] = f"{standard_compass_deg:.1f}° {cardinal_dir_str}"
            debug_info["_vehicle_world_yaw_rad"] = math.radians(raw_yaw_deg)
            # Pass the RADAR's relative transform to the vehicle for visualization purposes
            debug_info["_radar_to_vehicle_transform"] = self.radar_to_vehicle_transform 
            
            imu_data = self.latest_sensor_data.get('imu')
            accel = imu_data.accelerometer if imu_data else None
            gyro = imu_data.gyroscope if imu_data else None
            debug_info["Acceleration"] = f"({accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f})" if accel else "N/A"
            debug_info["Gyroscope"] = f"({gyro.x:.2f}, {gyro.y:.2f}, {gyro.z:.2f})" if gyro else "N/A"
            control = self.vehicle.get_control()
            debug_info["Throttle"] = f"{control.throttle:.2f}"
            debug_info["Steer"] = f"{control.steer:.2f}"
            debug_info["Brake"] = f"{control.brake:.2f}"
            gear = "N"
            if control.reverse: gear = "R"
            elif self.forward_speed_debug > 0.1 or control.throttle > 0.1: gear = "D"
            debug_info["Gear"] = gear
        else: 
            for k in ["Vehicle Model", "Map", "Time Scale", "Simulation Time", "Speed (km/h)", "Location (X,Y,Z)", "Compass", "Acceleration", "Gyroscope", "Throttle", "Steer", "Brake", "Gear"]:
                debug_info[k] = "N/A"
            if "Compass" not in debug_info: debug_info["Compass"] = "0.0° N"
            debug_info["_vehicle_world_x"] = 0.0; debug_info["_vehicle_world_y"] = 0.0; debug_info["_vehicle_world_z"] = 0.0; debug_info["_vehicle_world_yaw_rad"] = 0.0
            debug_info["_radar_to_vehicle_transform"] = carla.Transform() # Default empty transform if no vehicle
        
        # Block 3: Episode Stats & Action (Reverse order from image for this block)
        debug_info["Dist to Goal (m)"] = f"{self.dist_to_goal_debug:.2f}"
        debug_info["Action"] = self.current_action_debug 
        debug_info["Step Reward"] = f"{self.step_reward_debug:.2f}"
        debug_info["Episode Score"] = f"{self.episode_score_debug:.2f}"
        
        # Block 4: Environment State
        tl_state_str = str(self.relevant_traffic_light_state_debug) if self.relevant_traffic_light_state_debug is not None else "N/A"
        debug_info["Traffic Light"] = tl_state_str
        debug_info["Collision"] = str(self.collision_flag_debug) 
        debug_info["Proximity Penalty"] = str(self.proximity_penalty_flag_debug) 
        debug_info["On Sidewalk"] = str(self.on_sidewalk_debug_flag) # Add sidewalk state to HUD
        debug_info["Term Reason"] = self.last_termination_reason_debug
        return debug_info

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

        current_view_key_for_pygame = self.visualizer.get_current_view_key()
        raw_sensor_object_to_display = self.latest_sensor_data.get(current_view_key_for_pygame)
        debug_info_dict = self._get_pygame_debug_info()

        lidar_range_for_vis = 0.0
        if current_view_key_for_pygame == 'lidar':
            if hasattr(self, 'lidar_config') and 'range' in self.lidar_config:
                lidar_range_for_vis = float(self.lidar_config['range'])
            else:
                self.logger.warning("LIDAR range config not found in CarlaEnv, defaulting in visualizer.")

        if not self.visualizer.render(raw_sensor_object_to_display, 
                                     current_view_key_for_pygame, 
                                     debug_info_dict,
                                     lidar_sensor_range_from_env=lidar_range_for_vis):
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
                elif isinstance(data, carla.SemanticLidarMeasurement) and sensor_key == 'semantic_lidar': # New
                    # SemanticLidarMeasurement also has save_to_disk, saves as .ply with semantic info
                    data.save_to_disk(os.path.join(save_dir, f"{filename_base}_semantic.ply"))
                # Add more sensor types here (IMU, GNSS, RADAR to CSV/JSON)
                # elif sensor_key == 'imu': ...
            except Exception as e:
                self.logger.error(f"Error saving data for sensor {sensor_key}: {e}")

    def _create_lidar_config(self, lidar_params_override: Optional[dict]) -> dict:
        """Creates the LIDAR configuration dictionary, using defaults from config.py and allowing overrides."""
        return {
            'channels': lidar_params_override.get('channels', config.CARLA_DEFAULT_LIDAR_CHANNELS) if lidar_params_override else config.CARLA_DEFAULT_LIDAR_CHANNELS,
            'range': lidar_params_override.get('range', config.CARLA_DEFAULT_LIDAR_RANGE) if lidar_params_override else config.CARLA_DEFAULT_LIDAR_RANGE,
            'points_per_second': lidar_params_override.get('points_per_second', config.CARLA_DEFAULT_LIDAR_POINTS_PER_SECOND) if lidar_params_override else config.CARLA_DEFAULT_LIDAR_POINTS_PER_SECOND,
            'rotation_frequency': lidar_params_override.get('rotation_frequency', config.CARLA_DEFAULT_LIDAR_ROTATION_FREQUENCY) if lidar_params_override else config.CARLA_DEFAULT_LIDAR_ROTATION_FREQUENCY,
            'upper_fov': lidar_params_override.get('upper_fov', config.CARLA_DEFAULT_LIDAR_UPPER_FOV) if lidar_params_override else config.CARLA_DEFAULT_LIDAR_UPPER_FOV,
            'lower_fov': lidar_params_override.get('lower_fov', config.CARLA_DEFAULT_LIDAR_LOWER_FOV) if lidar_params_override else config.CARLA_DEFAULT_LIDAR_LOWER_FOV,
            'num_points_processed': lidar_params_override.get('num_points_processed', config.CARLA_PROCESSED_LIDAR_NUM_POINTS) if lidar_params_override else config.CARLA_PROCESSED_LIDAR_NUM_POINTS,
            'sensor_tick': self._get_scaled_sensor_tick() # Match scaled simulation timestep
        }

    def _create_radar_config(self, radar_params_override: Optional[dict]) -> dict:
        """Creates the RADAR configuration dictionary, using defaults from config.py and allowing overrides."""
        return {
            'range': radar_params_override.get('range', config.CARLA_DEFAULT_RADAR_RANGE) if radar_params_override else config.CARLA_DEFAULT_RADAR_RANGE,
            'horizontal_fov': radar_params_override.get('horizontal_fov', config.CARLA_DEFAULT_RADAR_HORIZONTAL_FOV) if radar_params_override else config.CARLA_DEFAULT_RADAR_HORIZONTAL_FOV,
            'vertical_fov': radar_params_override.get('vertical_fov', config.CARLA_DEFAULT_RADAR_VERTICAL_FOV) if radar_params_override else config.CARLA_DEFAULT_RADAR_VERTICAL_FOV,
            'points_per_second': radar_params_override.get('points_per_second', config.CARLA_DEFAULT_RADAR_POINTS_PER_SECOND) if radar_params_override else config.CARLA_DEFAULT_RADAR_POINTS_PER_SECOND,
            'max_detections_processed': radar_params_override.get('max_detections_processed', config.CARLA_PROCESSED_RADAR_MAX_DETECTIONS) if radar_params_override else config.CARLA_PROCESSED_RADAR_MAX_DETECTIONS,
            'sensor_tick': self._get_scaled_sensor_tick()
        }

    def _setup_sensor_saving_directories(self):
        """Creates directories for saving sensor data if enabled."""
        if not self.save_sensor_data_enabled: # Double check, though called within a condition
            return

        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.current_run_sensor_save_path = os.path.join(self.sensor_save_base_path, f"run_{timestamp}")
        try:
            os.makedirs(self.current_run_sensor_save_path, exist_ok=True)
            self.logger.info(f"Sensor data will be saved in: {self.current_run_sensor_save_path}")
            # Pre-create subdirectories for sensors we intend to save
            # Note: These keys MUST match the keys used in self.latest_sensor_data for easy lookup
            # And also align with observation_space keys if those are used to guide saving.
            # For now, let's assume we want to save for all sensors defined in obs space that are image-like or point-clouds.
            # A more robust approach might be to pass a list of sensors_to_save to __init__.
            sensors_to_log_individually = []
            if 'rgb_camera' in self._observation_space.spaces: sensors_to_log_individually.append('rgb_camera')
            if 'left_rgb_camera' in self._observation_space.spaces: sensors_to_log_individually.append('left_rgb_camera')
            if 'right_rgb_camera' in self._observation_space.spaces: sensors_to_log_individually.append('right_rgb_camera')
            if 'rear_rgb_camera' in self._observation_space.spaces: sensors_to_log_individually.append('rear_rgb_camera')
            if 'depth_camera' in self._observation_space.spaces: sensors_to_log_individually.append('depth_camera')
            if 'semantic_camera' in self._observation_space.spaces: sensors_to_log_individually.append('semantic_camera')
            if 'lidar' in self._observation_space.spaces: sensors_to_log_individually.append('lidar')
            if 'semantic_lidar' in self._observation_space.spaces: sensors_to_log_individually.append('semantic_lidar')
            # Add other types like radar if they have a raw savable format (e.g. to CSV/JSON per step)

            for sensor_key in sensors_to_log_individually:
                s_path = os.path.join(self.current_run_sensor_save_path, sensor_key)
                os.makedirs(s_path, exist_ok=True)
                self._sensor_save_dirs[sensor_key] = s_path
            self.logger.info(f"Created subdirectories for sensor data saving: {list(self._sensor_save_dirs.keys())}")
        except Exception as e:
            self.logger.error(f"Could not create sensor save directories: {e}")
            self.save_sensor_data_enabled = False # Disable if cant create dirs

    def _advance_curriculum_phase(self):
        """Checks and advances the curriculum phase if necessary."""
        self.total_episodes_for_curriculum_tracking += 1
        self.episode_in_current_phase += 1

        current_phase_config = self.curriculum_phases[self.current_curriculum_phase_idx]
        # Check if current phase is completed
        if self.episode_in_current_phase > current_phase_config["episodes"]:
            # Record phase end time and log duration if phase_start_time exists
            if self.phase_start_time:
                phase_duration = datetime.now() - self.phase_start_time
                self.total_phase_time += phase_duration
                self.logger.info(f"Curriculum phase '{current_phase_config['name']}' completed in {phase_duration.total_seconds():.2f}s "
                               f"({self.episode_in_current_phase-1} episodes)")
                
            # Check if there are more phases
            if self.current_curriculum_phase_idx < len(self.curriculum_phases) - 1:
                self.current_curriculum_phase_idx += 1
                self.episode_in_current_phase = 1 # Reset episode count for the new phase
                new_phase_config = self.curriculum_phases[self.current_curriculum_phase_idx]
                self.logger.info(f"Advanced to Curriculum Phase: {new_phase_config['name']} (Ep {self.episode_in_current_phase}/{new_phase_config['episodes']})")
                
                # Set start time for the new phase
                self.phase_start_time = datetime.now()
            elif self.episode_in_current_phase == current_phase_config["episodes"] + 1: # Log once after completing the last phase
                self.logger.info(f"Curriculum completed. Continuing with last phase settings: {current_phase_config['name']}")
        # Initialize phase_start_time if it's not set yet (for the very first phase)
        elif self.episode_in_current_phase == 1:
            self.phase_start_time = datetime.now()
            self.logger.info(f"Starting curriculum phase: {current_phase_config['name']}")
        # No change if current phase is not completed or it's the last phase and already logged completion

    def _get_fixed_straight_spawn_and_target(self) -> Tuple[Optional[carla.Transform], Optional[carla.Waypoint]]:
        """Determines spawn and target for 'fixed_straight' config."""
        if len(self._spawn_points) <= self.phase0_spawn_point_idx:
            self.logger.error(f"Phase0: Not enough spawn points ({len(self._spawn_points)}) for index {self.phase0_spawn_point_idx}. Cannot use fixed_straight.")
            return None, None # Indicate failure to allow fallback

        start_transform = self._spawn_points[self.phase0_spawn_point_idx]
        start_waypoint = self.map.get_waypoint(start_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)

        if not start_waypoint:
            self.logger.error(f"Phase0: Could not get valid start waypoint for spawn index {self.phase0_spawn_point_idx}. Cannot use fixed_straight.")
            return None, None # Indicate failure

        self.logger.debug(f"Phase0: Start Waypoint: {start_waypoint.transform.location}")
        target_waypoint = None
        target_candidates = start_waypoint.next(self.phase0_target_distance_m)
        if target_candidates and len(target_candidates) > 0:
            target_waypoint = target_candidates[0]
        else:
            self.logger.warning(f"Phase0: Could not get target {self.phase0_target_distance_m}m ahead. Trying 10m.")
            target_candidates = start_waypoint.next(10.0)
            if target_candidates and len(target_candidates) > 0:
                target_waypoint = target_candidates[0]
            else:
                self.logger.warning("Phase0: Could not get target 10m ahead. Trying any next distinct waypoint.")
                next_few_wps = start_waypoint.next_until_lane_end(5.0)
                if len(next_few_wps) > 1:
                    target_waypoint = next_few_wps[-1]
                elif len(next_few_wps) == 1 and next_few_wps[0].transform.location.distance(start_waypoint.transform.location) > 0.5:
                    target_waypoint = next_few_wps[0]
                else:
                    self.logger.error(f"Phase0: Cannot find a suitable distinct forward target from {start_waypoint.transform.location}. Spawn point {self.phase0_spawn_point_idx} might be problematic.")
                    return start_transform, None # Return start but no target to indicate issue
        
        if target_waypoint:
             self.logger.debug(f"Phase0: Target Set: {target_waypoint.transform.location}")
        return start_transform, target_waypoint

    def _get_random_spawn_and_target(self) -> Tuple[Optional[carla.Transform], Optional[carla.Waypoint]]:
        """Determines spawn and target for 'random' config."""
        if not self._spawn_points: 
            self.logger.error("Random Spawn: No spawn points available!")
            return None, None

        start_transform = random.choice(self._spawn_points)
        start_waypoint = self.map.get_waypoint(start_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        if not start_waypoint: # If chosen random spawn is not on a drivable lane, try to find another one.
            self.logger.warning(f"Random Spawn: Initial random choice {start_transform.location} not on drivable lane. Retrying...")
            for _ in range(5): # Try a few times
                start_transform = random.choice(self._spawn_points)
                start_waypoint = self.map.get_waypoint(start_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                if start_waypoint: break
            if not start_waypoint:
                self.logger.error("Random Spawn: Could not find a valid start waypoint on drivable lane after retries.")
                # Fallback: use the initially chosen transform and get any waypoint, even non-drivable
                start_waypoint = self.map.get_waypoint(start_transform.location)
                if not start_waypoint:
                    self.logger.critical("Random Spawn: CRITICAL - Cannot even get any waypoint for chosen spawn.")
                    return None, None # Major issue

        target_waypoint = None
        if len(self._spawn_points) < 2:
            target_waypoint = start_waypoint
            self.logger.warning("Random: Target set to start waypoint (less than 2 spawn points).")
        else:
            possible_target_transforms = [sp for sp in self._spawn_points if sp.location.distance(start_transform.location) > 30.0] 
            if not possible_target_transforms:
                possible_target_transforms = [sp for sp in self._spawn_points if sp != start_transform] 
            
            if not possible_target_transforms: 
                target_waypoint = start_waypoint 
                self.logger.warning("Random: No distinct target transforms found. Target set to start waypoint.")
            else: 
                chosen_target_transform = random.choice(possible_target_transforms)
                target_waypoint = self.map.get_waypoint(chosen_target_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                if not target_waypoint: 
                    self.logger.warning(f"Random: Chosen target transform {chosen_target_transform.location} not on drivable lane. Defaulting to 50m from start.")
                    fallback_targets = start_waypoint.next(50.0)
                    if fallback_targets: target_waypoint = fallback_targets[0]
                    else: target_waypoint = start_waypoint # Ultimate fallback for target
        
        return start_transform, target_waypoint

    def _determine_spawn_and_target(self, spawn_config_type: str) -> Tuple[Optional[carla.Transform], Optional[carla.Waypoint]]:
        """Determines and returns the spawn transform and target waypoint based on config."""
        start_spawn_transform = None
        target_wp = None

        # Handle known spawn config types
        if spawn_config_type == "fixed_straight":
            start_spawn_transform, target_wp = self._get_fixed_straight_spawn_and_target()
            if start_spawn_transform is None or target_wp is None:
                self.logger.warning("fixed_straight spawn/target failed. Falling back to random.")
                spawn_config_type = "random"
        elif spawn_config_type == "fixed_simple_turns":
            start_spawn_transform, target_wp = self._get_fixed_simple_turns_spawn_and_target()
            if start_spawn_transform is None or target_wp is None:
                self.logger.warning("fixed_simple_turns spawn/target failed. Falling back to random.")
                spawn_config_type = "random"
        
        # Map any random_* spawn types to base 'random'
        if spawn_config_type.startswith("random"):
            spawn_config_type = "random"
        
        if spawn_config_type == "random":
            start_spawn_transform, target_wp = self._get_random_spawn_and_target()
        elif start_spawn_transform is None or target_wp is None:
            self.logger.warning(f"Unknown spawn_config_type '{spawn_config_type}'. Falling back to random spawn.")
            start_spawn_transform, target_wp = self._get_random_spawn_and_target()

        return start_spawn_transform, target_wp

    def _synchronize_initial_observation(self) -> OrderedDict:
        """Synchronizes and retrieves the initial observation after a reset.
        Ticks the world until essential sensor data is available or retries are exhausted.
        Returns:
            An OrderedDict representing the observation, or a zeroed observation on failure.
        """
        observation = None
        agent_rgb_ready = False
        # Pygame camera is considered ready if display is not enabled.
        pygame_cam_ready = not self.enable_pygame_display 
        max_initial_ticks = 20 # Number of attempts to get initial sensor data

        for i in range(max_initial_ticks):
            observation = self._get_observation() # This populates self.latest_sensor_data via callbacks
            
            # Check if agent's primary camera data is available
            agent_rgb_raw = self.latest_sensor_data.get('rgb_camera')
            # Observation from _get_observation() already contains processed data. 
            # So, we check if the processed observation for rgb_camera is valid.
            agent_rgb_processed = observation.get('rgb_camera')
            agent_rgb_ready = agent_rgb_processed is not None and np.any(agent_rgb_processed) # Ensure not all zeros if that's an issue
            # A simpler check might be just `agent_rgb_raw is not None` if raw data arrival is the key sync point.
            # For now, let's assume processed observation being non-empty/non-zero implies readiness.
            if agent_rgb_raw is None: # More direct check on raw data arrival
                agent_rgb_ready = False
            else:
                agent_rgb_ready = True # If raw data arrived, we assume processing will make it ready

            if self.enable_pygame_display:
                spectator_cam_data = self.latest_sensor_data.get('spectator_camera')
                pygame_cam_ready = spectator_cam_data is not None
            
            if agent_rgb_ready and pygame_cam_ready: 
                self.logger.debug(f"Initial observation synchronized after {i+1} ticks.")
                break
            
            if self.world: 
                self.world.tick()
                # Apply a slightly longer sleep for later ticks if still waiting
            sleep_time = self.timestep if i < max_initial_ticks / 2 else self.timestep * 1.5
            time.sleep(sleep_time / max(1.0, self.time_scale))
        else: # Loop finished without break (max_initial_ticks reached)
            self.logger.error(f"Failed to get initial observation/display data after {max_initial_ticks} ticks.")
            self.logger.debug(f"Sync status: agent_rgb_ready={agent_rgb_ready}, pygame_cam_ready={pygame_cam_ready}")
            if not agent_rgb_ready and (observation is None or observation.get('rgb_camera') is None):
                self.logger.warning("Agent RGB camera data missing. Returning zeroed observation.")
                observation = self._get_zeroed_observation()
            elif observation is None: # Should be caught by the above, but as a safeguard
                self.logger.warning("Full observation is None. Returning zeroed observation.")
                observation = self._get_zeroed_observation()
        
        return observation

    def _get_scaled_sensor_tick(self):
        """Returns the sensor tick interval adjusted for time scale."""
        if self.time_scale > 0:
            return str(self.timestep * self.time_scale)
        return str(self.timestep)  # Fallback to original timestep

    def _get_fixed_simple_turns_spawn_and_target(self) -> Tuple[Optional[carla.Transform], Optional[carla.Waypoint]]:
        """Finds a spawn point that results in a simple left/right turn within ~20-30 m.
        Strategy: iterate over spawn points, pick one whose heading differs by at least
        15° after 20 m of roadway, indicating a curve/turn. The target is the waypoint
        located 30 m ahead (or lane end fallback)."""
        if not self._spawn_points:
            self.logger.error("SimpleTurns: No spawn points available!")
            return None, None

        for sp in self._spawn_points:
            wp_start = self.map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not wp_start:
                continue
            next_wps = wp_start.next(20.0)  # ~20 m ahead
            if not next_wps:
                continue
            wp_next = next_wps[0]
            yaw_start = wp_start.transform.rotation.yaw % 360.0
            yaw_next = wp_next.transform.rotation.yaw % 360.0
            yaw_diff = abs((yaw_next - yaw_start + 180.0) % 360.0 - 180.0)  # shortest signed diff
            if yaw_diff >= 15.0:  # qualifies as a turn
                # Determine target further ahead (~30 m from start) or lane end fallback
                target_candidates = wp_start.next(30.0)
                target_wp = target_candidates[0] if target_candidates else wp_next
                self.logger.debug(f"SimpleTurns: Selected spawn {sp.location} with yaw diff {yaw_diff:.1f}°")
                return sp, target_wp

        # Fallback if no turning spawn found
        self.logger.warning("SimpleTurns: Could not find a suitable turning spawn. Falling back to random spawn.")
        return self._get_random_spawn_and_target()

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