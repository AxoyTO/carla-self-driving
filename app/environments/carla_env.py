import sys
import os
import time
import random
import logging
import weakref
import math
import numpy as np
import carla
from collections import OrderedDict, Counter
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import app.config as config
import gymnasium as gym
from gymnasium import spaces

from .sensor_manager import SensorManager
from .vehicle_manager import VehicleManager
from .curriculum_manager import CurriculumManager
from .reward_calculator import RewardCalculator
from .traffic_manager import TrafficManager
from utils.pygame_visualizer import PygameVisualizer
from utils.open3d_visualizer import Open3DLidarVisualizer
from utils.data_logger import DataLogger
from utils import formatting_utils
from utils import action_utils

# For Global Path Planning
from agents.navigation.global_route_planner import GlobalRoutePlanner
# GlobalRoutePlannerDAO is not needed for direct instantiation with GRP for CARLA 0.9.15 GRP
# from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO 

class CarlaEnv(BaseEnv):
    def __init__(self, host='localhost', port=2000, town='Town03', timestep=0.05, time_scale=1.0, \
                 image_size=(config.CARLA_DEFAULT_IMAGE_WIDTH, config.CARLA_DEFAULT_IMAGE_HEIGHT), 
                 fov=90, discrete_actions=True, 
                 log_level=logging.INFO,
                 lidar_params=None, radar_params=None,
                 enable_pygame_display=False, 
                 pygame_window_width=1920, pygame_window_height=1080,
                 save_sensor_data=False,      
                 sensor_save_base_path=config.SENSOR_DATA_SAVE_PATH, 
                 sensor_save_interval=100, 
                 curriculum_phases: Optional[List[Dict[str, Any]]] = None,
                 run_name_prefix: Optional[str] = "carla_run",
                 start_from_phase: Optional[int] = None
               ):
        super().__init__()
        self.logger = logging.getLogger(f"CarlaEnv.{town}")
        self.logger.setLevel(log_level)

        self.host = host; self.port = port; self.town = town
        self.timestep = timestep; self.time_scale = time_scale
        self.image_width, self.image_height = image_size; self.fov = fov
        self.discrete_actions = discrete_actions
        self.num_actions = config.NUM_DISCRETE_ACTIONS if discrete_actions else 0
        self.enable_pygame_display = enable_pygame_display
        self.pygame_window_width = pygame_window_width; self.pygame_window_height = pygame_window_height

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.map: Optional[carla.Map] = None
        
        self.target_waypoint: Optional[carla.Waypoint] = None 
        self.previous_location: Optional[carla.Location] = None
        self._spawn_points: List[carla.Transform] = []

        self.episode_start_time: Optional[datetime] = None
        self.total_episode_time = timedelta(0)
        self.episode_count = 0
        
        self.latest_sensor_data: Dict[str, Any] = {}
        self._initialize_latest_sensor_data_keys()

        # Performance optimizations use ThreadPoolExecutor instead of async event loops
        # for better compatibility with CARLA sensor callbacks

        self.collision_info = {'count': 0, 'last_intensity': 0.0, 'last_other_actor_id': "N/A", 'last_other_actor_type': "N/A"}
        self.relevant_traffic_light_state_debug = None
        self.episode_count_debug, self.step_count_debug = 0, 0
        self.current_action_debug, self.step_reward_debug = "N/A (Reset)", 0.0
        self.episode_score_debug, self.forward_speed_debug, self.dist_to_goal_debug = 0.0, 0.0, float('inf')
        self.collision_flag_debug, self.proximity_penalty_flag_debug, self.on_sidewalk_debug_flag = False, False, False
        self.last_termination_reason_debug = "N/A"
        self.target_speed_kmh = config.REWARD_CALC_TARGET_SPEED_KMH_DEFAULT
        self.current_action_for_reward = None

        self.save_sensor_data_enabled = save_sensor_data
        self.sensor_save_interval = sensor_save_interval

        self.visualizer: Optional[PygameVisualizer] = None
        if self.enable_pygame_display:
            self.visualizer = PygameVisualizer(
                self.pygame_window_width, 
                self.pygame_window_height, 
                f"CARLA RL Agent View", 
                weakref.ref(self),
                disable_sensor_views_flag=config.DISABLE_SENSOR_VIEWS
            )
        self.o3d_lidar_vis: Optional[Open3DLidarVisualizer] = None
        if self.enable_pygame_display:
            try: self.o3d_lidar_vis = Open3DLidarVisualizer()
            except Exception as e: self.logger.error(f"Failed to init Open3D Lidar Visualizer: {e}")

        self.radar_to_vehicle_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.lidar_config = self._create_lidar_config(lidar_params)
        self.radar_config = self._create_radar_config(radar_params)
        self.pygame_display_camera_transform = carla.Transform(carla.Location(x=-5.5, y=0, z=3.5), carla.Rotation(pitch=-15))

        if self.discrete_actions: self._action_space = spaces.Discrete(self.num_actions)
        else: self._action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self._observation_space = self._define_observation_space()

        self.connect() # Establishes self.client, self.world, self.map
        
        # Global Route Planner (GRP) setup
        self.grp: Optional[GlobalRoutePlanner] = None
        if self.map:
            # For CARLA 0.9.15, GRP takes the map and sampling_resolution directly
            self.grp = GlobalRoutePlanner(self.map, sampling_resolution=2.0) 
            # The setup() method for GRP in 0.9.15 is usually implicitly handled by its constructor 
            # or not explicitly called after instantiation in common examples.
            # If it has a public setup() and it's necessary, it would be called here.
            # However, typical usage for 0.9.11+ GRP is direct instantiation.
            # Let's assume direct instantiation is sufficient as per common usage.
            # If a distinct self.grp.setup() call is needed and available, it can be added.
            self.logger.info("GlobalRoutePlanner initialized.")
        else:
            self.logger.error("Map not available after connect(), GlobalRoutePlanner NOT initialized.")
        
        # Initialize managers that depend on a valid world connection
        if self.world and self.map: 
            self._spawn_points = self.map.get_spawn_points() 
            if not self._spawn_points: 
                self.logger.warning("Map has no spawn points! CurriculumManager might face issues.")
                self._spawn_points = [] 
            
            self.vehicle_manager = VehicleManager(self.world, self.logger)
            self.curriculum_manager = CurriculumManager(
                world=self.world, 
                map_obj=self.map, 
                initial_spawn_points=self._spawn_points, 
                phases=curriculum_phases, 
                logger=self.logger,
                start_from_phase=start_from_phase
            )
            if self.client: 
                try: 
                    self.traffic_manager = TrafficManager(self.client, self.world, weakref.ref(self), log_level, self.time_scale)
                except Exception as e: 
                    self.logger.error(f"Failed to init TrafficManager: {e}")
                    self.traffic_manager = None 
            else:
                self.logger.error("Client not available for TrafficManager initialization after successful connect.")
                self.traffic_manager = None
        else:
            self.logger.critical("World or Map not initialized after connect. Cannot create core managers.")
            self.vehicle_manager = None 
            self.curriculum_manager = None
            self.traffic_manager = None
            self.grp = None # Ensure GRP is also None if map wasn't available
            raise RuntimeError("Failed to initialize CARLA world/map, critical for manager setup.")
            
        self.sensor_manager: Optional[SensorManager] = None
        
        self.data_logger: Optional[DataLogger] = None
        if self.save_sensor_data_enabled:
            run_name = f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self.data_logger = DataLogger(base_save_path=sensor_save_base_path, run_name=run_name, logger=self.logger)
            active_sensors_to_log = [key for key in self.latest_sensor_data.keys() if self._observation_space and key in self._observation_space.spaces or key == 'collision']
            if self.data_logger.current_run_save_path: 
                self.data_logger.setup_sensor_save_subdirs(active_sensors_to_log)
            else:
                self.logger.error("DataLogger run path not set, disabling sensor data saving."); self.save_sensor_data_enabled = False

        # Reward calculator might need curriculum_manager.phases, ensure curriculum_manager is valid
        if self.curriculum_manager:
            self.reward_calculator = RewardCalculator({}, self.target_speed_kmh, self.curriculum_manager.phases, weakref.ref(self))
        else:
            self.logger.error("CurriculumManager not initialized, RewardCalculator might not have correct phases.")
            self.reward_calculator = RewardCalculator({}, self.target_speed_kmh, [], weakref.ref(self)) # Fallback with empty phases

        self.global_plan: List[Tuple[carla.Waypoint, Any]] = [] 
        self.current_global_plan_segment_index = 0
        self.global_plan_ended_logged = False

    def _define_observation_space(self) -> spaces.Dict:
        obs_spaces = OrderedDict()
        # Assuming camera_handler and other handlers are still available for this setup phase
        # If not, these static methods might need to be part of SensorManager or a common util
        from .sensors import camera_handler as ch # Temp import for this method
        from .sensors import lidar_handler as lh
        from .sensors import gnss_imu_handler as gih
        from .sensors import radar_handler as rh

        obs_spaces.update(ch.get_camera_observation_spaces(self.image_width, self.image_height))
        rgb_shape = (3, self.image_height, self.image_width)
        for key in ['left_rgb_camera', 'right_rgb_camera', 'rear_rgb_camera']:
            if key not in obs_spaces: obs_spaces[key] = spaces.Box(low=0, high=255, shape=rgb_shape, dtype=np.uint8)
        obs_spaces['lidar'] = lh.get_lidar_observation_space(self.lidar_config['num_points_processed'])
        obs_spaces['semantic_lidar'] = lh.get_semantic_lidar_observation_space(self.lidar_config.get('num_points_processed', config.CARLA_PROCESSED_LIDAR_NUM_POINTS))
        obs_spaces['gnss'] = gih.get_gnss_observation_space()
        obs_spaces['imu'] = gih.get_imu_observation_space()
        obs_spaces['radar'] = rh.get_radar_observation_space(self.radar_config['max_detections_processed'])
        return spaces.Dict(obs_spaces)

    def _init_sensor_manager(self):
        """Initializes or re-initializes the SensorManager."""
        current_vehicle = self.vehicle_manager.get_vehicle() # Get vehicle from manager
        if not self.world or not current_vehicle or not current_vehicle.is_alive: # Check current_vehicle
            self.logger.error("Cannot initialize SensorManager: world or vehicle not ready.")
            return

        self.sensor_manager = SensorManager(
            world=self.world,
            vehicle=current_vehicle, # Pass vehicle from manager
            env_ref=weakref.ref(self),
            image_size=(self.image_width, self.image_height),
            fov=self.fov,
            timestep=self.timestep,
            time_scale=self.time_scale,
            enable_pygame_display=self.enable_pygame_display,
            pygame_window_size=(self.pygame_window_width, self.pygame_window_height),
            lidar_config=self.lidar_config,
            radar_config=self.radar_config,
            observation_space=self._observation_space,
            logger=self.logger
        )

    def connect(self):
        try:
            self.client = carla.Client(self.host, self.port); self.client.set_timeout(10.0)
            self.world = self.client.load_world(self.town)
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.timestep * self.time_scale if self.time_scale > 0 and self.time_scale != 1.0 else self.timestep
            self.world.apply_settings(settings)
            if self.time_scale != 1.0 and self.time_scale > 0:
                self.logger.info(f"Effective sim timestep: {settings.fixed_delta_seconds:.4f}s (target: {self.time_scale}x of {self.timestep:.4f}s)")
            self.map = self.world.get_map()
            self.logger.info(f"Connected to CARLA@{self.host}:{self.port}, loaded {self.town}")
        except Exception as e: self.logger.error(f"Error connecting/loading world: {e}", exc_info=True); raise

    def _reset_environment_state(self):
        self.step_count_debug, self.episode_score_debug, self.forward_speed_debug = 0, 0.0, 0.0
        self.previous_distance, self.dist_to_goal_debug = float('inf'), float('inf')
        self.collision_info = {'count': 0, 'last_intensity': 0.0, 'last_other_actor_id': "N/A", 'last_other_actor_type': "N/A"}
        self.relevant_traffic_light_state_debug, self.current_action_for_reward = None, None
        self.collision_flag_debug, self.proximity_penalty_flag_debug, self.on_sidewalk_debug_flag = False, False, False
        self.last_termination_reason_debug = "reset"
        # Reset sensor data dictionary for the environment
        self._initialize_latest_sensor_data_keys()

    def _reset_sensors(self):
        if self.sensor_manager:
            current_vehicle = self.vehicle_manager.get_vehicle()
            if current_vehicle and current_vehicle.is_alive:
                self.sensor_manager.vehicle = current_vehicle # Ensure SM has the latest vehicle ref
                self.sensor_manager.setup_all_sensors()
            else: self.logger.error("Cannot reset sensors: Vehicle is not valid.")
        else: self.logger.warning("SensorManager not initialized, cannot reset sensors.")

    def _get_observation(self) -> OrderedDict:
        if self.sensor_manager: return self.sensor_manager.get_observation_data()
        self.logger.warning("SensorManager not init; returning zeroed observation."); return self._get_zeroed_observation()

    def _get_zeroed_observation(self) -> OrderedDict:
        """
        Return a zeroed observation that matches the observation space structure.
        This is used when sensors are not ready or when there's an error condition.
        
        Returns:
            OrderedDict: Zeroed observation with the correct shape and dtype for each sensor
        """
        obs = OrderedDict()
        
        if self._observation_space is None:
            self.logger.error("Observation space not defined, cannot create zeroed observation")
            return obs
        
        # Create zeroed arrays for each sensor based on the observation space
        for key, space_def in self._observation_space.spaces.items():
            obs[key] = np.zeros(space_def.shape, dtype=space_def.dtype)
        
        self.logger.debug(f"Created zeroed observation with keys: {list(obs.keys())}")
        return obs

    def _apply_action(self, action: int):
        vehicle = self.vehicle_manager.get_vehicle()
        if not vehicle or not vehicle.is_alive:
            self.logger.warning("Cannot apply action: vehicle is None or not alive."); self.current_action_debug = "No Vehicle"; return
        
        final_control = carla.VehicleControl()
        action_name_for_debug = "Unknown"
        effective_action_idx = action # Store the original or remapped action for reward calc

        if self.discrete_actions:
            current_phase_cfg = self.curriculum_manager.get_current_phase_config()
            allow_reverse = current_phase_cfg.get("allow_reverse", False) if current_phase_cfg else False
            allow_steering = current_phase_cfg.get("allow_steering", True) if current_phase_cfg else True
            
            # Get control and action name from the utility function
            final_control, action_name_for_debug, effective_action_idx = action_utils.get_vehicle_control_from_discrete_action(
                action_index=action, 
                allow_reverse=allow_reverse,
                allow_steering=allow_steering
            )
            self.current_action_debug = f"{action_name_for_debug} ({action})" # Log with original action index for clarity
        
        else: # Continuous actions
            # This part remains the same if continuous actions are not using the discrete map
            final_control.throttle = float(max(0, action[0])) 
            final_control.brake = float(max(0, -action[0])) # Assuming action[0] < 0 means brake
            final_control.steer = float(action[1])
            # Continuous actions might not have a simple 'name', so debug string is direct values
            self.current_action_debug = f"T:{final_control.throttle:.2f} S:{final_control.steer:.2f} B:{final_control.brake:.2f}"
            # For continuous actions, effective_action_idx might not be applicable unless mapped.
            # For now, current_action_for_reward will store the raw continuous action tuple/array.
            # If RewardCalculator needs a discrete-like index for continuous, that needs specific mapping.
            effective_action_idx = None # Or some other placeholder for continuous
            
        vehicle.apply_control(final_control)
        # Store the effective action index (remapped if unknown) or the raw continuous action for reward calculation
        self.current_action_for_reward = effective_action_idx if self.discrete_actions else action

    def _update_debug_info(self):
        vehicle = self.vehicle_manager.get_vehicle()
        if vehicle and vehicle.is_alive: # Use local var vehicle
            velocity_vector = vehicle.get_velocity()
            vehicle_transform = vehicle.get_transform()
            vehicle_forward_vector = vehicle_transform.get_forward_vector()
            self.forward_speed_debug = np.dot(
                [velocity_vector.x, velocity_vector.y, velocity_vector.z],
                [vehicle_forward_vector.x, vehicle_forward_vector.y, vehicle_forward_vector.z]
            )
            self._update_relevant_traffic_light(vehicle) # Pass vehicle if needed by this func
        else:
            self.forward_speed_debug = 0.0
            self.relevant_traffic_light_state_debug = None

    def _update_relevant_traffic_light(self, vehicle: Optional[carla.Actor]): # Added vehicle param
        self.relevant_traffic_light_state_debug = None
        if vehicle and self.map: # Use passed vehicle
            tl_actor = vehicle.get_traffic_light()
            if tl_actor and isinstance(tl_actor, carla.TrafficLight):
                self.relevant_traffic_light_state_debug = tl_actor.get_state()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[OrderedDict, dict]:
        if seed is not None:
            random.seed(seed)
        
        self.curriculum_manager.advance_phase()
        current_phase_cfg = self.curriculum_manager.get_current_phase_config()
        phase_name, ep_in_phase, total_eps_in_phase = self.curriculum_manager.get_current_phase_details()
        self.logger.info(f"Phase '{phase_name}' (Episode: {ep_in_phase}/{total_eps_in_phase}), SpawnCfg: {current_phase_cfg.get('spawn_config', 'random')}, TargetType: {current_phase_cfg.get('target_config', 'default')}, Total Time: {self.total_episode_time.total_seconds():.2f}s")

        self.episode_start_time = datetime.now()
        self.episode_count += 1
        self.episode_count_debug = self.episode_count

        if self.traffic_manager: self.traffic_manager.destroy_npcs()
        if self.sensor_manager: self.sensor_manager.cleanup()
        self.vehicle_manager.destroy_vehicle()
        self._reset_environment_state()
        
        # Determine start_spawn_transform AND final_destination_waypoint from CurriculumManager
        start_spawn_transform, final_destination_waypoint = self.curriculum_manager.determine_spawn_and_target()
        
        if start_spawn_transform is None or final_destination_waypoint is None:
            self.logger.critical("CRITICAL FALLBACK: CurriculumManager failed to provide valid spawn or target. Defaulting.")
            if not self._spawn_points: raise RuntimeError("Map has no spawn points, cannot proceed.")
            start_spawn_transform = random.choice(self._spawn_points)
            # Fallback target: a waypoint ~50-100m ahead if possible, or random if not.
            temp_start_wp = self.map.get_waypoint(start_spawn_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if temp_start_wp and temp_start_wp.next(50.0):
                final_destination_waypoint = random.choice(temp_start_wp.next(50.0))
            else: # Absolute fallback for target
                final_destination_waypoint = self.map.get_waypoint(random.choice(self._spawn_points).location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not final_destination_waypoint: # Even more fallback
                final_destination_waypoint = self.map.get_waypoint(self.world.get_random_location_from_navigation())
            if not final_destination_waypoint: raise RuntimeError("Cannot determine a valid final_destination_waypoint even with fallbacks.")
            self.logger.warning(f"Fallback: Start: {start_spawn_transform.location}, Target: {final_destination_waypoint.transform.location}")

        # Generate global plan to the curriculum-defined final_destination_waypoint
        self.global_plan = []
        self.current_global_plan_segment_index = 0
        self.global_plan_ended_logged = False
        if self.grp and start_spawn_transform and final_destination_waypoint:
            try:
                self.global_plan = self.grp.trace_route(
                    start_spawn_transform.location, 
                    final_destination_waypoint.transform.location # Use location from the Waypoint object
                )
                if not self.global_plan:
                    self.logger.warning(f"GRP returned empty plan from {start_spawn_transform.location} to {final_destination_waypoint.transform.location}. Will use direct target.")
                else:
                    self.logger.debug(f"GRP generated plan ({len(self.global_plan)} waypoints) to curriculum target: {final_destination_waypoint.transform.location}")
            except Exception as e:
                self.logger.error(f"Error generating GRP plan: {e}. Will use direct target.", exc_info=True)
        else:
            self.logger.warning("GRP not available or start/target missing. Cannot generate global plan. Will use direct target.")

        if self.global_plan:
            # Find the first waypoint that's far enough from spawn to avoid immediate goal-reached rewards
            min_distance_to_first_target = self.reward_calculator.WAYPOINT_REACHED_THRESHOLD * 2.0  # Use 2x threshold as minimum
            self.target_waypoint = None
            
            for i, (waypoint, road_option) in enumerate(self.global_plan):
                distance_from_spawn = waypoint.transform.location.distance(start_spawn_transform.location)
                if distance_from_spawn >= min_distance_to_first_target:
                    self.target_waypoint = waypoint
                    self.current_global_plan_segment_index = i
                    self.logger.debug(f"Selected waypoint {i} as first target (distance: {distance_from_spawn:.2f}m from spawn)")
                    break
            
            # If no waypoint in the plan is far enough, use the final destination directly
            if self.target_waypoint is None:
                self.target_waypoint = final_destination_waypoint
                self.current_global_plan_segment_index = len(self.global_plan) - 1  # Point to last waypoint
                self.logger.warning(f"No waypoint in global plan far enough from spawn. Using final destination as target.")
        else:
            # If no global plan (e.g., GRP failed or for very simple curriculum targets),
            # self.target_waypoint is the final_destination_waypoint itself.
            self.target_waypoint = final_destination_waypoint

        new_vehicle = self.vehicle_manager.spawn_vehicle(start_spawn_transform)
        if not new_vehicle:
            self.logger.critical("Failed to spawn vehicle! Cannot continue.")
            return self._get_zeroed_observation(), {"error": "vehicle_spawn_failed", "termination_reason": "vehicle_spawn_failed"}
            
        self.previous_location = new_vehicle.get_location()
        # dist_to_goal_debug should be to the final curriculum-defined destination
        self.dist_to_goal_debug = self.previous_location.distance(final_destination_waypoint.transform.location)

        self._init_sensor_manager()
        self._reset_sensors()

        if self.traffic_manager and current_phase_cfg.get("traffic_config", {}).get("type", "none") != "none":
            self.traffic_manager.spawn_npcs(current_phase_cfg["traffic_config"])

        observation = self._synchronize_initial_observation()
        self._update_debug_info() 

        if self.enable_pygame_display and self.visualizer:
            self.visualizer.reset_notifications() 
            # HUD should show the current segment target (self.target_waypoint)
            self.visualizer.update_goal_waypoint_debug(self.target_waypoint)
            # Render the HUD immediately to show the true starting episode score of 0.0
            self._render_pygame()
            
        self.logger.debug(f"Reset complete. Vehicle: {new_vehicle.id}. Final Curriculum Target: {final_destination_waypoint.transform.location}. Next GRP Target: {self.target_waypoint.transform.location if self.target_waypoint else 'N/A'}")
        return observation, {}

    def step(self, action: int) -> Tuple[OrderedDict, float, bool, bool, dict]:
        vehicle = self.vehicle_manager.get_vehicle()
        if not vehicle or not vehicle.is_alive:
            self.logger.error("Step: vehicle is None/dead."); 
            return self._get_zeroed_observation(), 0.0, True, False, {"termination_reason": "no_vehicle_in_step"}

        self._apply_action(action) 
        if self.world.get_settings().synchronous_mode: self.world.tick()
        else: time.sleep(self.timestep)
        self.step_count_debug += 1

        raw_collision = self.latest_sensor_data.get('collision')
        if raw_collision: self._process_collision_event(raw_collision); self.latest_sensor_data['collision'] = None
        
        if self.save_sensor_data_enabled and self.data_logger and (self.step_count_debug % self.sensor_save_interval == 0):
            for sensor_key, data_val in self.latest_sensor_data.items():
                if data_val is not None:
                    self.data_logger.save_sensor_data(self.episode_count_debug, self.step_count_debug, sensor_key, data_val)

        current_loc = vehicle.get_location()
        observation = self._get_observation()
        
        # Update distance to the *final* destination of the global plan for HUD
        if self.global_plan:
            final_destination_location = self.global_plan[-1][0].transform.location
            self.dist_to_goal_debug = current_loc.distance(final_destination_location)
        elif self.target_waypoint: # Fallback if no global plan (should be rare)
             self.dist_to_goal_debug = current_loc.distance(self.target_waypoint.transform.location)
        else:
            self.dist_to_goal_debug = float('inf')

        # Check if current self.target_waypoint (segment target) is reached
        segment_target_reached = False
        if self.target_waypoint:
            if current_loc.distance(self.target_waypoint.transform.location) < self.reward_calculator.WAYPOINT_REACHED_THRESHOLD:
                segment_target_reached = True
                self.current_global_plan_segment_index += 1
                if self.current_global_plan_segment_index < len(self.global_plan):
                    self.target_waypoint = self.global_plan[self.current_global_plan_segment_index][0]
                    # self.logger.debug(f"Advanced to next segment in global plan. New target: {self.target_waypoint.transform.location}")
                    if self.visualizer: self.visualizer.update_goal_waypoint_debug(self.target_waypoint)
                else:
                    if not self.global_plan_ended_logged:
                        self.logger.debug("Reached end of global plan segments.")
                        self.global_plan_ended_logged = True
                    # self.target_waypoint will remain the last one, _check_done handles final goal

        self.collision_flag_debug, self.on_sidewalk_debug_flag, self.proximity_penalty_flag_debug = False, False, False
        
        current_road_option = None
        if self.global_plan and self.current_global_plan_segment_index < len(self.global_plan):
            current_road_option = self.global_plan[self.current_global_plan_segment_index][1]

        reward, self.collision_flag_debug, self.proximity_penalty_flag_debug, self.on_sidewalk_debug_flag = \
            self.reward_calculator.calculate_reward(
                vehicle=vehicle, 
                current_location=current_loc, 
                previous_location=self.previous_location, 
                collision_info=self.collision_info, 
                relevant_traffic_light_state=self.relevant_traffic_light_state_debug,
                current_action_for_reward=self.current_action_for_reward, 
                forward_speed_debug=self.forward_speed_debug,
                carla_map=self.map,
                target_waypoint=self.target_waypoint,
                segment_target_reached=segment_target_reached, 
                distance_to_final_goal=self.dist_to_goal_debug, 
                lane_invasion_event=self.latest_sensor_data.get('lane_invasion_event'),
                current_road_option=current_road_option # Pass current road option
            )

        self.latest_sensor_data['lane_invasion_event'] = None 
        self.collision_info['count'] = 0 
        
        if self.visualizer:
            if self.collision_flag_debug: self.visualizer.add_notification(f"COLLISION! Hit: {self.collision_info['last_other_actor_type']}", 4.0, (255,20,20))
            if self.on_sidewalk_debug_flag: self.visualizer.add_notification("SIDEWALK!", 4.0, (255,0,255))
            if self.proximity_penalty_flag_debug: self.visualizer.add_notification("PROXIMITY PENALTY!", 3.0, (255,165,0))

        self.step_reward_debug = reward
        self.episode_score_debug += reward
        self.previous_location = current_loc
        
        # _check_done will now use the final destination of the global plan
        terminated, term_info = self._check_done(current_loc, final_destination_location if self.global_plan else None)
        self.last_termination_reason_debug = term_info.get("termination_reason", "terminated") if terminated else "Running"

        # Apply distance-based penalty if terminated due to max steps reached
        if terminated and term_info.get("termination_reason") == "max_steps_reached":
            distance_penalty = self.reward_calculator.calculate_max_steps_distance_penalty(
                current_location=current_loc,
                target_waypoint=self.target_waypoint,
                distance_to_final_goal=self.dist_to_goal_debug
            )
            
            if distance_penalty != 0.0:  # Only apply and log if penalty is enabled and non-zero
                reward += distance_penalty
                self.step_reward_debug += distance_penalty
                self.episode_score_debug += distance_penalty
                self.logger.debug(f"Applied max steps distance penalty: {distance_penalty:.2f} (distance to goal: {self.dist_to_goal_debug:.2f}m)")
                
                if self.visualizer:
                    self.visualizer.add_notification(f"MAX STEPS! Distance penalty: {distance_penalty:.1f}", 5.0, (255, 100, 0))

        if terminated and self.episode_start_time:
            duration = (datetime.now() - self.episode_start_time).total_seconds()
            self.total_episode_time += timedelta(seconds=duration)
            avg_time = (self.total_episode_time / self.episode_count).total_seconds() if self.episode_count else 0
            self.logger.info(f"Ep {self.episode_count} end: {duration:.2f}s. Reason: {self.last_termination_reason_debug}, Score: {self.episode_score_debug:.2f}, AvgTime: {avg_time:.2f}s")

        self._update_debug_info()
        if self.enable_pygame_display and self.visualizer: self._render_pygame()
        return observation, reward, terminated, False, term_info

    def _destroy_actors(self):
        self.logger.debug("CarlaEnv: Destroying actors.")
        if self.sensor_manager: self.sensor_manager.cleanup(); self.sensor_manager = None
        self.vehicle_manager.destroy_vehicle() # Use VehicleManager to destroy vehicle
        
        self._initialize_latest_sensor_data_keys()
        self.collision_info = {'count': 0, 'last_intensity': 0.0, 'last_other_actor_id': "N/A", 'last_other_actor_type': "N/A"}
        if self.world and self.world.get_settings().synchronous_mode: self.world.tick()

    def close(self):
        self._destroy_actors()
        if self.traffic_manager: self.traffic_manager.destroy_npcs(); self.traffic_manager = None
        if self.world and self.world.get_settings().synchronous_mode:
            settings = self.world.get_settings(); settings.synchronous_mode = False; settings.fixed_delta_seconds = None
            self.world.apply_settings(settings) 
            
        # Thread pools will be cleaned up automatically by the garbage collector
                
        self.world, self.client = None, None
        self.logger.info("Closed CARLA environment.")
        if self.episode_count > 0:
            avg_ep_time = (self.total_episode_time / self.episode_count).total_seconds()
            self.logger.info(f"Timing: Total Eps: {self.episode_count}, Total Ep Time: {self.total_episode_time.total_seconds():.2f}s, Avg Ep Time: {avg_ep_time:.2f}s")
            if self.curriculum_manager and self.curriculum_manager.get_total_phase_time_spent_seconds() > 0:
                self.logger.info(f"  Total Phase Time (from CM): {self.curriculum_manager.get_total_phase_time_spent_seconds():.2f}s")
        if self.visualizer: self.visualizer.close(); self.visualizer = None; self.logger.info("Pygame visualizer closed.")
        if self.o3d_lidar_vis: self.o3d_lidar_vis.close(); self.o3d_lidar_vis = None; self.logger.info("Open3D Lidar visualizer closed.")

    def toggle_o3d_lidar_visualization(self):
        """Toggles the Open3D LIDAR visualization window on/off."""
        if not self.o3d_lidar_vis:
            self.logger.warning("Open3D LIDAR visualizer not initialized.")
            return

        if not self.o3d_lidar_vis.is_active():
            if self.o3d_lidar_vis.activate():
                self.logger.info("Open3D LIDAR visualizer activated. Will update in main render loop.")
                # Initial data update upon activation can be done here or rely on the next _render_pygame call
                # For simplicity, we'll let _render_pygame handle updates if active.
            else:
                self.logger.error("Failed to activate Open3D LIDAR visualizer window.")
        else:
            self.o3d_lidar_vis.close()
            self.logger.info("Open3D LIDAR visualizer closed via toggle.")

    @property
    def action_space(self):
        if self._action_space is None: raise NotImplementedError("Action space not defined.")
        return self._action_space

    @property
    def observation_space(self):
        if self._observation_space is None: raise NotImplementedError("Observation space not defined.")
        return self._observation_space

    def get_sensor_summary(self) -> OrderedDict:
        summary = OrderedDict()
        if not self.sensor_manager or not self.sensor_manager.sensor_list:
            summary["Sensor Info"] = "(No sensors or SensorManager not active)"; return summary
        sensor_counts = Counter()
        agent_sensor_count = sum(1 for se in self.sensor_manager.sensor_list if se.get('purpose') == 'agent' and se.get('actor') and hasattr(se['actor'], 'type_id'))
        summary[f"Total Agent Sensors"] = f"{agent_sensor_count}"
        for se in self.sensor_manager.sensor_list:
            if se.get('purpose') == 'agent' and se.get('actor') and hasattr(se['actor'], 'type_id'):
                sensor_counts[se['actor'].type_id] += 1
        for type_id, count in sorted(sensor_counts.items()):
            parts = type_id.split('.'); name = " ".join(p.capitalize() for p in parts[1:]) if len(parts) > 1 else type_id
            summary[name] = count
        return summary

    def _check_done(self, current_location: carla.Location, final_destination_loc: Optional[carla.Location] = None) -> Tuple[bool, dict]:
        info = {"termination_reason": "running"}
        vehicle = self.vehicle_manager.get_vehicle()
        
        if not vehicle or not vehicle.is_alive: 
            info["termination_reason"] = "vehicle_destroyed"
            return True, info
        
        if self.reward_calculator.last_on_sidewalk_flag:
            info["termination_reason"] = "on_sidewalk"
            
            # Add detailed sidewalk detection information
            if hasattr(self.reward_calculator, 'last_sidewalk_detection_details'):
                details = self.reward_calculator.last_sidewalk_detection_details
                
                # Create comprehensive termination message with phase awareness
                violation_summary = f"Sidewalk violation in {details.get('phase_type', 'unknown')} phase"
                detection_method = details.get('method', 'unknown')
                detection_reason = details.get('detection_reason', 'unknown')
                
                termination_details = [
                    f"Detection: {detection_method}",
                    f"Reason: {detection_reason}",
                    f"Phase: {details.get('phase_type', 'unknown')}",
                    f"Distance to road: {details.get('distance_to_road', 0):.2f}m",
                    f"Height difference: {details.get('height_difference', 0):.2f}m"
                ]
                
                # Add lane change analysis if available
                lane_change_analysis = details.get('lane_change_analysis', '')
                if lane_change_analysis:
                    termination_details.append(f"Lane change analysis: {lane_change_analysis}")
                
                # Add threshold information for context
                thresholds = details.get('thresholds_used', {})
                if thresholds:
                    threshold_info = []
                    for key, value in thresholds.items():
                        if isinstance(value, (int, float)):
                            threshold_info.append(f"{key}: {value}")
                        elif isinstance(value, bool):
                            threshold_info.append(f"{key}: {value}")
                    if threshold_info:
                        termination_details.append(f"Thresholds: {', '.join(threshold_info)}")
                
                # Log with color coding based on phase type
                phase_type = details.get('phase_type', 'unknown')
                if phase_type == 'straight':
                    self.logger.info(f"SIDEWALK TERMINATION (STRICT): {violation_summary}")
                elif phase_type == 'steering':
                    self.logger.info(f"SIDEWALK TERMINATION (PERMISSIVE): {violation_summary}")
                else:
                    self.logger.info(f"SIDEWALK TERMINATION: {violation_summary}")
                
                # Log detailed breakdown
                for detail in termination_details:
                    self.logger.info(f"   - {detail}")
                    
                # Additional context for steering phases
                if phase_type == 'steering' and details.get('is_lane_change', False):
                    self.logger.info(f"   NOTE: Detected during lane change in steering phase - confirmed curb violation")
                elif phase_type == 'straight' and details.get('is_lane_change', False):
                    self.logger.info(f"   NOTE: Lane change detected in straight-driving phase")
            
            return True, info
            
        if self.reward_calculator.last_collision_flag:
            info["termination_reason"] = "collision"
            return True, info

        # Use final_destination_loc if provided (from global plan), otherwise use self.target_waypoint (old behavior)
        target_loc_for_done_check = final_destination_loc
        if target_loc_for_done_check is None and self.target_waypoint: # Fallback if no global plan final dest
            target_loc_for_done_check = self.target_waypoint.transform.location

        if target_loc_for_done_check and current_location:
            dist_to_target = current_location.distance(target_loc_for_done_check)
            goal_thresh = self.reward_calculator.WAYPOINT_REACHED_THRESHOLD 
            
            if dist_to_target < goal_thresh:
                # If we are checking against the final destination of a plan, this means mission accomplished
                # If target_loc_for_done_check was just a segment target (no global plan), it's also goal reached for that segment.
                current_phase_cfg = self.curriculum_manager.get_current_phase_config()
                require_stop = current_phase_cfg.get("require_stop_at_goal", False) if current_phase_cfg else False
                
                if not require_stop:
                    info["termination_reason"] = "goal_reached" # Final goal if final_destination_loc was used
                    return True, info
                elif abs(self.forward_speed_debug) <= config.STOP_AT_GOAL_SPEED_THRESHOLD:
                    info["termination_reason"] = "goal_reached_and_stopped"
                    return True, info
        
        current_phase_cfg = self.curriculum_manager.get_current_phase_config()
        max_steps_for_phase = config.MAX_STEPS_PER_EPISODE 
        if current_phase_cfg and 'max_steps' in current_phase_cfg:
            max_steps_for_phase = current_phase_cfg['max_steps']
        elif current_phase_cfg:
            self.logger.debug(f"Phase '{current_phase_cfg.get('name')}' missing 'max_steps'. Using global default: {max_steps_for_phase}.")

        if self.step_count_debug >= max_steps_for_phase: 
            info["termination_reason"] = "max_steps_reached"
            return True, info
            
        return False, info

    def _create_lidar_config(self, lidar_params_override: Optional[dict]) -> dict:
        cfg = {k: getattr(config, f"CARLA_DEFAULT_LIDAR_{k.upper()}", None) for k in ['channels', 'range', 'points_per_second', 'rotation_frequency', 'upper_fov', 'lower_fov']}
        cfg['num_points_processed'] = config.CARLA_PROCESSED_LIDAR_NUM_POINTS
        if lidar_params_override: cfg.update(lidar_params_override)
        cfg['sensor_tick'] = self._get_scaled_sensor_tick()
        return cfg

    def _create_radar_config(self, radar_params_override: Optional[dict]) -> dict:
        cfg = {k: getattr(config, f"CARLA_DEFAULT_RADAR_{k.upper()}", None) for k in ['range', 'horizontal_fov', 'vertical_fov', 'points_per_second']}
        cfg['max_detections_processed'] = config.CARLA_PROCESSED_RADAR_MAX_DETECTIONS
        if radar_params_override: cfg.update(radar_params_override)
        cfg['sensor_tick'] = self._get_scaled_sensor_tick()
        return cfg
        
    def _setup_sensor_saving_directories(self):
        if not self.save_sensor_data_enabled: return
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.current_run_sensor_save_path = os.path.join(self.sensor_save_base_path, f"run_{timestamp}")
        try:
            os.makedirs(self.current_run_sensor_save_path, exist_ok=True)
            self.logger.info(f"Sensor data save dir: {self.current_run_sensor_save_path}")
            # This logic might also move to SensorManager or a DataLogger
            log_keys = [k for k in ['rgb_camera', 'lidar', 'semantic_lidar'] if self._observation_space and k in self._observation_space.spaces]
            for key in log_keys:
                s_path = os.path.join(self.current_run_sensor_save_path, key)
                os.makedirs(s_path, exist_ok=True); self._sensor_save_dirs[key] = s_path
            if log_keys: self.logger.info(f"Created subdirs for sensor data: {log_keys}")
        except Exception as e: self.logger.error(f"Could not create sensor save dirs: {e}"); self.save_sensor_data_enabled = False

    def _get_scaled_sensor_tick(self) -> str: 
        if self.time_scale > 0:
            return str(self.timestep * self.time_scale)
        return str(self.timestep)

    def _get_pygame_debug_info(self) -> OrderedDict:
        debug_info = OrderedDict()
        vehicle = self.vehicle_manager.get_vehicle()
        
        current_view_display_name = self.visualizer.get_current_view_display_name() if self.visualizer else "N/A"
        debug_info["Current View"] = current_view_display_name
        debug_info["Server FPS"] = "N/A (Sync Mode)"
        debug_info["Episode | Step"] = f"{self.episode_count_debug} | {self.step_count_debug}"

        if vehicle and vehicle.is_alive:
            debug_info["Vehicle Model"] = formatting_utils.format_vehicle_model_name(vehicle.type_id)
            debug_info["Map"] = self.map.name.split('/')[-1] if self.map else self.town
            debug_info["Time Scale"] = f"{self.time_scale:.1f}x"
            elapsed_seconds = self.world.get_snapshot().timestamp.elapsed_seconds if self.world else 0.0
            debug_info["Simulation Time"] = formatting_utils.format_time(elapsed_seconds)
            debug_info["Speed (km/h)"] = f"{int(self.forward_speed_debug * 3.6)}"
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            debug_info["Location (X,Y,Z)"] = f"({vehicle_location.x:.2f}, {vehicle_location.y:.2f}, {vehicle_location.z:.2f})"
            debug_info["_vehicle_world_x"] = vehicle_location.x; debug_info["_vehicle_world_y"] = vehicle_location.y; debug_info["_vehicle_world_z"] = vehicle_location.z
            debug_info["Compass"] = formatting_utils.format_compass_direction(vehicle_transform.rotation.yaw)
            debug_info["_vehicle_world_yaw_rad"] = math.radians(vehicle_transform.rotation.yaw)
            debug_info["_radar_to_vehicle_transform"] = self.radar_to_vehicle_transform 
            
            # Get processed IMU data (which is a NumPy array)
            imu_numpy_array = self.latest_sensor_data.get('imu') 
            if imu_numpy_array is not None and isinstance(imu_numpy_array, np.ndarray) and imu_numpy_array.shape == (6,):
                accel_str = f"({imu_numpy_array[0]:.2f}, {imu_numpy_array[1]:.2f}, {imu_numpy_array[2]:.2f})"
                gyro_str = f"({imu_numpy_array[3]:.2f}, {imu_numpy_array[4]:.2f}, {imu_numpy_array[5]:.2f})"
            else:
                accel_str = "N/A"
                gyro_str = "N/A"
            debug_info["Acceleration"] = accel_str
            debug_info["Gyroscope"] = gyro_str
            
            control = vehicle.get_control()
            debug_info["Throttle"] = f"{control.throttle:.2f}"; debug_info["Steer"] = f"{control.steer:.2f}"; debug_info["Brake"] = f"{control.brake:.2f}"
            gear = "N"; 
            if control.reverse: gear = "R"
            elif self.forward_speed_debug > 0.1 or control.throttle > 0.1: gear = "D"
            debug_info["Gear"] = gear
        else: 
            for k_ in ["Vehicle Model", "Map", "Time Scale", "Simulation Time", "Speed (km/h)", "Location (X,Y,Z)", "Compass", "Acceleration", "Gyroscope", "Throttle", "Steer", "Brake", "Gear"]:
                debug_info[k_] = "N/A"
            debug_info["_vehicle_world_x"] = 0.0; debug_info["_vehicle_world_y"] = 0.0; debug_info["_vehicle_world_z"] = 0.0
            debug_info["_vehicle_world_yaw_rad"] = 0.0
            debug_info["_radar_to_vehicle_transform"] = carla.Transform()
        
        debug_info["Dist to Goal (m)"] = f"{self.dist_to_goal_debug:.2f}"
        debug_info["Action"] = self.current_action_debug 
        debug_info["Step Reward"] = f"{self.step_reward_debug:.2f}"
        debug_info["Episode Score"] = f"{self.episode_score_debug:.2f}"
        tl_state_str = str(self.relevant_traffic_light_state_debug) if self.relevant_traffic_light_state_debug is not None else "N/A"
        debug_info["Traffic Light"] = tl_state_str
        debug_info["Collision"] = str(self.collision_flag_debug) 
        debug_info["Proximity Penalty"] = str(self.proximity_penalty_flag_debug) 
        debug_info["On Sidewalk"] = str(self.on_sidewalk_debug_flag)
        debug_info["Term Reason"] = self.last_termination_reason_debug
        return debug_info

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Renders the environment.
        In CARLA, rendering is primarily handled by the Pygame visualizer or by saving sensor data.
        This method provides a basic interface compliant with the Gym API.

        Args:
            mode: The mode to render with. Supported: 'human', 'rgb_array'. 
                  'human' mode relies on the Pygame display being active.
                  'rgb_array' attempts to return the main RGB camera image.

        Returns:
            If mode is 'rgb_array', returns an np.ndarray of the RGB image.
            Otherwise, returns None (for 'human' mode, display is handled by PygameVisualizer).
        """
        if mode == 'human':
            # Pygame rendering is handled by self._render_pygame() called in step()
            # if self.enable_pygame_display and self.visualizer:
            #     self.visualizer.render_once() # Or however PygameVisualizer exposes a direct render call
            # For now, just log that human mode relies on the continuous Pygame updates
            if not self.enable_pygame_display:
                self.logger.debug("Render(human) called, but Pygame display is not enabled. No direct output from this method.")
            return None # Human mode rendering is event-driven via Pygame
        
        elif mode == 'rgb_array':
            # Attempt to return the current main RGB camera image
            # This assumes 'rgb_camera' key holds the processed numpy array
            if self.latest_sensor_data and 'rgb_camera' in self.latest_sensor_data:
                rgb_image = self.latest_sensor_data['rgb_camera']
                if isinstance(rgb_image, np.ndarray):
                    return rgb_image
                else: # This else corresponds to `if isinstance(rgb_image, np.ndarray):`
                    self.logger.warning("Render(rgb_array): 'rgb_camera' data is not a numpy array.")
                    return None # Or a blank image of correct dimensions
            else: # This else corresponds to `if self.latest_sensor_data and 'rgb_camera' in self.latest_sensor_data:`
                self.logger.warning("Render(rgb_array): No 'rgb_camera' data available in latest_sensor_data.")
                # Fallback: return a black image of the expected shape if observation space is defined
                if self._observation_space and 'rgb_camera' in self._observation_space.spaces:
                    shape = self._observation_space.spaces['rgb_camera'].shape
                    dtype = self._observation_space.spaces['rgb_camera'].dtype
                    return np.zeros(shape, dtype=dtype)
                return None
        else: # This else corresponds to `if mode == 'human':` and `elif mode == 'rgb_array':`
            # Call super().render() if BaseEnv implements other modes, or raise error
            # For now, just log unsupported mode.
            self.logger.warning(f"Render called with unsupported mode: {mode}")
            # return super().render(mode=mode) # If BaseEnv has a render method
            return None

    def _initialize_latest_sensor_data_keys(self):
        """Helper to initialize/re-initialize self.latest_sensor_data with None for all known sensor keys."""
        # Ensure self.latest_sensor_data is a dictionary before trying to assign to it.
        if not hasattr(self, 'latest_sensor_data') or not isinstance(self.latest_sensor_data, dict):
            self.latest_sensor_data = {}

        keys = [
            'rgb_camera', 'left_rgb_camera', 'right_rgb_camera', 'rear_rgb_camera',
            'display_rgb_camera', 'display_left_rgb_camera', 'display_right_rgb_camera', 'display_rear_rgb_camera',
            'spectator_camera', 'depth_camera', 'semantic_camera',
            'display_depth_camera', 'display_semantic_camera',
            'lidar', 'semantic_lidar',
            'lidar_raw', 'semantic_lidar_raw',
            'collision', 'gnss', 'imu', 'radar', 'lane_invasion_event'
        ]
        for key in keys:
            self.latest_sensor_data[key] = None

    def _synchronize_initial_observation(self) -> OrderedDict:
        """Synchronizes and retrieves the initial observation after a reset.
        Ticks the world until essential sensor data is available or retries are exhausted.
        Returns:
            An OrderedDict representing the observation, or a zeroed observation on failure.
        """
        observation = None
        # Check for primary agent camera (e.g., 'rgb_camera') and display camera if active
        agent_cam_key = 'rgb_camera' # Assuming this is the primary camera for the agent
        display_cam_key = 'spectator_camera' # Or the main display camera key used by PygameVisualizer

        agent_cam_ready = False
        display_cam_ready = not self.enable_pygame_display # If display not enabled, it's considered "ready"
        
        max_initial_ticks = 20  # Number of attempts to get initial sensor data
        ticks_done = 0

        for i in range(max_initial_ticks):
            ticks_done = i + 1
            # Get current observation using the new manager-based method
            current_obs_data = self._get_observation() 
            
            # Check agent camera readiness (e.g., is the numpy array populated?)
            if agent_cam_key in current_obs_data and current_obs_data[agent_cam_key] is not None:
                # Add a more specific check if needed, e.g., np.any(current_obs_data[agent_cam_key])
                agent_cam_ready = True 
            
            # Check display camera readiness if display is enabled
            if self.enable_pygame_display and display_cam_key in self.latest_sensor_data:
                if self.latest_sensor_data[display_cam_key] is not None:
                    display_cam_ready = True
            
            if agent_cam_ready and display_cam_ready:
                self.logger.debug(f"Initial observation synchronized after {ticks_done} ticks.")
                observation = current_obs_data # This is the fully populated observation
                break
            
            if self.world and self.world.get_settings().synchronous_mode:
                self.world.tick()
            else: # Should not happen if synchronous_mode is True as expected
                time.sleep(self.timestep) 
            
            # Optional: A small sleep, especially if not in perfect sync or for very fast ticks
            # time.sleep(self.timestep / max(1.0, self.time_scale) / 2) 

        if not (agent_cam_ready and display_cam_ready):
            self.logger.error(f"Failed to get initial observation/display data after {max_initial_ticks} ticks.")
            self.logger.debug(f"Sync status: agent_cam_ready={agent_cam_ready} ({agent_cam_key}), display_cam_ready={display_cam_ready} ({display_cam_key})")
            if observation is None: # If loop finished without break and obs still None
                observation = self._get_observation() # Try one last time
            if not agent_cam_ready and (observation is None or observation.get(agent_cam_key) is None):
                 self.logger.warning(f"Agent camera '{agent_cam_key}' data missing. Returning zeroed observation.")
                 return self._get_zeroed_observation()
            if observation is None: # Should be caught by the above, but as a safeguard
                self.logger.warning("Full observation is None after sync attempts. Returning zeroed observation.")
                return self._get_zeroed_observation()
        
        return observation if observation is not None else self._get_zeroed_observation()

    def _render_pygame(self):
        """Renders the current state using PygameVisualizer if enabled."""
        if not self.enable_pygame_display or not self.visualizer or not self.visualizer.is_active:
            return

        # Get the key for the current view in PygameVisualizer
        current_view_key_for_pygame = self.visualizer.get_current_view_key()
        
        # Get the raw sensor data item corresponding to this view
        # This data comes from self.latest_sensor_data, which is updated by sensor callbacks
        raw_sensor_object_to_display = self.latest_sensor_data.get(current_view_key_for_pygame)
        
        # Prepare debug information dictionary
        debug_info_dict = self._get_pygame_debug_info()

        # Special handling for LIDAR range if that view is active
        lidar_range_for_vis = 0.0
        if current_view_key_for_pygame == 'lidar':
            if hasattr(self, 'lidar_config') and self.lidar_config and 'range' in self.lidar_config:
                try:
                    lidar_range_for_vis = float(self.lidar_config['range'])
                except ValueError:
                    self.logger.warning("Could not convert lidar_config range to float.")
            else:
                self.logger.debug("LIDAR range config not found in CarlaEnv for visualizer, PygameVisualizer will use its default.")

        # Call the visualizer's main render method
        if not self.visualizer.render(raw_sensor_object_to_display, 
                                     current_view_key_for_pygame, 
                                     debug_info_dict,
                                     lidar_sensor_range_from_env=lidar_range_for_vis):
            # If visualizer.render returns False, it means pygame was closed by user
            self.enable_pygame_display = False 
            if self.visualizer:
                self.visualizer.close() 
                self.visualizer = None
            self.logger.info("Pygame display has been closed by user and disabled.")
        
        # Update Open3D LIDAR visualizer if it's active
        if self.o3d_lidar_vis and self.o3d_lidar_vis.is_active():
            # Use the raw CARLA measurement for Open3D, not the agent's processed numpy array
            raw_lidar_for_o3d = self.latest_sensor_data.get('lidar_raw') 
            # Could add a toggle or logic here to choose 'semantic_lidar_raw' if desired
            
            ego_vehicle = self.vehicle_manager.get_vehicle()
            ego_transform = ego_vehicle.get_transform() if ego_vehicle and ego_vehicle.is_alive else None
            
            if raw_lidar_for_o3d is not None and ego_transform is not None:
                if not self.o3d_lidar_vis.update_data(raw_lidar_for_o3d, ego_transform):
                    self.logger.info("Open3D LIDAR window was closed by user during update.")
                    self.o3d_lidar_vis.close()
            # else: 
                # logger.debug("O3D: Raw LIDAR data or ego_transform not available for update.")

    # Removed vehicle setup methods (_setup_vehicle_blueprint, _setup_vehicle_physics, _setup_vehicle_autopilot, _spawn_vehicle, _reset_vehicle_state)
    # These are now handled by VehicleManager
    # Removed old _destroy_actors, now uses VehicleManager and SensorManager cleanup.
    # Kept _determine_spawn_and_target and curriculum logic for now.
    # Kept _update_relevant_traffic_light, _process_collision_event (event handlers)
    # Kept _get_pygame_debug_info, _format_time, _render_pygame, _synchronize_initial_observation (UI/Sync related)
    # Kept _get_zeroed_observation (utility)
