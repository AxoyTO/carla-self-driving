import carla
import random
import logging
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, timedelta

import config # For CARLA_DEFAULT_CURRICULUM_PHASES

class CurriculumManager:
    """Manages the training curriculum, phase transitions, and spawn/target selection."""

    def __init__(self, world: carla.World, map_obj: carla.Map, 
                 initial_spawn_points: List[carla.Transform],
                 phases: Optional[List[Dict[str, Any]]] = None, 
                 logger: Optional[logging.Logger] = None):
        """Initialize the CurriculumManager.
        Args:
            world: CARLA world instance (used for random location from navigation).
            map_obj: CARLA map object (for getting waypoints).
            initial_spawn_points: A list of available spawn points from the map.
            phases: A list of curriculum phase configurations. Uses default if None.
            logger: Optional logger instance.
        """
        self.world = world
        self.map = map_obj
        self.all_spawn_points = initial_spawn_points if initial_spawn_points else []
        self.phases = phases if phases is not None else config.CARLA_DEFAULT_CURRICULUM_PHASES
        self.logger = logger if logger else logging.getLogger(__name__ + ".CurriculumManager")

        self.current_phase_idx: int = 0
        self.episode_in_current_phase: int = 0
        self.total_episodes_tracked: int = 0 # Total episodes run through this manager instance

        # Timing for phases
        self.phase_start_time: Optional[datetime] = None
        self.total_phase_time_spent: timedelta = timedelta(0)

        # Specific settings often used by fixed spawn configurations
        self.phase0_spawn_point_idx = 41  # Default, can be overridden by phase config
        self.phase0_target_distance_m = 50.0 # Default

        if not self.all_spawn_points:
            self.logger.warning("CurriculumManager initialized with no spawn points!")
        if not self.phases:
            self.logger.error("CurriculumManager initialized with no phases! This should not happen.")
            self.phases = [{ "name": "FallbackPhase", "episodes": 1, "spawn_config": "random"}] # Minimal fallback

    def advance_phase(self):
        """Checks if the current phase is completed and advances to the next if so.
           Logs information about phase completion and start of new phases.
        """
        self.total_episodes_tracked += 1
        self.episode_in_current_phase += 1

        current_phase_cfg = self.get_current_phase_config()
        if current_phase_cfg is None: # Should not happen if phases list is valid
            self.logger.error("Cannot advance phase, current phase config is None.")
            return

        if self.episode_in_current_phase == 1: # First episode of any phase (initial or new)
            self.phase_start_time = datetime.now()
            self.logger.info(f"Starting/Continuing Curriculum Phase: '{current_phase_cfg['name']}' (Ep {self.episode_in_current_phase}/{current_phase_cfg['episodes']})")

        if self.episode_in_current_phase > current_phase_cfg["episodes"]:
            if self.phase_start_time:
                phase_duration = datetime.now() - self.phase_start_time
                self.total_phase_time_spent += phase_duration
                self.logger.info(f"Curriculum Phase '{current_phase_cfg['name']}' completed in {phase_duration.total_seconds():.2f}s ({self.episode_in_current_phase -1} episodes).")
            
            if self.current_phase_idx < len(self.phases) - 1:
                self.current_phase_idx += 1
                self.episode_in_current_phase = 1 # Reset for new phase
                new_phase_cfg = self.get_current_phase_config()
                self.phase_start_time = datetime.now() # Reset start time for the new phase
                self.logger.info(f"Advanced to Curriculum Phase: '{new_phase_cfg['name']}' (Ep {self.episode_in_current_phase}/{new_phase_cfg['episodes']})")
            elif self.episode_in_current_phase == current_phase_cfg["episodes"] + 1: # Log once after last phase completion
                self.logger.info(f"Entire curriculum completed. Continuing with settings from last phase: '{current_phase_cfg['name']}'.")
                # Keep current_phase_idx at max, episode_in_current_phase will continue to increment beyond episodes count

    def get_current_phase_config(self) -> Optional[Dict[str, Any]]:
        if 0 <= self.current_phase_idx < len(self.phases):
            return self.phases[self.current_phase_idx]
        self.logger.error(f"Current phase index {self.current_phase_idx} is out of bounds for phases list (len {len(self.phases)}).")
        return None # Or return the last phase config as a fallback

    def get_current_phase_details(self) -> Tuple[str, int, int]:
        """Returns details of the current phase for logging or display."""
        cfg = self.get_current_phase_config()
        if cfg:
            return cfg["name"], self.episode_in_current_phase, cfg["episodes"]
        return "UnknownPhase", 0, 0

    def determine_spawn_and_target(self) -> Tuple[Optional[carla.Transform], Optional[carla.Waypoint]]:
        current_phase_cfg = self.get_current_phase_config()
        if not current_phase_cfg:
            self.logger.error("Cannot determine spawn/target: No current phase config.")
            return self._get_fallback_spawn_and_target()

        spawn_config_type = current_phase_cfg.get("spawn_config", "random")
        self.logger.debug(f"Determining spawn/target for config type: {spawn_config_type}")

        start_transform: Optional[carla.Transform] = None
        target_wp: Optional[carla.Waypoint] = None

        if spawn_config_type == "fixed_straight":
            # Allow phase config to override defaults for fixed_straight
            idx = current_phase_cfg.get("phase0_spawn_point_idx", self.phase0_spawn_point_idx)
            dist = current_phase_cfg.get("phase0_target_distance_m", self.phase0_target_distance_m)
            start_transform, target_wp = self._get_fixed_straight_spawn_and_target(idx, dist)
        elif spawn_config_type == "fixed_simple_turns":
            start_transform, target_wp = self._get_fixed_simple_turns_spawn_and_target()
        elif spawn_config_type.startswith("random"):
            start_transform, target_wp = self._get_random_spawn_and_target(spawn_config_type)
        else:
            self.logger.warning(f"Unknown spawn_config_type '{spawn_config_type}'. Falling back to random.")
            start_transform, target_wp = self._get_random_spawn_and_target("random")

        if start_transform is None or target_wp is None:
            self.logger.warning(f"Primary spawn/target determination failed for '{spawn_config_type}'. Using fallback.")
            return self._get_fallback_spawn_and_target()
            
        return start_transform, target_wp

    def _get_fallback_spawn_and_target(self) -> Tuple[Optional[carla.Transform], Optional[carla.Waypoint]]:
        self.logger.warning("Using fallback spawn and target mechanism.")
        if not self.all_spawn_points:
            self.logger.error("CRITICAL FALLBACK: No spawn points available at all!")
            # Attempt to get any location from navigation as a last resort for spawn
            loc = self.world.get_random_location_from_navigation()
            if not loc: raise RuntimeError("Ultimate fallback failed: Cannot get any location from navigation.")
            # This is not ideal as it doesn't guarantee a drivable spawn for a vehicle
            fallback_spawn_transform = carla.Transform(loc, carla.Rotation())
            fallback_target_wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not fallback_target_wp:
                 fallback_target_wp = self.map.get_waypoint(self.world.get_random_location_from_navigation()) # Try again for target
            return fallback_spawn_transform, fallback_target_wp
        
        start_transform = random.choice(self.all_spawn_points)
        target_wp = self.map.get_waypoint(start_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        # Try to find a different target if possible
        if len(self.all_spawn_points) > 1:
            for _ in range(5):
                potential_target_transform = random.choice(self.all_spawn_points)
                if potential_target_transform.location.distance(start_transform.location) > 10.0: # Min distance
                    target_wp = self.map.get_waypoint(potential_target_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if target_wp: break
        if not target_wp: # If still no target, use a point 30m ahead of spawn
            start_wp = self.map.get_waypoint(start_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if start_wp:
                next_wps = start_wp.next(30.0)
                if next_wps: target_wp = next_wps[0]
        if not target_wp: # Absolute last resort for target
            target_wp = self.map.get_waypoint(self.world.get_random_location_from_navigation())
        
        self.logger.debug(f"Fallback selected: Spawn {start_transform.location}, Target {target_wp.transform.location if target_wp else 'None'}")
        return start_transform, target_wp

    def _get_fixed_straight_spawn_and_target(self, spawn_idx: int, target_dist: float) -> Tuple[Optional[carla.Transform], Optional[carla.Waypoint]]:
        if not self.all_spawn_points or len(self.all_spawn_points) <= spawn_idx:
            self.logger.error(f"FixedStraight: Not enough spawn points ({len(self.all_spawn_points)}) for index {spawn_idx}.")
            return None, None

        start_transform = self.all_spawn_points[spawn_idx]
        start_waypoint = self.map.get_waypoint(start_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not start_waypoint:
            self.logger.error(f"FixedStraight: Could not get valid start waypoint for spawn index {spawn_idx}.")
            return None, None

        target_wp = None
        target_candidates = start_waypoint.next(target_dist)
        if target_candidates: target_wp = target_candidates[0]
        else:
            self.logger.warning(f"FixedStraight: Could not get target {target_dist}m ahead. Trying shorter distances.")
            for dist_m in [max(10.0, target_dist/2), 10.0, 5.0]:
                target_candidates = start_waypoint.next(dist_m)
                if target_candidates: target_wp = target_candidates[0]; break
            if not target_wp:
                self.logger.error(f"FixedStraight: Cannot find any suitable forward target from {start_waypoint.transform.location}.")
                return start_transform, None 
        return start_transform, target_wp

    def _get_fixed_simple_turns_spawn_and_target(self) -> Tuple[Optional[carla.Transform], Optional[carla.Waypoint]]:
        if not self.all_spawn_points: return None, None
        shuffled_spawn_points = random.sample(self.all_spawn_points, len(self.all_spawn_points))

        for sp_transform in shuffled_spawn_points:
            wp_start = self.map.get_waypoint(sp_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if not wp_start: continue
            
            # Check for a turn within ~20-30m
            wp_at_20m_list = wp_start.next(20.0)
            if not wp_at_20m_list: continue
            wp_at_20m = wp_at_20m_list[0]
            
            yaw_start_deg = wp_start.transform.rotation.yaw % 360
            yaw_at_20m_deg = wp_at_20m.transform.rotation.yaw % 360
            yaw_diff = abs((yaw_at_20m_deg - yaw_start_deg + 180) % 360 - 180)

            if yaw_diff >= 15.0: # Qualifies as a turn
                target_wp_list = wp_start.next(30.0) # Target further ahead
                target_wp = target_wp_list[0] if target_wp_list else wp_at_20m # Fallback target
                self.logger.debug(f"SimpleTurns: Selected spawn {sp_transform.location} (yaw diff {yaw_diff:.1f}°), target {target_wp.transform.location}")
                return sp_transform, target_wp
                
        self.logger.warning("SimpleTurns: Could not find a suitable turning spawn. Returning None, None.")
        return None, None # Fallback handled by caller

    def _get_random_spawn_and_target(self, spawn_config_type: str) -> Tuple[Optional[carla.Transform], Optional[carla.Waypoint]]:
        if not self.all_spawn_points: return None, None
        
        start_transform = random.choice(self.all_spawn_points)
        start_waypoint = self.map.get_waypoint(start_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        # Retry if initial random choice is not on a drivable lane
        for _ in range(5):
            if start_waypoint: break
            start_transform = random.choice(self.all_spawn_points)
            start_waypoint = self.map.get_waypoint(start_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        if not start_waypoint:
            self.logger.error("RandomSpawn: Could not find valid start waypoint on drivable lane after retries.")
            return None, None # Let caller handle fallback

        # Determine target waypoint
        # For simplicity here, let's pick another random spawn point as target if far enough, 
        # or a point ~50m ahead.
        # More sophisticated logic (e.g., path finding for urban_full) can be added based on spawn_config_type.
        
        possible_targets = [sp for sp in self.all_spawn_points if sp.location.distance(start_transform.location) > 30.0]
        target_waypoint = None
        if possible_targets:
            chosen_target_transform = random.choice(possible_targets)
            target_waypoint = self.map.get_waypoint(chosen_target_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        if not target_waypoint: # Fallback if no distant spawn point or it's not on road
            target_candidates = start_waypoint.next(50.0)
            if target_candidates: target_waypoint = target_candidates[0]
            else: # Ultimate fallback for target if 50m ahead is not valid (e.g., end of road)
                self.logger.warning(f"RandomSpawn: Could not find target 50m ahead of {start_waypoint.transform.location}. Using start waypoint as target.")
                target_waypoint = start_waypoint
                
        self.logger.debug(f"RandomSpawn: Start {start_transform.location}, Target {target_waypoint.transform.location}")
        return start_transform, target_waypoint

    def get_total_phase_time_spent_seconds(self) -> float:
        """Returns the total time spent in all completed phases so far."""
        return self.total_phase_time_spent.total_seconds() 