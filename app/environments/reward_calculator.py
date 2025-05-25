import carla
import numpy as np
import math
import logging
from typing import Tuple, Dict, List, Optional, Any
import config # Import the config module

logger = logging.getLogger(__name__)

class RewardCalculator:
    def __init__(self, reward_configs: Optional[Dict] = None, 
                 target_speed_kmh: Optional[float] = None, 
                 curriculum_phases: Optional[List[Dict]] = None, 
                 carla_env_ref = None):
        """
        Initializes the RewardCalculator.
        Args:
            reward_configs (Optional[Dict]): Dictionary to override specific reward values for phases.
            target_speed_kmh (Optional[float]): Target speed for the agent. Overrides config default.
            curriculum_phases (Optional[List[Dict]]): List of curriculum phase configurations.
            carla_env_ref (weakref): Weak reference to CarlaEnv.
        """
        self.reward_configs = reward_configs if reward_configs else {}
        self.target_speed_kmh = target_speed_kmh if target_speed_kmh is not None else config.REWARD_CALC_TARGET_SPEED_KMH_DEFAULT
        self.curriculum_phases = curriculum_phases if curriculum_phases is not None else []
        self.carla_env_ref = carla_env_ref

        # --- Load Reward Constants from config.py ---
        self.PENALTY_COLLISION = config.REWARD_CALC_PENALTY_COLLISION
        self.REWARD_GOAL_REACHED = config.REWARD_CALC_REWARD_GOAL_REACHED
        self.PENALTY_PER_STEP = config.REWARD_CALC_PENALTY_PER_STEP
        self.REWARD_DISTANCE_FACTOR = config.REWARD_CALC_REWARD_DISTANCE_FACTOR
        self.WAYPOINT_REACHED_THRESHOLD = config.REWARD_CALC_WAYPOINT_REACHED_THRESHOLD
        
        self.TARGET_SPEED_REWARD_FACTOR = config.REWARD_CALC_TARGET_SPEED_REWARD_FACTOR
        self.TARGET_SPEED_STD_DEV_KMH = config.REWARD_CALC_TARGET_SPEED_STD_DEV_KMH
        self.LANE_CENTERING_REWARD_FACTOR = config.REWARD_CALC_LANE_CENTERING_REWARD_FACTOR
        self.LANE_ORIENTATION_PENALTY_FACTOR = config.REWARD_CALC_LANE_ORIENTATION_PENALTY_FACTOR
        self.PENALTY_OFFROAD = config.REWARD_CALC_PENALTY_OFFROAD
        self.MIN_FORWARD_SPEED_THRESHOLD = config.REWARD_CALC_MIN_FORWARD_SPEED_THRESHOLD
        self.PENALTY_STUCK_OR_REVERSING_BASE = config.REWARD_CALC_PENALTY_STUCK_OR_REVERSING_BASE
        
        self.PENALTY_TRAFFIC_LIGHT_RED_MOVING = config.REWARD_CALC_PENALTY_TRAFFIC_LIGHT_RED_MOVING
        self.REWARD_TRAFFIC_LIGHT_GREEN_PROCEED = config.REWARD_CALC_REWARD_TRAFFIC_LIGHT_GREEN_PROCEED
        self.REWARD_TRAFFIC_LIGHT_STOPPED_AT_RED = config.REWARD_CALC_REWARD_TRAFFIC_LIGHT_STOPPED_AT_RED
        self.VEHICLE_STOPPED_SPEED_THRESHOLD = config.REWARD_CALC_VEHICLE_STOPPED_SPEED_THRESHOLD

        self.PROXIMITY_THRESHOLD_VEHICLE = config.REWARD_CALC_PROXIMITY_THRESHOLD_VEHICLE
        self.PENALTY_PROXIMITY_VEHICLE_FRONT = config.REWARD_CALC_PENALTY_PROXIMITY_VEHICLE_FRONT

        # Load new penalty for crossing solid lane
        self.PENALTY_SOLID_LANE_CROSS = config.REWARD_CALC_PENALTY_SOLID_LANE_CROSS

        # Load new penalty for driving on sidewalk
        self.PENALTY_SIDEWALK = config.REWARD_CALC_PENALTY_SIDEWALK

        # Load steering penalty parameters
        self.PENALTY_EXCESSIVE_STEER_BASE = config.REWARD_CALC_PENALTY_EXCESSIVE_STEER_BASE
        self.STEER_THRESHOLD_STRAIGHT = config.REWARD_CALC_STEER_THRESHOLD_STRAIGHT
        self.MIN_SPEED_FOR_STEER_PENALTY_KMH = config.REWARD_CALC_MIN_SPEED_FOR_STEER_PENALTY_KMH

        # Phase 0 specific adjustments (can be overridden by curriculum config or reward_configs dict)
        # These are defaults if not specified in curriculum phase reward_configs
        self.phase0_penalty_per_step = config.REWARD_CALC_PHASE0_PENALTY_PER_STEP
        self.phase0_distance_factor_multiplier = config.REWARD_CALC_PHASE0_DISTANCE_FACTOR_MULTIPLIER
        self.phase0_goal_reward_multiplier = config.REWARD_CALC_PHASE0_GOAL_REWARD_MULTIPLIER
        self.phase0_stuck_penalty_base = config.REWARD_CALC_PHASE0_STUCK_PENALTY_BASE
        self.phase0_stuck_multiplier_stuck = config.REWARD_CALC_PHASE0_STUCK_MULTIPLIER_STUCK
        self.phase0_stuck_multiplier_reversing = config.REWARD_CALC_PHASE0_STUCK_MULTIPLIER_REVERSING
        self.phase0_offroad_penalty = config.REWARD_CALC_PHASE0_OFFROAD_PENALTY
        self.phase0_offroad_no_waypoint_multiplier = config.REWARD_CALC_PHASE0_OFFROAD_NO_WAYPOINT_MULTIPLIER

        # Load phase-specific sidewalk detection parameters
        self.sidewalk_detection_straight = config.SIDEWALK_DETECTION_STRAIGHT_PHASES
        self.sidewalk_detection_steering = config.SIDEWALK_DETECTION_STEERING_PHASES
        self.sidewalk_detection_default = config.SIDEWALK_DETECTION_DEFAULT

        # Allow reward_configs dictionary to override any of the above loaded from config
        # This provides flexibility for per-phase specific overrides passed via curriculum
        for key, value in self.reward_configs.items():
            if hasattr(self, key.upper()): # Match config naming convention (e.g. PENALTY_COLLISION)
                setattr(self, key.upper(), value)
                logger.info(f"RewardCalculator: Overriding '{key.upper()}' with {value} from reward_configs.")
            elif hasattr(self, key): # Match direct attribute name (e.g. phase0_penalty_per_step)
                setattr(self, key, value)
                logger.info(f"RewardCalculator: Overriding '{key}' with {value} from reward_configs.")

        # Initialize sidewalk detection details
        self.last_sidewalk_detection_details = {
            'detected': False,
            'method': 'none',
            'distance_to_road': 0.0,
            'height_difference': 0.0,
            'detection_reason': '',
            'thresholds_used': {},
            'is_lane_change': False
        }

    def _calculate_distance_goal_reward(self, current_location, previous_location, target_waypoint, reward_type, current_speed_mps, require_stop) -> float:
        reward = 0.0
        if not target_waypoint or not previous_location or not target_waypoint.transform:
            return reward

        dist_to_target = current_location.distance(target_waypoint.transform.location)
        prev_dist_to_target = previous_location.distance(target_waypoint.transform.location)
        dist_reduction = prev_dist_to_target - dist_to_target

        stop_ok = (not require_stop) or (current_speed_mps <= config.STOP_AT_GOAL_SPEED_THRESHOLD)

        if reward_type == "phase0":
            if dist_reduction > 0.01: # More sensitive for phase0
                distance_component = dist_reduction * self.REWARD_DISTANCE_FACTOR * self.phase0_distance_factor_multiplier
                reward += distance_component
            elif dist_reduction < -0.1: # Penalize moving away more strongly in phase0
                distance_penalty = abs(dist_reduction) * self.REWARD_DISTANCE_FACTOR * 0.5 # Standard factor for moving away
                reward -= distance_penalty
            if dist_to_target < self.WAYPOINT_REACHED_THRESHOLD and stop_ok:
                goal_reward = self.REWARD_GOAL_REACHED * self.phase0_goal_reward_multiplier
                reward += goal_reward
        elif reward_type == "standard":
            distance_component = dist_reduction * self.REWARD_DISTANCE_FACTOR
            reward += distance_component
            if dist_to_target < self.WAYPOINT_REACHED_THRESHOLD and stop_ok:
                reward += self.REWARD_GOAL_REACHED
        
        return reward

    def _calculate_speed_reward(self, vehicle, current_speed_kmh, is_reversing_action) -> float:
        reward = 0.0
        speed_diff = current_speed_kmh - self.target_speed_kmh
        speed_rew = self.TARGET_SPEED_REWARD_FACTOR * math.exp(-0.5 * (speed_diff / self.TARGET_SPEED_STD_DEV_KMH)**2)
        
        # Apply speed reward if not braking hard while slow, and not reversing (unless reversing is intended)
        if not (vehicle.get_control().brake > 0.5 and current_speed_kmh < 5) and not is_reversing_action:
            reward += speed_rew
        
        # Penalize for excessive speeding
        if current_speed_kmh > self.target_speed_kmh + 2 * self.TARGET_SPEED_STD_DEV_KMH:
            reward -= (current_speed_kmh - (self.target_speed_kmh + 2 * self.TARGET_SPEED_STD_DEV_KMH)) * 0.1
        return reward

    def _calculate_lane_keeping_rewards_penalties(self, vehicle, current_location, carla_map, lane_invasion_event) -> Tuple[float, bool]:
        reward = 0.0
        on_sidewalk_flag = False # This will be the definitive flag for sidewalk incursion for this step

        if not carla_map:
            return reward, on_sidewalk_flag

        # 1. Prioritize Lane Invasion for Curb Detection
        if lane_invasion_event:
            for marking in lane_invasion_event.crossed_lane_markings:
                if marking.type == carla.LaneMarkingType.Curb:
                    reward += self.PENALTY_SIDEWALK  # Apply the strong sidewalk penalty
                    on_sidewalk_flag = True
                    break # Curb crossing is definitive for sidewalk penalty this step
        
        # 2. If no curb was hit, check for direct sidewalk lane type
        if not on_sidewalk_flag: # Only if a curb wasn't already processed
            current_waypoint_at_location = carla_map.get_waypoint(current_location, project_to_road=False) 
            if current_waypoint_at_location and current_waypoint_at_location.lane_type == carla.LaneType.Sidewalk:
                reward += self.PENALTY_SIDEWALK # Apply penalty
                on_sidewalk_flag = True      # Set flag

        # 3. If still not flagged for sidewalk, proceed with general off-road and lane keeping
        if not on_sidewalk_flag:
            projected_driving_wp = carla_map.get_waypoint(current_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if projected_driving_wp and projected_driving_wp.transform:
                if projected_driving_wp.lane_type != carla.LaneType.Driving: 
                    reward += self.PENALTY_OFFROAD # General offroad if not on sidewalk
                else: # On a driving lane (by projection)
                    # Lane Centering
                    max_dev = projected_driving_wp.lane_width / 1.8 
                    lat_dist = current_location.distance(projected_driving_wp.transform.location) 
                    reward += self.LANE_CENTERING_REWARD_FACTOR * (1.0 - min(lat_dist / max_dev, 1.0)**2)

                    # Lane Orientation
                    v_fwd = vehicle.get_transform().get_forward_vector()
                    l_fwd = projected_driving_wp.transform.get_forward_vector()
                    v_fwd_2d = np.array([v_fwd.x, v_fwd.y])
                    l_fwd_2d = np.array([l_fwd.x, l_fwd.y])
                    norm_v = np.linalg.norm(v_fwd_2d)
                    norm_l = np.linalg.norm(l_fwd_2d)
                    if norm_v > 1e-4 and norm_l > 1e-4:
                        dot_product = np.dot(v_fwd_2d, l_fwd_2d) / (norm_v * norm_l)
                        angle_d = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
                        if angle_d > 20.0:
                            reward -= self.LANE_ORIENTATION_PENALTY_FACTOR * (angle_d / 90.0)
                    
                    # Solid Lane Crossing (only if on driving lane and not already a sidewalk event)
                    if lane_invasion_event: # Re-check for other markings if on driving lane
                        for marking in lane_invasion_event.crossed_lane_markings:
                            # Check for solid lines, but ensure not to process .Curb again here if it was missed above
                            # (though it shouldn't be if the first block for curb detection ran)
                            if marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid]:
                                reward += self.PENALTY_SOLID_LANE_CROSS
                                # Don't break, could be multiple solid lines or other non-curb events.
            else: # Cannot project to a driving lane (very off-road) AND not on sidewalk
                reward += self.PENALTY_OFFROAD 
        
        return reward, on_sidewalk_flag

    def _calculate_sidewalk_penalty_only(self, vehicle, current_location, carla_map, lane_invasion_event) -> Tuple[float, bool]:
        """
        Phase-aware sidewalk detection that distinguishes between legitimate lane changes and actual curb violations.
        Returns (sidewalk_penalty, on_sidewalk_flag)
        """
        reward = 0.0
        on_sidewalk_flag = False
        
        # Get phase-specific detection parameters
        phase_type, detection_params = self._get_current_phase_type_and_detection_params()
        
        # Store detailed detection info for debug logging
        self.last_sidewalk_detection_details = {
            'detected': False,
            'method': 'none',
            'distance_to_road': 0.0,
            'height_difference': 0.0,
            'detection_reason': '',
            'thresholds_used': detection_params.copy(),
            'is_lane_change': False,
            'phase_type': phase_type,
            'lane_change_analysis': ''
        }

        if not carla_map:
            return reward, on_sidewalk_flag

        # 1. PRIORITY: Direct curb detection via lane invasion sensor
        if lane_invasion_event:
            for marking in lane_invasion_event.crossed_lane_markings:
                if marking.type == carla.LaneMarkingType.Curb:
                    reward += self.PENALTY_SIDEWALK
                    on_sidewalk_flag = True
                    self.last_sidewalk_detection_details.update({
                        'detected': True,
                        'method': 'curb_lane_invasion',
                        'detection_reason': 'Direct curb marking crossed via lane invasion sensor',
                        'is_lane_change': False
                    })
                    logger.debug(f"Sidewalk detected: Direct curb crossing (phase: {phase_type})")
                    return reward, on_sidewalk_flag

        # 2. Check for direct sidewalk lane type
        current_waypoint_at_location = carla_map.get_waypoint(current_location, project_to_road=False) 
        if current_waypoint_at_location and current_waypoint_at_location.lane_type == carla.LaneType.Sidewalk:
            reward += self.PENALTY_SIDEWALK
            on_sidewalk_flag = True
            self.last_sidewalk_detection_details.update({
                'detected': True,
                'method': 'direct_sidewalk_lane',
                'detection_reason': 'Vehicle located directly on sidewalk lane type',
                'is_lane_change': False
            })
            logger.debug(f"Sidewalk detected: Direct sidewalk lane type (phase: {phase_type})")
            return reward, on_sidewalk_flag

        # 3. ADVANCED: Phase-aware geometric detection with lane change analysis
        projected_driving_wp = carla_map.get_waypoint(current_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        if projected_driving_wp:
            distance_to_road = current_location.distance(projected_driving_wp.transform.location)
            height_diff = current_location.z - projected_driving_wp.transform.location.z
            
            # Store basic measurements
            self.last_sidewalk_detection_details.update({
                'distance_to_road': distance_to_road,
                'height_difference': height_diff
            })
            
            # Advanced lane change analysis
            is_legitimate_lane_change, lane_change_reason = self._is_legitimate_lane_change_advanced(
                lane_invasion_event, detection_params, distance_to_road
            )
            
            self.last_sidewalk_detection_details.update({
                'is_lane_change': is_legitimate_lane_change,
                'lane_change_analysis': lane_change_reason
            })
            
            # Get phase-specific thresholds
            distance_threshold = detection_params.get('distance_threshold', 1.5)
            height_threshold = detection_params.get('height_threshold', 0.08)
            curb_edge_distance = detection_params.get('curb_edge_distance', 0.8)
            
            # If it's a legitimate lane change, use more permissive detection
            if is_legitimate_lane_change:
                logger.debug(f"Legitimate lane change detected: {lane_change_reason} (phase: {phase_type})")
                
                # For legitimate lane changes, only flag obvious curb violations
                # Use stricter thresholds to avoid false positives
                distance_threshold *= 1.5  # More permissive distance
                height_threshold *= 1.8    # More permissive height
                
                # Only check for clear curb violations during lane changes
                if (distance_to_road > distance_threshold and height_diff > height_threshold):
                    detection_reason = f"Curb violation during lane change: {distance_to_road:.2f}m > {distance_threshold:.2f}m AND {height_diff:.2f}m > {height_threshold:.2f}m"
                else:
                    # Not a curb violation - legitimate lane change
                    logger.debug(f"Lane change approved: {lane_change_reason}, distance: {distance_to_road:.2f}m, height: {height_diff:.2f}m (phase: {phase_type})")
                    return reward, on_sidewalk_flag
                    
            else:
                # Not a legitimate lane change - apply normal strict detection
                logger.debug(f"Not a legitimate lane change: {lane_change_reason}, applying strict detection (phase: {phase_type})")
                
                detection_reason = ""
                is_violation = False
                
                # Multiple detection criteria based on phase
                if distance_to_road > distance_threshold:
                    detection_reason = f"Far from driving lane: {distance_to_road:.2f}m > {distance_threshold:.2f}m"
                    is_violation = True
                elif height_diff > height_threshold:
                    detection_reason = f"Elevated above road: {height_diff:.2f}m > {height_threshold:.2f}m"
                    is_violation = True
                elif (distance_to_road > curb_edge_distance and height_diff > height_threshold * 0.5):
                    detection_reason = f"Curb edge detection: {distance_to_road:.2f}m > {curb_edge_distance:.2f}m + {height_diff:.2f}m elevation"
                    is_violation = True
                elif (distance_to_road > curb_edge_distance * 0.6 and height_diff > height_threshold * 0.8):
                    detection_reason = f"Close elevated detection: {distance_to_road:.2f}m + {height_diff:.2f}m elevation"
                    is_violation = True
                
                if not is_violation:
                    return reward, on_sidewalk_flag  # No violation detected
                
            # Check if this is a legitimate off-road area (parking, etc.)
            nearby_waypoints = self._check_nearby_waypoints(carla_map, current_location)
            is_legitimate_offroad = self._is_legitimate_offroad_area(nearby_waypoints)
            
            if is_legitimate_offroad:
                logger.debug(f"Legitimate off-road area detected, ignoring violation (phase: {phase_type})")
                return reward, on_sidewalk_flag
                
            # Apply the penalty
            reward += self.PENALTY_SIDEWALK
            on_sidewalk_flag = True
            self.last_sidewalk_detection_details.update({
                'detected': True,
                'method': 'phase_aware_geometric',
                'detection_reason': detection_reason
            })
            
            logger.debug(f"Sidewalk violation detected (phase: {phase_type}): {detection_reason}")

        return reward, on_sidewalk_flag

    def _calculate_lane_keeping_rewards_penalties_excluding_sidewalk(self, vehicle, current_location, carla_map, lane_invasion_event) -> Tuple[float, bool]:
        """
        Calculates lane keeping rewards/penalties but excludes sidewalk detection (handled separately).
        Returns (lane_reward, dummy_sidewalk_flag) where dummy_sidewalk_flag is always False since sidewalk is handled elsewhere.
        """
        reward = 0.0
        
        if not carla_map:
            return reward, False

        # Skip sidewalk detection here since it's handled separately in _calculate_sidewalk_penalty_only
        # Proceed directly with general off-road and lane keeping
        
        projected_driving_wp = carla_map.get_waypoint(current_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if projected_driving_wp and projected_driving_wp.transform:
            if projected_driving_wp.lane_type != carla.LaneType.Driving: 
                reward += self.PENALTY_OFFROAD # General offroad 
            else: # On a driving lane (by projection)
                # Lane Centering
                max_dev = projected_driving_wp.lane_width / 1.8 
                lat_dist = current_location.distance(projected_driving_wp.transform.location) 
                reward += self.LANE_CENTERING_REWARD_FACTOR * (1.0 - min(lat_dist / max_dev, 1.0)**2)

                # Lane Orientation
                v_fwd = vehicle.get_transform().get_forward_vector()
                l_fwd = projected_driving_wp.transform.get_forward_vector()
                v_fwd_2d = np.array([v_fwd.x, v_fwd.y])
                l_fwd_2d = np.array([l_fwd.x, l_fwd.y])
                norm_v = np.linalg.norm(v_fwd_2d)
                norm_l = np.linalg.norm(l_fwd_2d)
                if norm_v > 1e-4 and norm_l > 1e-4:
                    dot_product = np.dot(v_fwd_2d, l_fwd_2d) / (norm_v * norm_l)
                    angle_d = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
                    if angle_d > 20.0:
                        reward -= self.LANE_ORIENTATION_PENALTY_FACTOR * (angle_d / 90.0)
                
                # Solid Lane Crossing (only if on driving lane)
                if lane_invasion_event: # Re-check for other markings if on driving lane
                    for marking in lane_invasion_event.crossed_lane_markings:
                        # Check for solid lines, but skip curbs since they're handled in sidewalk detection
                        if marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid]:
                            reward += self.PENALTY_SOLID_LANE_CROSS
                            # Don't break, could be multiple solid lines or other non-curb events.
        else: # Cannot project to a driving lane (very off-road)
            reward += self.PENALTY_OFFROAD 
        
        return reward, False  # Always return False for sidewalk flag since it's handled separately

    def _check_nearby_waypoints(self, carla_map, location, radius=5.0):
        """Check waypoints in a radius around the location"""
        waypoints = []
        # Check waypoints in a small grid around the location
        for dx in [-radius, 0, radius]:
            for dy in [-radius, 0, radius]:
                check_location = carla.Location(location.x + dx, location.y + dy, location.z)
                wp = carla_map.get_waypoint(check_location, project_to_road=False)
                if wp:
                    waypoints.append(wp)
        return waypoints

    def _is_legitimate_offroad_area(self, waypoints):
        """Check if the area represents a legitimate off-road area like parking"""
        if not waypoints:
            return False
        
        # Check if any nearby waypoints are parking, shoulder, or other legitimate off-road types
        legitimate_types = [carla.LaneType.Parking, carla.LaneType.Shoulder]
        for wp in waypoints:
            if wp.lane_type in legitimate_types:
                return True
        return False

    def _calculate_stuck_reversing_penalty(self, current_speed_mps, is_reversing_action, intended_reverse_action, reward_type) -> float:
        reward = 0.0
        
        if reward_type == "phase0":
            # Simplified stuck/reversing for phase0
            if current_speed_mps < self.MIN_FORWARD_SPEED_THRESHOLD * 0.25 and not is_reversing_action:
                stuck_penalty = self.phase0_stuck_penalty_base * self.phase0_stuck_multiplier_stuck
                reward += stuck_penalty
            elif current_speed_mps < 0 and not is_reversing_action: # Moving backward without reverse gear
                reversing_penalty = self.phase0_stuck_penalty_base * self.phase0_stuck_multiplier_reversing
                reward += reversing_penalty
            # Phase0 also has a small reward for any forward motion
            if current_speed_mps > self.MIN_FORWARD_SPEED_THRESHOLD * 0.5 and not is_reversing_action:
                forward_reward = 0.1
                reward += forward_reward
        elif reward_type == "standard":
            if is_reversing_action: 
                if not intended_reverse_action: # Reversing when not intended (e.g. action was not reverse)
                    unintended_reverse_penalty = self.PENALTY_STUCK_OR_REVERSING_BASE * 1.5
                    reward += unintended_reverse_penalty
                elif current_speed_mps > -self.MIN_FORWARD_SPEED_THRESHOLD: # Intended reverse, but not moving much
                    slow_reverse_penalty = self.PENALTY_STUCK_OR_REVERSING_BASE / 2
                    reward += slow_reverse_penalty
            elif not is_reversing_action: # Not in reverse gear
                if -self.MIN_FORWARD_SPEED_THRESHOLD < current_speed_mps < self.MIN_FORWARD_SPEED_THRESHOLD: # Stuck (low speed)
                    stuck_penalty = self.PENALTY_STUCK_OR_REVERSING_BASE
                    reward += stuck_penalty
                elif current_speed_mps < -self.MIN_FORWARD_SPEED_THRESHOLD: # Moving backward without reverse gear
                    backward_penalty = self.PENALTY_STUCK_OR_REVERSING_BASE * 3
                    reward += backward_penalty
        
        return reward

    def _calculate_traffic_light_reward(self, vehicle, relevant_traffic_light_state, current_speed_mps) -> float:
        reward = 0.0
        if relevant_traffic_light_state and vehicle.is_at_traffic_light():
            if relevant_traffic_light_state == carla.TrafficLightState.Red:
                if current_speed_mps > self.VEHICLE_STOPPED_SPEED_THRESHOLD:
                    reward += self.PENALTY_TRAFFIC_LIGHT_RED_MOVING
                else:
                    reward += self.REWARD_TRAFFIC_LIGHT_STOPPED_AT_RED
            elif relevant_traffic_light_state == carla.TrafficLightState.Green:
                if current_speed_mps > self.MIN_FORWARD_SPEED_THRESHOLD: # Encourage going on green
                    reward += self.REWARD_TRAFFIC_LIGHT_GREEN_PROCEED
        return reward

    def _calculate_proximity_penalty(self, vehicle, current_location, world) -> Tuple[float, bool]:
        reward = 0.0
        proximity_flag = False
        if not world:
            return reward, proximity_flag

        vehicle_transform = vehicle.get_transform()
        vehicle_forward_vector = vehicle_transform.get_forward_vector()
        # Ensure 2D for dot product direction checking
        vehicle_forward_vec_2d = np.array([vehicle_forward_vector.x, vehicle_forward_vector.y])
        norm_fwd = np.linalg.norm(vehicle_forward_vec_2d)

        if norm_fwd < 1e-4: # Avoid issues if vehicle forward vector is zero
            return reward, proximity_flag
        vehicle_forward_vec_2d_norm = vehicle_forward_vec_2d / norm_fwd

        for other_actor in world.get_actors().filter('vehicle.*'):
            if other_actor.id == vehicle.id:
                continue
            
            other_loc = other_actor.get_location()
            if other_loc.distance(current_location) < self.PROXIMITY_THRESHOLD_VEHICLE:
                vec_to_other = other_loc - current_location
                vec_to_other_2d = np.array([vec_to_other.x, vec_to_other.y])
                dist_2d = np.linalg.norm(vec_to_other_2d)

                if dist_2d > 0.01: # Avoid division by zero if overlapping
                    vec_to_other_2d_norm = vec_to_other_2d / dist_2d
                    dot_product = np.dot(vehicle_forward_vec_2d_norm, vec_to_other_2d_norm)
                    if dot_product > 0.707:  # If other vehicle is roughly in front (within 45-degree cone)
                        reward += self.PENALTY_PROXIMITY_VEHICLE_FRONT
                        proximity_flag = True
                        break # One proximity penalty is enough per step
        return reward, proximity_flag

    def _calculate_collision_penalty(self, collision_info) -> Tuple[float, bool]:
        reward = 0.0
        collision_flag = False
        if collision_info.get('count', 0) > 0:
            reward += self.PENALTY_COLLISION
            collision_flag = True
        return reward, collision_flag
    
    def _get_current_reward_type_and_env(self) -> Tuple[Optional[str], Optional['CarlaEnv']]:
        env = self.carla_env_ref() if self.carla_env_ref else None
        if not env:
            logger.error("RewardCalculator: CarlaEnv reference is None!")
            return None, None
        
        # Access curriculum config via the CurriculumManager instance in CarlaEnv
        if not hasattr(env, 'curriculum_manager') or env.curriculum_manager is None:
            logger.error("RewardCalculator: CarlaEnv has no_curriculum_manager or it is None!")
            return "standard", env # Fallback to standard reward type, but log error

        current_phase_config = env.curriculum_manager.get_current_phase_config()
        if not current_phase_config:
            logger.error("RewardCalculator: CurriculumManager returned no current phase config!")
            return "standard", env # Fallback
            
        reward_type = current_phase_config.get("reward_config", "standard")
        return reward_type, env

    def calculate_reward(self, vehicle, current_location, previous_location, 
                         collision_info, relevant_traffic_light_state, 
                         current_action_for_reward, 
                         forward_speed_debug, 
                         carla_map, target_waypoint,
                         lane_invasion_event: Optional[carla.LaneInvasionEvent] = None,
                         action_taken: Optional[Any] = None, 
                         segment_target_reached: bool = False, 
                         distance_to_final_goal: Optional[float] = None,
                         current_road_option: Optional[Any] = None # New: from carla.agents.navigation.local_planner import RoadOption
                        ) -> Tuple[float, bool, bool, bool]:
        """
        Calculates the reward for the current step.
        Returns a tuple: (reward_value, collision_flag_for_hud, proximity_flag_for_hud, on_sidewalk_flag_for_hud)
        """
        total_reward = 0.0
        hud_collision_flag = False
        hud_proximity_flag = False
        hud_on_sidewalk_flag = False

        reward_type, env = self._get_current_reward_type_and_env()
        if not env or not reward_type:
            return self.PENALTY_COLLISION, True, False, False 

        if vehicle is None or not vehicle.is_alive or current_location is None:
            return self.PENALTY_COLLISION, True, False, False

        current_speed_mps = forward_speed_debug
        current_speed_kmh = current_speed_mps * 3.6

        is_reversing_control = vehicle.get_control().reverse 
        intended_reverse_action = (action_taken == 5) if env.discrete_actions else False

        # --- Per-step penalty --- 
        if reward_type == "phase0":
            total_reward += self.phase0_penalty_per_step
        elif reward_type == "standard":
            total_reward += self.PENALTY_PER_STEP

        # --- Goal and Distance Rewards --- 
        current_phase_config = env.curriculum_manager.get_current_phase_config() if env.curriculum_manager else {}
        require_stop_at_goal = current_phase_config.get("require_stop_at_goal", False)
        distance_reward = self._calculate_distance_goal_reward(
            current_location, previous_location, target_waypoint, reward_type, current_speed_mps, require_stop_at_goal
        )
        total_reward += distance_reward

        if segment_target_reached and reward_type == "standard":
            is_not_final_goal = True
            if distance_to_final_goal is not None and distance_to_final_goal < self.WAYPOINT_REACHED_THRESHOLD:
                is_not_final_goal = False
            if is_not_final_goal:
                total_reward += 5.0 
                logger.debug(f"Intermediate segment target reached, +5 reward.")

        # --- Control and Behavior Rewards/Penalties (mostly for standard) --- 
        total_reward += self._calculate_stuck_reversing_penalty(
            current_speed_mps, is_reversing_control, intended_reverse_action, reward_type
        )

        # --- Sidewalk Detection (apply to ALL reward types) ---
        sidewalk_penalty, on_sidewalk_from_detection = self._calculate_sidewalk_penalty_only(
            vehicle, current_location, carla_map, lane_invasion_event
        )
        total_reward += sidewalk_penalty
        if on_sidewalk_from_detection: 
            hud_on_sidewalk_flag = True

        if reward_type == "standard":
            total_reward += self._calculate_speed_reward(vehicle, current_speed_kmh, is_reversing_control)
            
            # Apply remaining lane keeping rewards/penalties (excluding sidewalk which is handled above)
            lane_reward, _ = self._calculate_lane_keeping_rewards_penalties_excluding_sidewalk(
                vehicle, current_location, carla_map, lane_invasion_event
            )
            total_reward += lane_reward

            total_reward += self._calculate_traffic_light_reward(
                vehicle, relevant_traffic_light_state, current_speed_mps
            )
            
            prox_penalty, prox_flag_from_calc = self._calculate_proximity_penalty(
                vehicle, current_location, env.world
            )
            total_reward += prox_penalty
            if prox_flag_from_calc: hud_proximity_flag = True

            # --- New Steering Penalty --- 
            steer_input = vehicle.get_control().steer
            # Check if current_road_option is available and indicates a straight path
            # (RoadOption is an enum, so we need to import it or use its integer values if known)
            # For now, let's assume RoadOption.LANEFOLLOW and RoadOption.STRAIGHT are relevant
            # We might need to import RoadOption from carla.agents.navigation.local_planner
            # For simplicity here, let's assume specific RoadOption values if the import is tricky for now
            # RoadOption.LANEFOLLOW = 4, RoadOption.STRAIGHT = 5 (based on common CARLA agent code)
            is_on_straight_segment = False
            if current_road_option is not None:
                # self.logger.debug(f"Current RoadOption: {current_road_option} (type: {type(current_road_option)})")
                # Convert to int if it's an enum, or compare directly if it's already int/str
                try: # Attempt to convert to int if it's an enum like RoadOption
                    road_option_value = int(current_road_option)
                    if road_option_value == 4 or road_option_value == 5: # LANEFOLLOW or STRAIGHT
                        is_on_straight_segment = True
                except (TypeError, ValueError):
                    # self.logger.warning(f"Could not convert current_road_option '{current_road_option}' to int for steer penalty check.")
                    pass # If it's not an enum that converts to int, this check might fail silently or need adjustment
            
            if is_on_straight_segment and abs(steer_input) > self.STEER_THRESHOLD_STRAIGHT and current_speed_kmh > self.MIN_SPEED_FOR_STEER_PENALTY_KMH:
                penalty = self.PENALTY_EXCESSIVE_STEER_BASE * (abs(steer_input) - self.STEER_THRESHOLD_STRAIGHT) * 5.0 # Scale penalty
                total_reward += penalty
                # self.logger.debug(f"Steer penalty: {penalty:.2f} for steer: {steer_input:.2f} on straight segment.")

        # --- Critical Event Penalties (apply to all reward types) --- 
        coll_penalty, coll_flag_from_calc = self._calculate_collision_penalty(collision_info)
        total_reward += coll_penalty
        if coll_flag_from_calc: hud_collision_flag = True
        
        self.last_collision_flag = hud_collision_flag
        self.last_on_sidewalk_flag = hud_on_sidewalk_flag
        self.last_proximity_penalty_flag = hud_proximity_flag

        return total_reward, hud_collision_flag, hud_proximity_flag, hud_on_sidewalk_flag

    def _check_nearby_driving_lanes(self, carla_map, location, radius=3.0):
        """Check distances to multiple nearby driving lanes to avoid false positives during lane changes"""
        distances = []
        # Check waypoints in a small grid around the location for driving lanes
        for dx in [-radius, 0, radius]:
            for dy in [-radius, 0, radius]:
                check_location = carla.Location(location.x + dx, location.y + dy, location.z)
                wp = carla_map.get_waypoint(check_location, project_to_road=True, lane_type=carla.LaneType.Driving)
                if wp and wp.lane_type == carla.LaneType.Driving:
                    distance = location.distance(wp.transform.location)
                    distances.append(distance)
        return distances 

    def _get_current_phase_type_and_detection_params(self):
        """
        Determine the current phase type and return appropriate sidewalk detection parameters.
        
        Returns:
            Tuple of (phase_type, detection_params) where:
            - phase_type: 'straight', 'steering', or 'default'
            - detection_params: Dictionary with detection thresholds
        """
        env = self.carla_env_ref() if self.carla_env_ref else None
        
        if not env or not hasattr(env, 'curriculum_manager') or not env.curriculum_manager:
            return 'default', self.sidewalk_detection_default
            
        current_phase_config = env.curriculum_manager.get_current_phase_config()
        if not current_phase_config:
            return 'default', self.sidewalk_detection_default
            
        phase_name = current_phase_config.get('name', '').lower()
        spawn_config = current_phase_config.get('spawn_config', '').lower()
        
        # Determine phase type based on phase name and spawn config
        if ('straight' in phase_name or 'phase0' in phase_name or 
            spawn_config == 'fixed_straight' or 'basic' in phase_name):
            return 'straight', self.sidewalk_detection_straight
        elif ('turn' in phase_name or 'steer' in phase_name or 
              spawn_config in ['fixed_simple_turns', 'random'] or 
              'steering' in phase_name or 'complex' in phase_name):
            return 'steering', self.sidewalk_detection_steering
        else:
            return 'default', self.sidewalk_detection_default

    def _is_legitimate_lane_change_advanced(self, lane_invasion_event, detection_params, distance_to_road):
        """
        Advanced detection of legitimate lane changes vs actual curb violations.
        
        Args:
            lane_invasion_event: CARLA lane invasion event
            detection_params: Phase-specific detection parameters
            distance_to_road: Distance from vehicle to nearest driving lane
            
        Returns:
            Tuple of (is_legitimate, reason)
        """
        if not lane_invasion_event:
            return False, "no_lane_invasion_event"
            
        # If phase doesn't allow broken line crossings, any lane invasion is suspicious
        if not detection_params.get('allow_broken_line_crossings', True):
            return False, "phase_disallows_lane_changes"
            
        markings = lane_invasion_event.crossed_lane_markings
        if not markings:
            return False, "no_markings_in_event"
            
        # Analyze the types of markings crossed
        has_curb = any(marking.type == carla.LaneMarkingType.Curb for marking in markings)
        has_solid = any(marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid] for marking in markings)
        has_broken = any(marking.type in [carla.LaneMarkingType.Broken, carla.LaneMarkingType.BrokenBroken] for marking in markings)
        has_crossable = any(marking.type in [carla.LaneMarkingType.Broken, carla.LaneMarkingType.BrokenBroken, carla.LaneMarkingType.BrokenSolid] for marking in markings)
        
        # If curb is involved, it's definitely not a legitimate lane change
        if has_curb:
            return False, "curb_marking_detected"
            
        # If only broken/crossable lines and reasonable distance, it's likely legitimate
        if has_crossable and not has_solid and not has_curb:
            grace_distance = detection_params.get('broken_line_grace_distance', 3.0)
            if distance_to_road <= grace_distance:
                return True, f"legitimate_broken_line_crossing_within_{grace_distance}m"
        
        # If solid lines are crossed, it's not legitimate (unless very close to road)
        if has_solid and distance_to_road > 1.0:
            return False, "solid_line_crossing_far_from_road"
            
        # Mixed case: has both crossable and non-crossable markings
        if has_crossable and has_solid:
            # Could be crossing from broken to solid lane - allow if close to road
            grace_distance = detection_params.get('broken_line_grace_distance', 3.0) * 0.7  # Reduce grace
            if distance_to_road <= grace_distance:
                return True, f"mixed_marking_crossing_within_{grace_distance:.1f}m"
            else:
                return False, f"mixed_marking_crossing_too_far_{distance_to_road:.1f}m"
        
        # Default: if we have any crossable markings and reasonable distance, allow it
        if has_crossable:
            return True, "default_crossable_marking_allowance"
            
        return False, "no_crossable_markings_found" 