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

        # Allow reward_configs dictionary to override any of the above loaded from config
        # This provides flexibility for per-phase specific overrides passed via curriculum
        for key, value in self.reward_configs.items():
            if hasattr(self, key.upper()): # Match config naming convention (e.g. PENALTY_COLLISION)
                setattr(self, key.upper(), value)
                logger.info(f"RewardCalculator: Overriding '{key.upper()}' with {value} from reward_configs.")
            elif hasattr(self, key): # Match direct attribute name (e.g. phase0_penalty_per_step)
                setattr(self, key, value)
                logger.info(f"RewardCalculator: Overriding '{key}' with {value} from reward_configs.")

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
                reward += dist_reduction * self.REWARD_DISTANCE_FACTOR * self.phase0_distance_factor_multiplier
            elif dist_reduction < -0.1: # Penalize moving away more strongly in phase0
                reward -= abs(dist_reduction) * self.REWARD_DISTANCE_FACTOR * 0.5 # Standard factor for moving away
            if dist_to_target < self.WAYPOINT_REACHED_THRESHOLD and stop_ok:
                reward += self.REWARD_GOAL_REACHED * self.phase0_goal_reward_multiplier
        elif reward_type == "standard":
            reward += dist_reduction * self.REWARD_DISTANCE_FACTOR
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
        on_sidewalk_flag = False
        if not carla_map:
            return reward, on_sidewalk_flag

        wp = carla_map.get_waypoint(current_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        if wp and wp.transform: 
            if wp.lane_type != carla.LaneType.Driving: 
                reward += self.PENALTY_OFFROAD
                # Check for sidewalk if off-road (e.g. on grass leading to curb)
                if lane_invasion_event:
                    for marking in lane_invasion_event.crossed_lane_markings:
                        if marking.type == carla.LaneMarkingType.Curb:
                            reward += self.PENALTY_SIDEWALK # Additional penalty for sidewalk
                            on_sidewalk_flag = True
                            logger.debug(f"Penalty for off-road curb crossing (sidewalk): {marking.type}")
                            break # Sidewalk is critical
            else: # On a driving lane
                # Lane Centering
                max_dev = wp.lane_width / 1.8 # Stricter centering
                lat_dist = current_location.distance(wp.transform.location) 
                reward += self.LANE_CENTERING_REWARD_FACTOR * (1.0 - min(lat_dist / max_dev, 1.0)**2)

                # Lane Orientation
                v_fwd = vehicle.get_transform().get_forward_vector()
                l_fwd = wp.transform.get_forward_vector()
                # Ensure vectors are 2D (x, y) for heading calculation
                v_fwd_2d = np.array([v_fwd.x, v_fwd.y])
                l_fwd_2d = np.array([l_fwd.x, l_fwd.y])
                norm_v = np.linalg.norm(v_fwd_2d)
                norm_l = np.linalg.norm(l_fwd_2d)

                if norm_v > 1e-4 and norm_l > 1e-4: # Avoid division by zero
                    dot_product = np.dot(v_fwd_2d, l_fwd_2d) / (norm_v * norm_l)
                    angle_d = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
                    if angle_d > 20.0: # Penalize large deviations
                        reward -= self.LANE_ORIENTATION_PENALTY_FACTOR * (angle_d / 90.0) # Scale penalty by deviation

                # Solid Lane Crossing & Sidewalk via Lane Invasion (when on a driving lane)
                if lane_invasion_event:
                    for marking in lane_invasion_event.crossed_lane_markings:
                        if marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid]:
                            reward += self.PENALTY_SOLID_LANE_CROSS
                            logger.debug(f"Penalty for crossing solid lane: {marking.type}")
                            # Don't break, sidewalk might be more severe or also present
                        elif marking.type == carla.LaneMarkingType.Curb:
                            reward += self.PENALTY_SIDEWALK
                            on_sidewalk_flag = True
                            logger.debug(f"Penalty for crossing curb (sidewalk): {marking.type}")
                            # break # Sidewalk is critical, prioritize this penalty
        else: # Off-road (wp is None or wp.lane_type is not Driving)
            reward += self.PENALTY_OFFROAD 
            if lane_invasion_event: # Check for sidewalk even if broadly off-road
                for marking in lane_invasion_event.crossed_lane_markings:
                    if marking.type == carla.LaneMarkingType.Curb:
                        reward += self.PENALTY_SIDEWALK # Additional penalty for sidewalk
                        on_sidewalk_flag = True
                        logger.debug(f"Penalty for general off-road curb crossing (sidewalk): {marking.type}")
                        break # Sidewalk is critical
        return reward, on_sidewalk_flag

    def _calculate_stuck_reversing_penalty(self, current_speed_mps, is_reversing_action, intended_reverse_action, reward_type) -> float:
        reward = 0.0
        if reward_type == "phase0":
            # Simplified stuck/reversing for phase0
            if current_speed_mps < self.MIN_FORWARD_SPEED_THRESHOLD * 0.25 and not is_reversing_action:
                reward += self.phase0_stuck_penalty_base * self.phase0_stuck_multiplier_stuck
            elif current_speed_mps < 0 and not is_reversing_action: # Moving backward without reverse gear
                reward += self.phase0_stuck_penalty_base * self.phase0_stuck_multiplier_reversing
            # Phase0 also has a small reward for any forward motion
            if current_speed_mps > self.MIN_FORWARD_SPEED_THRESHOLD * 0.5 and not is_reversing_action:
                reward += 0.1 # Small incentive to move forward
        elif reward_type == "standard":
            if is_reversing_action: 
                if not intended_reverse_action: # Reversing when not intended (e.g. action was not reverse)
                    reward += self.PENALTY_STUCK_OR_REVERSING_BASE * 1.5 
                elif current_speed_mps > -self.MIN_FORWARD_SPEED_THRESHOLD : # Intended reverse, but not moving much
                    reward += self.PENALTY_STUCK_OR_REVERSING_BASE / 2 
            elif not is_reversing_action: # Not in reverse gear
                if -self.MIN_FORWARD_SPEED_THRESHOLD < current_speed_mps < self.MIN_FORWARD_SPEED_THRESHOLD: # Stuck (low speed)
                    reward += self.PENALTY_STUCK_OR_REVERSING_BASE
                elif current_speed_mps < -self.MIN_FORWARD_SPEED_THRESHOLD: # Moving backward without reverse gear
                    reward += self.PENALTY_STUCK_OR_REVERSING_BASE * 3
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
                         current_action_for_reward, # This name matches the old direct param
                         forward_speed_debug, 
                         carla_map, target_waypoint,
                         lane_invasion_event: Optional[carla.LaneInvasionEvent] = None,
                         action_taken: Optional[Any] = None # Add the new action_taken parameter
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
        # Use the `action_taken` parameter now, which corresponds to `current_action_for_reward` from CarlaEnv
        intended_reverse_action = (action_taken == 5) if env.discrete_actions else False

        if reward_type == "phase0":
            total_reward += self.phase0_penalty_per_step
            if carla_map:
                wp_phase0 = carla_map.get_waypoint(current_location, project_to_road=True, lane_type=carla.LaneType.Any)
                if wp_phase0 and wp_phase0.lane_type != carla.LaneType.Driving:
                    total_reward += self.phase0_offroad_penalty
                elif not wp_phase0:
                    total_reward += self.phase0_offroad_penalty * self.phase0_offroad_no_waypoint_multiplier
        elif reward_type == "standard":
            total_reward += self.PENALTY_PER_STEP

        current_phase_config = env.curriculum_manager.get_current_phase_config() if env.curriculum_manager else {}
        require_stop_at_goal = current_phase_config.get("require_stop_at_goal", False)
        total_reward += self._calculate_distance_goal_reward(
            current_location, previous_location, target_waypoint, reward_type, current_speed_mps, require_stop_at_goal
        )

        total_reward += self._calculate_stuck_reversing_penalty(
            current_speed_mps, is_reversing_control, intended_reverse_action, reward_type # intended_reverse_action uses action_taken
        )

        if reward_type == "standard":
            total_reward += self._calculate_speed_reward(vehicle, current_speed_kmh, is_reversing_control)
            
            lane_reward, on_sidewalk_from_lane_calc = self._calculate_lane_keeping_rewards_penalties(
                vehicle, current_location, carla_map, lane_invasion_event
            )
            total_reward += lane_reward
            if on_sidewalk_from_lane_calc: hud_on_sidewalk_flag = True

            total_reward += self._calculate_traffic_light_reward(
                vehicle, relevant_traffic_light_state, current_speed_mps
            )
            
            prox_penalty, prox_flag_from_calc = self._calculate_proximity_penalty(
                vehicle, current_location, env.world
            )
            total_reward += prox_penalty
            if prox_flag_from_calc: hud_proximity_flag = True

        coll_penalty, coll_flag_from_calc = self._calculate_collision_penalty(collision_info)
        total_reward += coll_penalty
        if coll_flag_from_calc: hud_collision_flag = True
        
        # Update internal state flags that CarlaEnv._check_done might use
        self.last_collision_flag = hud_collision_flag
        self.last_on_sidewalk_flag = hud_on_sidewalk_flag
        self.last_proximity_penalty_flag = hud_proximity_flag

        return total_reward, hud_collision_flag, hud_proximity_flag, hud_on_sidewalk_flag 