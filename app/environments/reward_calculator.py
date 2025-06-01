import carla
import numpy as np
import math
import logging
from typing import Tuple, Dict, List, Optional, Any
from functools import lru_cache
import config

# Performance optimization imports
try:
    import numba
    from numba import jit, types
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba not available. Install numba for reward calculation JIT optimizations.")

logger = logging.getLogger(__name__)

# Numba JIT optimized functions for mathematical computations
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _calculate_distance_numba(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
        """Optimized 3D distance calculation using Numba JIT."""
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    @jit(nopython=True, cache=True)
    def _calculate_2d_distance_numba(x1: float, y1: float, x2: float, y2: float) -> float:
        """Optimized 2D distance calculation using Numba JIT."""
        dx = x2 - x1
        dy = y2 - y1
        return math.sqrt(dx*dx + dy*dy)

    @jit(nopython=True, cache=True)
    def _calculate_gaussian_reward_numba(value: float, target: float, std_dev: float, scale: float) -> float:
        """Optimized Gaussian reward calculation using Numba JIT."""
        diff = value - target
        exponent = -0.5 * (diff / std_dev) * (diff / std_dev)
        return scale * math.exp(exponent)

    @jit(nopython=True, cache=True)
    def _calculate_speed_reward_numba(current_speed: float, target_speed: float, std_dev: float, 
                                     scale: float, is_above_target: bool) -> float:
        """Optimized speed reward calculation with asymmetric penalties using Numba JIT."""
        speed_diff = current_speed - target_speed
        
        if speed_diff >= 0:  # Above target speed
            # More lenient for slightly above target, harsh for excessive speed
            speed_rew = scale * math.exp(-0.5 * (speed_diff / std_dev) * (speed_diff / std_dev))
            if speed_diff > 2 * std_dev:
                # Exponential penalty for dangerous speeding
                excess_speed = speed_diff - 2 * std_dev
                speed_rew -= 0.2 * (excess_speed / 10.0) * (excess_speed / 10.0)
            return speed_rew
        else:  # Below target speed
            # More forgiving for being below target
            abs_diff = abs(speed_diff)
            return scale * math.exp(-0.3 * (abs_diff / std_dev) * (abs_diff / std_dev))

    @jit(nopython=True, cache=True)
    def _calculate_lane_centering_reward_numba(lateral_distance: float, lane_width: float, 
                                              reward_factor: float) -> float:
        """Optimized lane centering reward calculation using Numba JIT."""
        max_dev = lane_width / 1.8
        normalized_dist = min(lateral_distance / max_dev, 1.0)
        return reward_factor * (1.0 - normalized_dist * normalized_dist)

    @jit(nopython=True, cache=True)
    def _calculate_orientation_penalty_numba(angle_degrees: float, threshold: float, 
                                            penalty_factor: float) -> float:
        """Optimized orientation penalty calculation using Numba JIT."""
        if angle_degrees > threshold:
            return -penalty_factor * (angle_degrees / 90.0)
        return 0.0

    @jit(nopython=True, cache=True)
    def _calculate_dot_product_2d_numba(v1x: float, v1y: float, v2x: float, v2y: float) -> float:
        """Optimized 2D dot product calculation using Numba JIT."""
        return v1x * v2x + v1y * v2y

    @jit(nopython=True, cache=True)
    def _calculate_vector_magnitude_2d_numba(x: float, y: float) -> float:
        """Optimized 2D vector magnitude calculation using Numba JIT."""
        return math.sqrt(x*x + y*y)

    @jit(nopython=True, cache=True)
    def _calculate_smoothness_score_numba(throttle_change: float, brake_change: float, 
                                         steer_change: float, weight_steer: float) -> float:
        """Optimized smoothness score calculation using Numba JIT."""
        return max(0.0, 1.0 - (throttle_change + brake_change + steer_change * weight_steer))

else:
    # Fallback functions when numba is not available
    def _calculate_distance_numba(x1, y1, z1, x2, y2, z2):
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def _calculate_2d_distance_numba(x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        return math.sqrt(dx*dx + dy*dy)

    def _calculate_gaussian_reward_numba(value, target, std_dev, scale):
        diff = value - target
        return scale * math.exp(-0.5 * (diff / std_dev)**2)

    def _calculate_speed_reward_numba(current_speed, target_speed, std_dev, scale, is_above_target):
        speed_diff = current_speed - target_speed
        if speed_diff >= 0:
            speed_rew = scale * math.exp(-0.5 * (speed_diff / std_dev)**2)
            if speed_diff > 2 * std_dev:
                excess_speed = speed_diff - 2 * std_dev
                speed_rew -= 0.2 * (excess_speed / 10.0)**2
            return speed_rew
        else:
            abs_diff = abs(speed_diff)
            return scale * math.exp(-0.3 * (abs_diff / std_dev)**2)

    def _calculate_lane_centering_reward_numba(lateral_distance, lane_width, reward_factor):
        max_dev = lane_width / 1.8
        normalized_dist = min(lateral_distance / max_dev, 1.0)
        return reward_factor * (1.0 - normalized_dist**2)

    def _calculate_orientation_penalty_numba(angle_degrees, threshold, penalty_factor):
        if angle_degrees > threshold:
            return -penalty_factor * (angle_degrees / 90.0)
        return 0.0

    def _calculate_dot_product_2d_numba(v1x, v1y, v2x, v2y):
        return v1x * v2x + v1y * v2y

    def _calculate_vector_magnitude_2d_numba(x, y):
        return math.sqrt(x*x + y*y)

    def _calculate_smoothness_score_numba(throttle_change, brake_change, steer_change, weight_steer):
        return max(0.0, 1.0 - (throttle_change + brake_change + steer_change * weight_steer))

class RewardCalculator:
    def __init__(self, reward_configs: Optional[Dict] = None, 
                 target_speed_kmh: Optional[float] = None, 
                 curriculum_phases: Optional[List[Dict]] = None, 
                 carla_env_ref = None):
        """
        Initializes the RewardCalculator with performance optimizations.
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

        # Performance optimization caches
        self._location_cache = {}
        self._cache_max_size = 100
        self._computation_cache = {}
        
        # Previous values for smoothness calculation (performance tracking)
        self._prev_throttle = 0.0
        self._prev_brake = 0.0
        self._prev_steer = 0.0
        self._smoothness_history = []
        self._max_history_size = 10

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

        # Load distance-based penalty parameters for max steps reached
        self.MAX_STEPS_DISTANCE_PENALTY_ENABLED = config.MAX_STEPS_DISTANCE_PENALTY_ENABLED
        self.MAX_STEPS_DISTANCE_PENALTY_MAX = config.MAX_STEPS_DISTANCE_PENALTY_MAX
        self.MAX_STEPS_DISTANCE_PENALTY_MIN = config.MAX_STEPS_DISTANCE_PENALTY_MIN
        self.MAX_STEPS_DISTANCE_PENALTY_MAX_DISTANCE = config.MAX_STEPS_DISTANCE_PENALTY_MAX_DISTANCE
        self.MAX_STEPS_DISTANCE_PENALTY_CLOSE_MULTIPLIER = config.MAX_STEPS_DISTANCE_PENALTY_CLOSE_MULTIPLIER

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

    @lru_cache(maxsize=128)
    def _cached_distance_calculation(self, x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
        """Cached distance calculation for frequently computed distances."""
        return _calculate_distance_numba(x1, y1, z1, x2, y2, z2)

    def _clear_caches_if_full(self):
        """Clear caches if they exceed maximum size to prevent memory issues."""
        if len(self._location_cache) > self._cache_max_size:
            # Keep only the most recent half of entries
            keep_size = self._cache_max_size // 2
            items = list(self._location_cache.items())
            self._location_cache = dict(items[-keep_size:])
        
        if len(self._computation_cache) > self._cache_max_size:
            keep_size = self._cache_max_size // 2
            items = list(self._computation_cache.items())
            self._computation_cache = dict(items[-keep_size:])

    def _calculate_distance_goal_reward(self, current_location, previous_location, target_waypoint, reward_type, current_speed_mps, require_stop) -> float:
        """Optimized distance-to-goal reward calculation with caching."""
        reward = 0.0
        if not target_waypoint or not previous_location or not target_waypoint.transform:
            return reward

        # Use optimized distance calculations
        target_loc = target_waypoint.transform.location
        dist_to_target = _calculate_distance_numba(
            current_location.x, current_location.y, current_location.z,
            target_loc.x, target_loc.y, target_loc.z
        )
        prev_dist_to_target = _calculate_distance_numba(
            previous_location.x, previous_location.y, previous_location.z,
            target_loc.x, target_loc.y, target_loc.z
        )
        
        dist_reduction = prev_dist_to_target - dist_to_target
        stop_ok = (not require_stop) or (current_speed_mps <= config.STOP_AT_GOAL_SPEED_THRESHOLD)

        if reward_type == "phase0":
            if dist_reduction > 0.01:
                distance_component = dist_reduction * self.REWARD_DISTANCE_FACTOR * self.phase0_distance_factor_multiplier
                reward += distance_component
            elif dist_reduction < -0.1:
                distance_penalty = abs(dist_reduction) * self.REWARD_DISTANCE_FACTOR * 0.5
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
        """Enhanced speed reward with performance optimizations and better shaping."""
        reward = 0.0
        control = vehicle.get_control()
        
        # Context-aware target speed adjustment (cached)
        context_adjusted_target = self._get_context_adjusted_target_speed(vehicle, current_speed_kmh)
        
        # Use optimized speed reward calculation
        speed_diff = current_speed_kmh - context_adjusted_target
        is_above_target = speed_diff >= 0
        
        speed_rew = _calculate_speed_reward_numba(
            current_speed_kmh, context_adjusted_target, 
            self.TARGET_SPEED_STD_DEV_KMH, self.TARGET_SPEED_REWARD_FACTOR, 
            is_above_target
        )
        
        # Apply speed reward with context considerations
        if not (control.brake > 0.5 and current_speed_kmh < 5) and not is_reversing_action:
            reward += speed_rew
        
        # Smooth acceleration/deceleration bonus (optimized)
        reward += self._calculate_smoothness_bonus_optimized(vehicle, current_speed_kmh)
        
        # Energy efficiency bonus (reward for coasting when appropriate)
        if 0.1 < control.throttle < 0.3 and abs(speed_diff) < 5:
            reward += 0.1  # Small bonus for efficient driving
        
        return reward

    @lru_cache(maxsize=64)
    def _get_context_adjusted_target_speed(self, vehicle, current_speed_kmh) -> float:
        """Cached context-aware target speed adjustment."""
        base_target = self.target_speed_kmh
        
        # Get environment reference for context
        env = self.carla_env_ref() if self.carla_env_ref else None
        if not env:
            return base_target
        
        # Reduce target speed near intersections or traffic lights
        if hasattr(env, 'world') and env.world:
            vehicle_location = vehicle.get_location()
            
            # Check for nearby traffic lights (optimized distance check)
            traffic_lights = env.world.get_actors().filter('traffic.traffic_light')
            for tl in traffic_lights:
                tl_loc = tl.get_location()
                distance = _calculate_2d_distance_numba(
                    vehicle_location.x, vehicle_location.y,
                    tl_loc.x, tl_loc.y
                )
                if distance < 30:
                    if tl.state == carla.TrafficLightState.Red:
                        base_target *= 0.3  # Slow down for red lights
                    elif tl.state == carla.TrafficLightState.Yellow:
                        base_target *= 0.6  # Moderate slowdown for yellow
        
        # Reduce target speed in curves (based on steering input)
        steering_magnitude = abs(vehicle.get_control().steer)
        if steering_magnitude > 0.3:
            curve_factor = max(0.5, 1.0 - steering_magnitude * 0.5)
            base_target *= curve_factor
        
        return base_target

    def _calculate_smoothness_bonus_optimized(self, vehicle, current_speed_kmh) -> float:
        """Optimized smoothness bonus calculation with history tracking."""
        control = vehicle.get_control()
        
        # Calculate control input changes
        throttle_change = abs(control.throttle - self._prev_throttle)
        brake_change = abs(control.brake - self._prev_brake)
        steer_change = abs(control.steer - self._prev_steer)
        
        # Use optimized smoothness calculation
        smoothness_score = _calculate_smoothness_score_numba(
            throttle_change, brake_change, steer_change, 2.0  # Steering weighted more
        )
        
        # Update smoothness history for trend analysis
        self._smoothness_history.append(smoothness_score)
        if len(self._smoothness_history) > self._max_history_size:
            self._smoothness_history.pop(0)
        
        # Update previous values
        self._prev_throttle = control.throttle
        self._prev_brake = control.brake
        self._prev_steer = control.steer
        
        # Calculate bonus with trend consideration
        smoothness_bonus = smoothness_score * 0.05
        
        # Additional bonus for consistent smoothness
        if len(self._smoothness_history) >= 3:
            avg_smoothness = sum(self._smoothness_history) / len(self._smoothness_history)
            if avg_smoothness > 0.8:
                smoothness_bonus += 0.02  # Extra bonus for sustained smooth driving
        
        return max(0, smoothness_bonus)

    def _calculate_lane_keeping_rewards_penalties(self, vehicle, current_location, carla_map, lane_invasion_event) -> Tuple[float, bool]:
        """Optimized lane keeping calculation with performance improvements."""
        reward = 0.0
        on_sidewalk_flag = False

        if not carla_map:
            return reward, on_sidewalk_flag

        # 1. Prioritize Lane Invasion for Curb Detection
        if lane_invasion_event:
            for marking in lane_invasion_event.crossed_lane_markings:
                if marking.type == carla.LaneMarkingType.Curb:
                    reward += self.PENALTY_SIDEWALK
                    on_sidewalk_flag = True
                    break

        # 2. If no curb was hit, check for direct sidewalk lane type
        if not on_sidewalk_flag:
            current_waypoint_at_location = carla_map.get_waypoint(current_location, project_to_road=False) 
            if current_waypoint_at_location and current_waypoint_at_location.lane_type == carla.LaneType.Sidewalk:
                reward += self.PENALTY_SIDEWALK
                on_sidewalk_flag = True

        # 3. If still not flagged for sidewalk, proceed with general off-road and lane keeping
        if not on_sidewalk_flag:
            projected_driving_wp = carla_map.get_waypoint(current_location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if projected_driving_wp and projected_driving_wp.transform:
                if projected_driving_wp.lane_type != carla.LaneType.Driving: 
                    reward += self.PENALTY_OFFROAD
                else:
                    # Optimized Lane Centering calculation
                    proj_loc = projected_driving_wp.transform.location
                    lateral_distance = _calculate_distance_numba(
                        current_location.x, current_location.y, current_location.z,
                        proj_loc.x, proj_loc.y, proj_loc.z
                    )
                    
                    lane_centering_reward = _calculate_lane_centering_reward_numba(
                        lateral_distance, projected_driving_wp.lane_width, 
                        self.LANE_CENTERING_REWARD_FACTOR
                    )
                    reward += lane_centering_reward

                    # Optimized Lane Orientation calculation
                    v_fwd = vehicle.get_transform().get_forward_vector()
                    l_fwd = projected_driving_wp.transform.get_forward_vector()
                    
                    # Use optimized vector operations
                    v_norm = _calculate_vector_magnitude_2d_numba(v_fwd.x, v_fwd.y)
                    l_norm = _calculate_vector_magnitude_2d_numba(l_fwd.x, l_fwd.y)
                    
                    if v_norm > 1e-4 and l_norm > 1e-4:
                        dot_product = _calculate_dot_product_2d_numba(v_fwd.x, v_fwd.y, l_fwd.x, l_fwd.y)
                        dot_product_normalized = dot_product / (v_norm * l_norm)
                        dot_product_clamped = max(-1.0, min(1.0, dot_product_normalized))
                        angle_d = math.degrees(math.acos(dot_product_clamped))
                        
                        orientation_penalty = _calculate_orientation_penalty_numba(
                            angle_d, 20.0, self.LANE_ORIENTATION_PENALTY_FACTOR
                        )
                        reward += orientation_penalty
                    
                    # Solid Lane Crossing check
                    if lane_invasion_event:
                        for marking in lane_invasion_event.crossed_lane_markings:
                            if marking.type in [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidSolid]:
                                reward += self.PENALTY_SOLID_LANE_CROSS
            else:
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
                
                # Solid Lane Crossing (only if on driving lane and not already a sidewalk event)
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
        """Enhanced traffic light reward with anticipatory behavior and smooth transitions."""
        reward = 0.0
        
        if not relevant_traffic_light_state:
            return reward
            
        # Get environment reference for enhanced context
        env = self.carla_env_ref() if self.carla_env_ref else None
        vehicle_location = vehicle.get_location()
        
        # Enhanced traffic light behavior
        if relevant_traffic_light_state == carla.TrafficLightState.Red:
            if vehicle.is_at_traffic_light():
                if current_speed_mps > self.VEHICLE_STOPPED_SPEED_THRESHOLD:
                    # Graduated penalty based on speed when running red light
                    speed_factor = min(current_speed_mps / 5.0, 2.0)  # Cap at 2x penalty
                    reward += self.PENALTY_TRAFFIC_LIGHT_RED_MOVING * speed_factor
                else:
                    # Bonus for proper stopping
                    reward += self.REWARD_TRAFFIC_LIGHT_STOPPED_AT_RED
                    
                    # Additional bonus for stopping smoothly (not abruptly)
                    if hasattr(self, '_prev_speed_mps') and self._prev_speed_mps > 2.0:
                        deceleration = self._prev_speed_mps - current_speed_mps
                        if 0.5 < deceleration < 3.0:  # Smooth deceleration range
                            reward += 5.0  # Bonus for smooth stopping
            else:
                # Anticipatory behavior: reward for slowing down when approaching red light
                if env and hasattr(env, 'world'):
                    traffic_lights = env.world.get_actors().filter('traffic.traffic_light')
                    for tl in traffic_lights:
                        distance = tl.get_location().distance(vehicle_location)
                        if 10 < distance < 50 and tl.state == carla.TrafficLightState.Red:
                            # Reward for anticipatory slowing
                            if current_speed_mps < self.target_speed_kmh / 3.6 * 0.7:
                                anticipation_bonus = (50 - distance) / 50 * 3.0
                                reward += anticipation_bonus
                                
        elif relevant_traffic_light_state == carla.TrafficLightState.Yellow:
            # Enhanced yellow light behavior
            if env and hasattr(env, 'world'):
                traffic_lights = env.world.get_actors().filter('traffic.traffic_light')
                for tl in traffic_lights:
                    distance = tl.get_location().distance(vehicle_location)
                    if distance < 30 and tl.state == carla.TrafficLightState.Yellow:
                        # Decision-making reward based on distance and speed
                        stopping_distance = (current_speed_mps ** 2) / (2 * 4.0)  # Assume 4 m/s² deceleration
                        
                        if distance > stopping_distance + 5:  # Safe to proceed
                            if current_speed_mps > self.MIN_FORWARD_SPEED_THRESHOLD:
                                reward += 3.0  # Reward for proceeding when safe
                        else:  # Should stop
                            if current_speed_mps < self.VEHICLE_STOPPED_SPEED_THRESHOLD:
                                reward += 8.0  # Reward for stopping at yellow when appropriate
                            elif current_speed_mps < self.target_speed_kmh / 3.6 * 0.5:
                                reward += 2.0  # Partial reward for slowing down
                                
        elif relevant_traffic_light_state == carla.TrafficLightState.Green:
            if vehicle.is_at_traffic_light():
                if current_speed_mps > self.MIN_FORWARD_SPEED_THRESHOLD:
                    # Base reward for proceeding on green
                    reward += self.REWARD_TRAFFIC_LIGHT_GREEN_PROCEED
                    
                    # Bonus for appropriate acceleration from stop
                    if hasattr(self, '_prev_speed_mps') and self._prev_speed_mps < 1.0:
                        acceleration = current_speed_mps - self._prev_speed_mps
                        if 0.5 < acceleration < 2.0:  # Smooth acceleration
                            reward += 3.0
                else:
                    # Small penalty for hesitating at green light
                    reward -= 1.0
        
        # Store speed for next calculation
        self._prev_speed_mps = current_speed_mps
        
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
        if ('straight' in phase_name or 'phase1' in phase_name or 'phase2' in phase_name or 'phase3' in phase_name or
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

    def calculate_max_steps_distance_penalty(self, current_location: carla.Location, 
                                            target_waypoint, distance_to_final_goal: Optional[float] = None) -> float:
        """
        Calculate distance-based penalty when max steps are reached without success.
        
        Args:
            current_location: Current vehicle location
            target_waypoint: Target waypoint
            distance_to_final_goal: Distance to final goal (preferred if available)
            
        Returns:
            float: Distance-based penalty (always negative), or 0 if disabled
        """
        if not self.MAX_STEPS_DISTANCE_PENALTY_ENABLED:
            return 0.0
            
        if not current_location:
            return self.MAX_STEPS_DISTANCE_PENALTY_MAX
            
        # Use distance_to_final_goal if available, otherwise calculate from target_waypoint
        if distance_to_final_goal is not None:
            distance_to_goal = distance_to_final_goal
        elif target_waypoint and target_waypoint.transform:
            target_loc = target_waypoint.transform.location
            distance_to_goal = current_location.distance(target_loc)
        else:
            # No goal information available, apply maximum penalty
            return self.MAX_STEPS_DISTANCE_PENALTY_MAX
            
        # Calculate penalty based on distance - closer to goal gets smaller penalty
        close_distance_threshold = self.WAYPOINT_REACHED_THRESHOLD * self.MAX_STEPS_DISTANCE_PENALTY_CLOSE_MULTIPLIER
        
        if distance_to_goal <= self.WAYPOINT_REACHED_THRESHOLD:
            # Very close to goal - minimal penalty (agent almost made it)
            penalty = self.MAX_STEPS_DISTANCE_PENALTY_MIN * 0.5
        elif distance_to_goal <= close_distance_threshold:
            # Close to goal - small penalty
            penalty = self.MAX_STEPS_DISTANCE_PENALTY_MIN
        elif distance_to_goal >= self.MAX_STEPS_DISTANCE_PENALTY_MAX_DISTANCE:
            # Very far from goal - maximum penalty
            penalty = self.MAX_STEPS_DISTANCE_PENALTY_MAX
        else:
            # Linear interpolation between min and max penalty based on distance
            distance_ratio = (distance_to_goal - close_distance_threshold) / \
                           (self.MAX_STEPS_DISTANCE_PENALTY_MAX_DISTANCE - close_distance_threshold)
            distance_ratio = min(1.0, max(0.0, distance_ratio))  # Clamp to [0, 1]
            penalty = self.MAX_STEPS_DISTANCE_PENALTY_MIN + (self.MAX_STEPS_DISTANCE_PENALTY_MAX - self.MAX_STEPS_DISTANCE_PENALTY_MIN) * distance_ratio
            
        logger.debug(f"Max steps penalty: {penalty:.2f} for distance {distance_to_goal:.2f}m to goal")
        return penalty 