#!/usr/bin/env python3
"""
Action utilities for CARLA RL environment.

This module provides utilities for handling discrete actions, including
remapping actions when certain capabilities are disabled (e.g., reverse, steering).
All action mappings are loaded from the centralized configuration system.
"""

import carla
import logging
import os
import sys
from typing import Dict, Any, Tuple, Optional

# Add project root to Python path for local development
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from utils/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import app.config as config

logger = logging.getLogger(__name__)

def get_vehicle_control_from_discrete_action(action_index: int, allow_reverse: bool = True, allow_steering: bool = True) -> Tuple[carla.VehicleControl, str, int]:
    """
    Get vehicle control and action name from discrete action index using centralized configuration.
    
    Args:
        action_index: The discrete action index selected by the agent
        allow_reverse: Whether reverse actions are allowed
        allow_steering: Whether steering actions are allowed
        
    Returns:
        Tuple of (carla.VehicleControl, action_name, effective_action_index)
        
    Raises:
        ValueError: If no action maps are available or action map is empty
    """
    # Determine which action map to use based on capabilities
    action_map = _get_action_map_for_capabilities(allow_reverse, allow_steering)
    
    if not action_map:
        raise ValueError("No action map available for the specified capabilities")
    
    # Get the action definition with bounds checking
    if action_index in action_map:
        action_def = action_map[action_index]
        effective_action_index = action_index
    else:
        # Log warning and fallback to a safe action
        logger.warning(f"Action index {action_index} out of bounds for current action map. "
                      f"Available indices: {list(action_map.keys())}. Falling back to safe brake action.")
        
        # Find the first brake action or fallback to index 0
        safe_action_index = _find_safe_action_index(action_map)
        action_def = action_map[safe_action_index]
        effective_action_index = safe_action_index
    
    # Create vehicle control from action definition
    control = _create_vehicle_control_from_action_def(action_def)
    action_name = action_def.get("name", f"Action_{effective_action_index}")
    
    return control, action_name, effective_action_index

def _get_action_map_for_capabilities(allow_reverse: bool, allow_steering: bool) -> Dict[int, Dict[str, Any]]:
    """
    Get the appropriate action map based on allowed capabilities.
    
    Args:
        allow_reverse: Whether reverse actions are allowed
        allow_steering: Whether steering actions are allowed
        
    Returns:
        Dictionary mapping action indices to control parameters
        
    Raises:
        ValueError: If configuration is invalid or action maps are empty
    """
    try:
        if not allow_steering:
            # Use forward-only action map (no steering)
            action_map = config.DISCRETE_ACTION_MAP_NO_STEERING
            map_name = "no_steering"
        elif not allow_reverse:
            # Use no-reverse action map
            action_map = config.DISCRETE_ACTION_MAP_NO_REVERSE
            map_name = "no_reverse"
        else:
            # Use full action map
            action_map = config.DISCRETE_ACTION_MAP
            map_name = "full"
        
        if not action_map:
            raise ValueError(f"Action map '{map_name}' is empty or not properly configured")
            
        # Validate action map structure
        for action_idx, action_def in action_map.items():
            if not isinstance(action_def, dict):
                raise ValueError(f"Invalid action definition for index {action_idx} in '{map_name}' map")
            
            # Check required fields
            required_fields = ["throttle", "steer", "brake", "reverse", "name"]
            missing_fields = [field for field in required_fields if field not in action_def]
            if missing_fields:
                raise ValueError(f"Action {action_idx} in '{map_name}' map missing required fields: {missing_fields}")
        
        logger.debug(f"Using action map '{map_name}' with {len(action_map)} actions")
        return action_map
        
    except (AttributeError, KeyError) as e:
        raise ValueError(f"Failed to load action map from configuration: {e}")


def _find_safe_action_index(action_map: Dict[int, Dict[str, Any]]) -> int:
    """
    Find a safe action index (preferably brake) from the action map.
    
    Args:
        action_map: Dictionary mapping action indices to control parameters
        
    Returns:
        Index of a safe action
    """
    # Look for brake actions first
    for idx, action_def in action_map.items():
        if action_def.get("brake", 0.0) > 0.5 and action_def.get("throttle", 1.0) == 0.0:
            logger.debug(f"Found brake action at index {idx}")
            return idx
    
    # If no brake action found, use the first available action
    first_index = min(action_map.keys()) if action_map else 0
    logger.debug(f"No brake action found, using first available action at index {first_index}")
    return first_index


def _create_vehicle_control_from_action_def(action_def: Dict[str, Any]) -> carla.VehicleControl:
    """
    Create a CARLA VehicleControl object from an action definition.
    
    Args:
        action_def: Dictionary containing action parameters
        
    Returns:
        carla.VehicleControl object
        
    Raises:
        ValueError: If action definition contains invalid values
    """
    try:
        control = carla.VehicleControl()
        
        # Set control values with validation
        throttle = action_def.get("throttle", 0.0)
        steer = action_def.get("steer", 0.0)
        brake = action_def.get("brake", 0.0)
        reverse = action_def.get("reverse", False)
        
        # Validate ranges
        if not (0.0 <= throttle <= 1.0):
            raise ValueError(f"Throttle value {throttle} out of range [0.0, 1.0]")
        if not (-1.0 <= steer <= 1.0):
            raise ValueError(f"Steer value {steer} out of range [-1.0, 1.0]")
        if not (0.0 <= brake <= 1.0):
            raise ValueError(f"Brake value {brake} out of range [0.0, 1.0]")
        
        control.throttle = float(throttle)
        control.steer = float(steer)
        control.brake = float(brake)
        control.reverse = bool(reverse)
        
        return control
        
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid action definition: {e}")

def get_action_space_size(allow_reverse: bool = True, allow_steering: bool = True) -> int:
    """
    Get the size of the action space based on enabled capabilities using centralized configuration.
    
    Args:
        allow_reverse: Whether reverse actions are allowed
        allow_steering: Whether steering actions are allowed
        
    Returns:
        Integer size of the action space
        
    Raises:
        ValueError: If action map is not properly configured
    """
    try:
        action_map = _get_action_map_for_capabilities(allow_reverse, allow_steering)
        return len(action_map)
    except ValueError as e:
        logger.error(f"Failed to determine action space size: {e}")
        raise


def get_available_actions_info(allow_reverse: bool = True, allow_steering: bool = True) -> Dict[int, str]:
    """
    Get information about available actions for the given capabilities.
    
    Args:
        allow_reverse: Whether reverse actions are allowed
        allow_steering: Whether steering actions are allowed
        
    Returns:
        Dictionary mapping action indices to action names
        
    Raises:
        ValueError: If action map is not properly configured
    """
    try:
        action_map = _get_action_map_for_capabilities(allow_reverse, allow_steering)
        return {idx: action_def.get("name", f"Action_{idx}") 
                for idx, action_def in action_map.items()}
    except ValueError as e:
        logger.error(f"Failed to get available actions info: {e}")
        raise


def validate_action_configuration() -> bool:
    """
    Validate that all action configurations are properly loaded and valid.
    
    Returns:
        True if all configurations are valid, False otherwise
    """
    try:
        # Test all action map combinations
        test_combinations = [
            (True, True),    # Full capabilities
            (False, True),   # No reverse
            (True, False),   # No steering
            (False, False),  # No reverse, no steering
        ]
        
        for allow_reverse, allow_steering in test_combinations:
            action_map = _get_action_map_for_capabilities(allow_reverse, allow_steering)
            
            # Validate each action in the map
            for action_idx, action_def in action_map.items():
                # Test control creation
                _create_vehicle_control_from_action_def(action_def)
        
        logger.info("All action configurations validated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Action configuration validation failed: {e}")
        return False

def apply_action_to_vehicle(vehicle: carla.Actor, action_index: int, action_map: Dict[int, Dict[str, Any]], logger: Optional[logging.Logger] = None) -> carla.VehicleControl:
    """
    Applies a discrete action to a CARLA vehicle and returns the resulting VehicleControl.
    
    Args:
        vehicle: The CARLA vehicle actor to control
        action_index: The discrete action index to apply
        action_map: Mapping of action indices to action definitions
        logger: Optional logger for debugging
    
    Returns:
        carla.VehicleControl: The control object applied to the vehicle
    
    Raises:
        ValueError: If action_index is invalid or action_map is malformed
    """
    if not action_map:
        raise ValueError("Action map cannot be empty")
    
    if action_index not in action_map:
        if logger:
            logger.warning(f"Invalid action index {action_index}. Using default action (index 0).")
        action_index = 0 if 0 in action_map else list(action_map.keys())[0]
    
    action_def = action_map[action_index]
    
    if not isinstance(action_def, dict):
        raise ValueError(f"Action definition for index {action_index} must be a dictionary")
    
    try:
        control = create_vehicle_control_from_action(action_def)
        vehicle.apply_control(control)
        
        if logger:
            logger.debug(f"Applied action {action_index}: {action_def}")
        
        return control
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to apply action {action_index}: {e}")
        raise

def create_vehicle_control_from_action(action_def: Dict[str, Any]) -> carla.VehicleControl:
    """
    Creates a CARLA VehicleControl object from an action definition dictionary.
    
    Args:
        action_def: Dictionary containing control values (throttle, steer, brake, etc.)
    
    Returns:
        carla.VehicleControl: The control object
    
    Raises:
        ValueError: If action_def is invalid or contains invalid values
    """
    if not isinstance(action_def, dict):
        raise ValueError("Action definition must be a dictionary")
    
    control = carla.VehicleControl()
    
    control.throttle = _validate_control_value(action_def.get('throttle', 0.0), 'throttle')
    control.steer = _validate_control_value(action_def.get('steer', 0.0), 'steer', -1.0, 1.0)
    control.brake = _validate_control_value(action_def.get('brake', 0.0), 'brake')
    control.hand_brake = bool(action_def.get('hand_brake', False))
    control.reverse = bool(action_def.get('reverse', False))
    control.manual_gear_shift = bool(action_def.get('manual_gear_shift', False))
    control.gear = int(action_def.get('gear', 0))
    
    return control 