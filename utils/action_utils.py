#!/usr/bin/env python3
"""
Action utilities for CARLA RL environment.

This module provides utilities for handling discrete actions, including
remapping actions when certain capabilities are disabled (e.g., reverse, steering).
"""

import carla
import logging
import os
import sys
from typing import Dict, Any, Tuple

# Add project root to Python path for local development
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from utils/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# It might be cleaner to import config and use config.DISCRETE_ACTION_MAP directly,
# or pass the maps as arguments if this util needs to be ultra-generic.
# For now, let's assume direct import for simplicity within this project structure.
import app.config as config

logger = logging.getLogger(__name__)

def get_vehicle_control_from_discrete_action(action_index: int, allow_reverse: bool = True, allow_steering: bool = True):
    """
    Get vehicle control and action name from discrete action index.
    
    Args:
        action_index: The discrete action index selected by the agent
        allow_reverse: Whether reverse actions are allowed
        allow_steering: Whether steering actions are allowed
        
    Returns:
        Tuple of (carla.VehicleControl, action_name, effective_action_index)
    """
    # Determine which action map to use
    if not allow_steering:
        # Use forward-only action map (no steering)
        action_map = _get_no_steering_action_map()
    elif not allow_reverse:
        # Use no-reverse action map
        action_map = config.DISCRETE_ACTION_MAP_NO_REVERSE
    else:
        # Use full action map
        action_map = config.DISCRETE_ACTION_MAP
    
    # Get the action definition
    if action_index in action_map:
        action_def = action_map[action_index]
        effective_action_index = action_index
    else:
        # Fallback to a safe action if index is out of bounds
        action_def = action_map.get(0, {
            "throttle": 0.0, "steer": 0.0, "brake": 1.0, "reverse": False, "name": "SafeBrake"
        })
        effective_action_index = 0
    
    # Create vehicle control
    control = carla.VehicleControl()
    control.throttle = float(action_def.get("throttle", 0.0))
    control.steer = float(action_def.get("steer", 0.0))
    control.brake = float(action_def.get("brake", 0.0))
    control.reverse = bool(action_def.get("reverse", False))
    
    action_name = action_def.get("name", f"Action_{action_index}")
    
    return control, action_name, effective_action_index

def _get_no_steering_action_map():
    """
    Get action map with steering disabled (forward-only driving).
    
    Returns:
        Dictionary mapping action indices to control parameters
    """
    return {
        0: {
            "throttle": 0.6,
            "steer": 0.0,
            "brake": 0.0,
            "reverse": False,
            "name": "Forward-Medium"
        },
        1: {
            "throttle": 0.3,
            "steer": 0.0,
            "brake": 0.0,
            "reverse": False,
            "name": "Forward-Slow"
        },
        2: {
            "throttle": 0.8,
            "steer": 0.0,
            "brake": 0.0,
            "reverse": False,
            "name": "Forward-Fast"
        },
        3: {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 1.0,
            "reverse": False,
            "name": "Brake"
        },
        4: {
            "throttle": 0.2,
            "steer": 0.0,
            "brake": 0.0,
            "reverse": False,
            "name": "Coast"
        },
        5: {
            "throttle": 0.0,
            "steer": 0.0,
            "brake": 0.5,
            "reverse": False,
            "name": "Brake-Light"
        }
    }

def get_action_space_size(allow_reverse: bool = True, allow_steering: bool = True):
    """
    Get the size of the action space based on enabled capabilities.
    
    Args:
        allow_reverse: Whether reverse actions are allowed
        allow_steering: Whether steering actions are allowed
        
    Returns:
        Integer size of the action space
    """
    if not allow_steering:
        return len(_get_no_steering_action_map())
    elif not allow_reverse:
        return len(config.DISCRETE_ACTION_MAP_NO_REVERSE)
    else:
        return len(config.DISCRETE_ACTION_MAP) 