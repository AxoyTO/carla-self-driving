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

def get_vehicle_control_from_discrete_action(
    action_index: int, 
    allow_reverse: bool, 
    action_map: Dict[int, Dict[str, Any]] = config.DISCRETE_ACTION_MAP, 
    no_reverse_action_map: Dict[int, Dict[str, Any]] = config.DISCRETE_ACTION_MAP_NO_REVERSE
) -> Tuple[carla.VehicleControl, str, int]: # Returns control, action_name, effective_action_index
    """ 
    Converts a discrete action index into a carla.VehicleControl object and action name.

    Args:
        action_index: The integer action selected by the agent.
        allow_reverse: Boolean flag from the current curriculum phase.
        action_map: The primary dictionary mapping action indices to control parameters.
        no_reverse_action_map: The alternative map for when reverse is disallowed.

    Returns:
        A tuple containing:
            - carla.VehicleControl object.
            - string: Name of the action for logging/debug.
            - int: The effective action index (could be remapped if unknown).
    """
    control = carla.VehicleControl()  # Default: all zeros
    action_to_use = action_index
    
    current_map = action_map if allow_reverse else no_reverse_action_map
    
    action_params = current_map.get(action_index)
    
    if action_params:
        control.throttle = float(action_params.get("throttle", 0.0))
        control.steer = float(action_params.get("steer", 0.0))
        control.brake = float(action_params.get("brake", 0.0))
        control.reverse = bool(action_params.get("reverse", False))
        action_name = action_params.get("name", f"Action_{action_index}")
        
        # Log if an intended reverse action was remapped
        if not allow_reverse and config.DISCRETE_ACTION_MAP.get(action_index, {}).get("reverse") == True and action_index != 5:
             # This case should ideally not happen if no_reverse_action_map correctly remaps all reverse actions
             logger.debug(f"Action {action_index} ({config.DISCRETE_ACTION_MAP.get(action_index, {}).get('name')}) was intended as reverse but remapped by no_reverse_action_map to: {action_name}")
        elif not allow_reverse and action_index == 5 and current_map[5]["name"] != config.DISCRETE_ACTION_MAP[5]["name"]:
             logger.debug(f"Action 5 (Reverse) remapped to '{action_name}' due to allow_reverse=False.")

    else:
        logger.warning(f"Unknown discrete action index: {action_index}. Applying default (Brake). Remapping to action 3.")
        # Fallback to a defined safe action (e.g., Brake - action 3 from the standard map)
        fallback_action_params = config.DISCRETE_ACTION_MAP.get(3, {"throttle": 0.0, "steer": 0.0, "brake": 1.0, "reverse": False, "name": "Brake (Fallback)"})
        control.throttle = float(fallback_action_params.get("throttle", 0.0))
        control.steer = float(fallback_action_params.get("steer", 0.0))
        control.brake = float(fallback_action_params.get("brake", 0.0))
        control.reverse = bool(fallback_action_params.get("reverse", False))
        action_name = fallback_action_params.get("name")
        action_to_use = 3 # The effective action index is now 3
        
    return control, action_name, action_to_use 