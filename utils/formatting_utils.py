import math

def format_time(seconds: float) -> str:
    """Formats seconds into hh:mm:ss string."""
    if seconds is None or not isinstance(seconds, (int, float)) or seconds < 0:
        return "00:00:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def format_vehicle_model_name(type_id_str: str) -> str:
    """Formats a CARLA vehicle type_id string into a more readable model name."""
    if not type_id_str:
        return "Unknown Vehicle"
    name_parts = type_id_str.split('.')
    if name_parts[0] == 'vehicle':
        name_parts = name_parts[1:]
    
    formatted_name = ' '.join([part.capitalize() for part in name_parts])
    return formatted_name

def format_compass_direction(raw_yaw_deg: float) -> str:
    """Formats a yaw in degrees (CARLA convention) to a compass string like '180.0° S'."""
    standard_compass_deg = (90 - raw_yaw_deg + 360) % 360
    cardinal_directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"]
    cardinal_idx = round(standard_compass_deg / 45.0) % 8 
    cardinal_dir_str = cardinal_directions[cardinal_idx]
    return f"{standard_compass_deg:.1f}° {cardinal_dir_str}" 