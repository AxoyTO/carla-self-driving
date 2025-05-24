import sys
import os
import glob
import logging # For logging messages from this module

logger = logging.getLogger(__name__)

def add_carla_to_sys_path(carla_root_path: str):
    """
    Adds the CARLA PythonAPI to sys.path.
    Args:
        carla_root_path (str): The root directory of the CARLA installation. 
                                (e.g., "/opt/carla-simulator")
    """
    if not os.path.isdir(carla_root_path):
        logger.error(f"CARLA_ROOT path does not exist: {carla_root_path}")
        logger.error("Please ensure CARLA is installed and CARLA_ROOT is set correctly.")
        # Optionally, raise an error or exit if CARLA path is critical
        # raise FileNotFoundError(f"CARLA_ROOT path not found: {carla_root_path}")
        return

    carla_api_path = os.path.join(carla_root_path, "PythonAPI")
    if not os.path.isdir(carla_api_path):
        logger.error(f"CARLA PythonAPI directory not found at: {carla_api_path}")
        return

    # 1. Add .egg file for the core CARLA library
    # Prioritize Python 3.x eggs if multiple are found
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    specific_py_version_egg_pattern = os.path.join(carla_api_path, f"carla/dist/carla-*-py{py_major}.{py_minor}-*.egg")
    generic_py3_egg_pattern = os.path.join(carla_api_path, "carla/dist/carla-*-py3*.egg")
    any_egg_pattern = os.path.join(carla_api_path, "carla/dist/carla-*.egg")

    egg_to_add = None
    specific_eggs = glob.glob(specific_py_version_egg_pattern)
    if specific_eggs:
        egg_to_add = specific_eggs[0] # Take the first match for the specific version
    else:
        generic_py3_eggs = glob.glob(generic_py3_egg_pattern)
        if generic_py3_eggs:
            egg_to_add = generic_py3_eggs[0] # Fallback to any Python 3 egg
        else:
            all_eggs = glob.glob(any_egg_pattern)
            if all_eggs:
                egg_to_add = all_eggs[0] # Fallback to any egg found
    
    if egg_to_add:
        if egg_to_add not in sys.path:
            sys.path.append(egg_to_add)
            logger.info(f"Added CARLA egg to sys.path: {egg_to_add}")
        else:
            logger.debug(f"CARLA egg already in sys.path: {egg_to_add}")
    else:
        logger.warning(f"No CARLA .egg file found in {os.path.join(carla_api_path, 'carla/dist/')}. Core 'carla' module might not be loadable.")

    # 2. Add the directory containing the 'agents' module (typically PythonAPI/carla/)
    # This allows `from agents.navigation...` if 'agents' is a folder there.
    carla_agents_parent_dir = os.path.join(carla_api_path, "carla")
    if os.path.isdir(os.path.join(carla_agents_parent_dir, "agents")):
        if carla_agents_parent_dir not in sys.path:
            sys.path.insert(0, carla_agents_parent_dir) # Insert at high priority
            logger.info(f"Added CARLA agents parent dir to sys.path: {carla_agents_parent_dir}")
        else:
            logger.debug(f"CARLA agents parent dir already in sys.path: {carla_agents_parent_dir}")
    else:
        # Fallback: if 'agents' is directly under PythonAPI (less common for 0.9.15 but possible in some setups)
        if os.path.isdir(os.path.join(carla_api_path, "agents")):
            if carla_api_path not in sys.path:
                sys.path.insert(0, carla_api_path)
                logger.info(f"Added CARLA PythonAPI (for agents fallback) to sys.path: {carla_api_path}")
            else:
                logger.debug(f"CARLA PythonAPI (for agents fallback) already in sys.path: {carla_api_path}")
        else:
            logger.warning(f"CARLA 'agents' directory not found in expected locations ({carla_agents_parent_dir} or {carla_api_path}). Imports like 'from agents...' might fail.")

if __name__ == '__main__':
    # Example of how to use it if this script were run directly (for testing)
    # In the main application, config.CARLA_ROOT would be passed from main.py
    print("Testing carla_loader.py...")
    # Configure a basic logger for the test
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test with a common default path or a path from an environment variable if set
    test_carla_root = os.environ.get("CARLA_ROOT", "/opt/carla-simulator") 
    print(f"Using CARLA_ROOT for test: {test_carla_root}")
    add_carla_to_sys_path(test_carla_root)
    print("Current sys.path:")
    for p in sys.path:
        print(f"  {p}")
    
    try:
        import carla
        print(f"Successfully imported 'carla' version: {carla.VERSION}")
        # Try importing an agent utility if path was set correctly
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        print("Successfully imported GlobalRoutePlanner from agents.navigation")
    except ImportError as e:
        print(f"Failed to import CARLA or GRP after path modification: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during test import: {e}") 