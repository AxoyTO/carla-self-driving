# Handles Collision sensor setup and data processing 

import carla
import numpy as np
import weakref
import logging

# Configure a logger for this handler
logger = logging.getLogger(__name__)

# Note: Collision data is not part of the observation space by default in this setup,
# it directly updates CarlaEnv.collision_info.

def setup_collision_sensor(world, vehicle, carla_env_weak_ref, transform=None):
    """Spawns and configures a collision sensor."""
    blueprint_library = world.get_blueprint_library()
    collision_bp = blueprint_library.find('sensor.other.collision')
    # collision_bp.set_attribute('ignore_actor', '0') # This attribute might not exist or be needed.
                                                     # By default, it detects all collisions.

    if transform is None:
        transform = carla.Transform(carla.Location(x=0.0, z=0.0)) # Default at vehicle origin

    collision_sensor = world.spawn_actor(collision_bp, transform, attach_to=vehicle)
    logger.debug(f"Spawned Collision Sensor: {collision_sensor.id} at {transform}")

    def on_collision_event(event): # Renamed from callback for clarity
        env_instance = carla_env_weak_ref()
        if not env_instance:
            return

        # Update collision_info in the CarlaEnv instance
        env_instance.collision_info['count'] += 1
        impulse = event.normal_impulse
        intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        env_instance.collision_info['last_intensity'] = intensity
        
        other_actor = event.other_actor
        env_instance.collision_info['last_other_actor'] = other_actor.type_id # Store type_id
        # env_instance.collision_info['last_other_actor_id'] = other_actor.id # Could store actor ID too

        # Log the collision event
        logger.debug(
            f"Collision detected for vehicle {vehicle.id}: "
            f"Other Actor ID: {other_actor.id}, Type: {other_actor.type_id}, "
            f"Impulse: ({impulse.x:.2f}, {impulse.y:.2f}, {impulse.z:.2f}), Intensity: {intensity:.2f}"
        )
        
        # Example of how you might want to use this in the main env for done check:
        # env_instance.collided_this_step = True 
        # (Requires adding `collided_this_step` attribute to CarlaEnv and resetting it each step)

    collision_sensor.listen(on_collision_event)
    return collision_sensor

# Note: No get_observation_space or process_data function here as collision data
# is primarily used for reward/done checks and updates CarlaEnv.collision_info directly.
# If collision features were to be part of the observation, those functions would be added. 