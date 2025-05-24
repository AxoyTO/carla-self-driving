import carla
import random
import logging
from typing import Optional, Tuple

class VehicleManager:
    """Manages the ego vehicle's lifecycle and configuration."""

    def __init__(self, world: carla.World, logger: Optional[logging.Logger] = None):
        """Initialize the VehicleManager.
        Args:
            world: CARLA world instance.
            logger: Optional logger instance.
        """
        self.world = world
        self.logger = logger if logger else logging.getLogger(__name__ + ".VehicleManager")
        self.vehicle: Optional[carla.Actor] = None
        self.initial_transform: Optional[carla.Transform] = None
        
        # Default blueprint, can be customized
        self.vehicle_blueprint_name = 'vehicle.tesla.model3'

    def _setup_vehicle_blueprint(self) -> Optional[carla.ActorBlueprint]:
        """Sets up the vehicle blueprint with appropriate attributes."""
        try:
            blueprint = self.world.get_blueprint_library().find(self.vehicle_blueprint_name)
            blueprint.set_attribute('role_name', 'hero')
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', '255,255,255') # White color for ego
            return blueprint
        except Exception as e:
            self.logger.error(f"Failed to find or setup blueprint {self.vehicle_blueprint_name}: {e}")
            return None

    def _apply_vehicle_physics(self):
        """Sets up vehicle physics parameters."""
        if not self.vehicle:
            self.logger.warning("Cannot apply physics, vehicle is None.")
            return
            
        physics_control = self.vehicle.get_physics_control()
        # Example: Basic physics, can be expanded from CarlaEnv's original settings
        physics_control.mass = 1500.0
        # ... (add more physics settings as needed, e.g., wheel friction, damping)
        self.vehicle.apply_physics_control(physics_control)
        self.logger.debug(f"Applied physics to vehicle {self.vehicle.id}")

    def spawn_vehicle(self, spawn_transform: carla.Transform) -> Optional[carla.Actor]:
        """Spawns and configures the vehicle at the given transform.
        Args:
            spawn_transform: The transform where the vehicle should be spawned.
        Returns:
            Spawned vehicle actor if successful, None otherwise.
        """
        self.destroy_vehicle() # Ensure any previous vehicle is gone

        blueprint = self._setup_vehicle_blueprint()
        if not blueprint:
            return None

        try:
            self.vehicle = self.world.try_spawn_actor(blueprint, spawn_transform)
            if self.vehicle:
                self.initial_transform = spawn_transform # Store the intended spawn transform
                self._apply_vehicle_physics()
                self.vehicle.set_autopilot(False) # Ego vehicle typically doesn't use TM autopilot
                self.logger.debug(f"Spawned vehicle {self.vehicle.id} ({self.vehicle.type_id}) at {spawn_transform.location}")
                
                # Wait for physics to settle after spawning
                if self.world.get_settings().synchronous_mode:
                    for _ in range(5): # Tick a few times
                        self.world.tick()
                return self.vehicle
            else:
                self.logger.error(f"Failed to spawn vehicle {self.vehicle_blueprint_name} at {spawn_transform.location}")
                return None
        except Exception as e:
            self.logger.error(f"Exception during vehicle spawn: {e}", exc_info=True)
            if self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy() # Cleanup if spawn failed mid-way
            self.vehicle = None
            return None

    def reset_vehicle_state(self):
        """Resets the vehicle to its initial spawn transform and zero velocity."""
        if self.vehicle and self.vehicle.is_alive and self.initial_transform:
            self.vehicle.set_transform(self.initial_transform)
            self.vehicle.set_target_velocity(carla.Vector3D(0,0,0))
            self.vehicle.set_target_angular_velocity(carla.Vector3D(0,0,0))
            if self.world.get_settings().synchronous_mode:
                for _ in range(5): # Tick a few times to ensure state is applied
                    self.world.tick()
            self.logger.debug(f"Reset vehicle {self.vehicle.id} to transform {self.initial_transform.location}")
        elif not self.vehicle:
            self.logger.warning("Cannot reset, vehicle is None.")
        elif not self.initial_transform:
            self.logger.warning("Cannot reset, initial_transform is None.")

    def destroy_vehicle(self):
        """Destroys the current vehicle if it exists."""
        if self.vehicle and self.vehicle.is_alive:
            actor_id = self.vehicle.id
            self.vehicle.destroy()
            if self.world.get_settings().synchronous_mode:
                self.world.tick() # Ensure destruction is processed
            self.logger.debug(f"Destroyed vehicle {actor_id}")
        self.vehicle = None
        self.initial_transform = None

    def get_vehicle(self) -> Optional[carla.Actor]:
        """Returns the current vehicle actor."""
        return self.vehicle

    def is_alive(self) -> bool:
        """Checks if the vehicle exists and is alive."""
        return self.vehicle is not None and self.vehicle.is_alive 