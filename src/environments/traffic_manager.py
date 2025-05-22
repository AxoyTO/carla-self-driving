import carla
import random
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class TrafficManager:
    def __init__(self, client: carla.Client, world: carla.World, carla_env_ref: Optional[object] = None, log_level=logging.INFO, time_scale=1.0):
        self.client = client
        self.world = world
        self.map = self.world.get_map() # Get map once
        self.carla_env_ref = carla_env_ref # Weak reference to CarlaEnv for accessing ego vehicle, etc.
        self.logger = logging.getLogger(__name__ + ".TrafficManager") # Specific logger
        self.logger.setLevel(log_level)
        self.time_scale = time_scale

        self.npc_vehicles: List[carla.Actor] = []
        self.npc_walkers: List[carla.Actor] = []
        self.walker_controllers: List[carla.WalkerAIController] = []

        self.vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        self.walker_blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')

    def spawn_npcs(self, traffic_config: Dict):
        num_vehicles = traffic_config.get("num_vehicles", 0)
        num_walkers = traffic_config.get("num_walkers", 0)
        spawn_type = traffic_config.get("type", "none")

        if spawn_type == "none" or (num_vehicles == 0 and num_walkers == 0):
            self.logger.debug("No NPCs to spawn based on traffic_config.")
            return

        self.logger.debug(f"Spawning NPCs: {num_vehicles} vehicles, {num_walkers} walkers (type: {spawn_type})")
        
        available_spawn_points = list(self.map.get_spawn_points()) # Get a mutable list
        random.shuffle(available_spawn_points)

        # Spawn Vehicles
        for i in range(num_vehicles):
            if not available_spawn_points:
                self.logger.warning("No more available spawn points for vehicles.")
                break
            
            blueprint = random.choice(self.vehicle_blueprints)
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', "{},{},{}".format(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
            if blueprint.has_attribute('driver_id'):
                blueprint.set_attribute('driver_id', str(random.randint(1000, 9999)))
            
            spawn_point = available_spawn_points.pop(0) # Take one spawn point
            
            vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
            if vehicle:
                self.npc_vehicles.append(vehicle)
                if spawn_type == "dynamic":
                    # Get CARLA's internal traffic manager
                    tm = self.client.get_trafficmanager(self.client.get_trafficmanager_port())
                    vehicle.set_autopilot(True, tm.get_port()) # Use TM port for autopilot
                    
                    # Adjust speeds based on time scale
                    if self.time_scale != 1.0:
                        # Set vehicle speed to scale with time_scale
                        # Lower percentage = faster vehicle (counter-intuitive in CARLA TrafficManager)
                        if self.time_scale > 1.0:
                            # Scale is greater than 1 - vehicles should move faster
                            speed_factor = 100.0 / self.time_scale  # Lower percentage = faster
                        else:
                            # Scale is less than 1 - vehicles should move slower
                            speed_factor = 100.0 * (1.0 / self.time_scale)  # Higher percentage = slower
                        
                        tm.vehicle_percentage_speed_difference(vehicle, speed_factor)
                        self.logger.debug(f"Vehicle {vehicle.id} speed factor set to {speed_factor}% to match time_scale {self.time_scale}")
                    
                # For "static", we just spawn them, they won't move.
                self.logger.debug(f"Spawned vehicle {vehicle.id} at {spawn_point.location} (type: {spawn_type})")
            else:
                self.logger.warning(f"Failed to spawn vehicle at {spawn_point.location}")
                available_spawn_points.append(spawn_point) # Add back if failed, try again later or for walkers

        # Spawn Walkers (Pedestrians)
        for i in range(num_walkers):
            if not available_spawn_points: # Check again, though walkers use random locations, not spawn points
                self.logger.warning("No spawn points left, cannot guarantee safe walker spawn near road.")
                # For walkers, we don't strictly need a vehicle spawn point.
                # We can pick a random location on the sidewalk.
                spawn_location = carla.Location() # Placeholder
                found_location = False
                for _ in range(10): # Try 10 times to find a random sidewalk location
                    spawn_location = self.world.get_random_location_from_navigation()
                    if spawn_location:
                        found_location = True
                        break
                if not found_location:
                    self.logger.warning("Could not find random navigation location for walker. Skipping walker.")
                    continue
            else:
                # Try to spawn near a road using remaining vehicle spawn points as a hint
                # This is a simple way to get them somewhat near drivable areas.
                # A more robust method would be to query sidewalk locations directly.
                hint_spawn_point = random.choice(available_spawn_points) if available_spawn_points else self.map.get_spawn_points()[0]
                spawn_location = self.world.get_random_location_from_navigation()
                if not spawn_location: # Fallback if random nav location fails
                    self.logger.warning(f"Failed to get random nav location, using hint: {hint_spawn_point.location}")
                    spawn_location = hint_spawn_point.location

            blueprint = random.choice(self.walker_blueprints)
            if blueprint.has_attribute('is_invincible'):
                blueprint.set_attribute('is_invincible', 'false')
            
            walker_actor = self.world.try_spawn_actor(blueprint, carla.Transform(spawn_location))

            if walker_actor:
                self.npc_walkers.append(walker_actor)
                if spawn_type == "dynamic":
                    controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                    ai_controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker_actor)
                    if ai_controller:
                        self.walker_controllers.append(ai_controller)
                        ai_controller.start()
                        ai_controller.go_to_location(self.world.get_random_location_from_navigation())
                        
                        # Adjust walker speed based on time_scale
                        base_speed = 1.0 + random.random()  # Random speed: 1.0 to 2.0 m/s
                        scaled_speed = base_speed * self.time_scale
                        ai_controller.set_max_speed(scaled_speed)
                        
                        self.logger.debug(f"Spawned walker {walker_actor.id} at {spawn_location} with speed {scaled_speed:.2f} m/s (base: {base_speed:.2f}, scale: {self.time_scale}x)")
                    else:
                        self.logger.warning(f"Failed to spawn AI controller for walker {walker_actor.id}. Walker will be static.")
                        walker_actor.destroy() # Clean up walker if controller fails
                        self.npc_walkers.pop() # Remove from list
                else:
                    self.logger.debug(f"Spawned static walker {walker_actor.id} at {spawn_location}")
            else:
                self.logger.warning(f"Failed to spawn walker at {spawn_location}")
        
        self.logger.info(f"Finished NPC spawning: {len(self.npc_vehicles)} vehicles, {len(self.npc_walkers)} walkers.")

    def destroy_npcs(self):
        self.logger.debug(f"Destroying {len(self.npc_vehicles)} vehicles and {len(self.npc_walkers)} walkers.")
        
        # Stop AI controllers for walkers first
        for controller in self.walker_controllers:
            if controller and controller.is_alive:
                controller.stop()
                # controller.destroy() # Controller is destroyed when walker is destroyed if attached
        self.walker_controllers = []

        # Create batches for destruction
        destroy_commands = []
        for actor_list in [self.npc_vehicles, self.npc_walkers]: # npc_walkers includes those whose controllers might have failed
            for actor in actor_list:
                if actor and actor.is_alive:
                    destroy_commands.append(carla.command.DestroyActor(actor))
        
        if destroy_commands:
            try:
                self.client.apply_batch_sync(destroy_commands, True) # Synchronous destruction
                self.logger.debug(f"Successfully applied batch destruction for {len(destroy_commands)} NPC actors.")
            except RuntimeError as e:
                self.logger.error(f"RuntimeError during NPC batch destruction: {e}", exc_info=True)
                # Fallback to individual destruction if batch fails
                self.logger.info("Falling back to individual NPC destruction...")
                for actor_list in [self.npc_vehicles, self.npc_walkers]:
                    for actor in actor_list:
                        if actor and actor.is_alive:
                            try:
                                actor.destroy()
                            except RuntimeError as e_ind:
                                self.logger.error(f"Failed to destroy individual NPC {actor.id}: {e_ind}")
        
        self.npc_vehicles = []
        self.npc_walkers = []
        self.logger.debug("NPC destruction complete.")

    def set_global_traffic_light_state(self, state: carla.TrafficLightState, duration_ms: Optional[int] = None):
        """
        Sets all traffic lights in the world to a specific state.
        Args:
            state (carla.TrafficLightState): The desired state (Red, Yellow, Green).
            duration_ms (Optional[int]): If provided, traffic lights will freeze in this state for this duration (in milliseconds).
                                          If None, they are set to the state and resume normal cycling if applicable.
        """
        all_tls = self.world.get_actors().filter('*.traffic_light')
        self.logger.info(f"Attempting to set {len(all_tls)} traffic lights to state: {state}")
        for tl in all_tls:
            if hasattr(tl, 'set_state'):
                tl.set_state(state)
                if duration_ms is not None:
                    tl.freeze(True) # Freeze to maintain state
                    # Unfreezing might need to be handled separately if a timed freeze is desired
                    # For now, this sets them and freezes them if duration_ms is given (implies freeze)
                    # A more complex system would use time.sleep or similar for timed freeze then unfreeze.
                else:
                    tl.freeze(False) # Ensure not frozen if no duration
            if hasattr(tl, 'set_red_time') and state == carla.TrafficLightState.Red:
                pass # Adjusting times is more complex, set_state is primary for now
        self.logger.info(f"Traffic light states set.")

    def unfreeze_all_traffic_lights(self):
        all_tls = self.world.get_actors().filter('*.traffic_light')
        self.logger.info(f"Unfreezing {len(all_tls)} traffic lights.")
        for tl in all_tls:
            if hasattr(tl, 'freeze'):
                tl.freeze(False)
        self.logger.info("All traffic lights unfrozen.")

# Example usage (for testing within this file, not for direct execution in the RL pipeline)
# if __name__ == '__main__':
#     try:
#         client = carla.Client('localhost', 2000)
#         client.set_timeout(10.0)
#         world = client.load_world('Town03')
#         # world = client.get_world()

#         # Ensure synchronous mode for stable testing if manipulating world
#         settings = world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = 0.05
#         world.apply_settings(settings)
#         world.tick() 

#         # traffic_manager = TrafficManager(client, world, log_level=logging.DEBUG)
#         # traffic_config_dynamic = {"num_vehicles": 5, "num_walkers": 5, "type": "dynamic"}
#         # traffic_config_static = {"num_vehicles": 10, "type": "static"}
#         # traffic_config_none = {"type": "none"}

#         # print("Spawning dynamic NPCs...")
#         # traffic_manager.spawn_npcs(traffic_config_dynamic)
#         # world.tick() # Allow them to spawn and settle
#         # for _ in range(100): # Let them move a bit
#         #     world.tick()
#         #     time.sleep(0.05)
#         # print("Destroying NPCs...")
#         # traffic_manager.destroy_npcs()
#         # world.tick()

#         # print("Spawning static NPCs...")
#         # traffic_manager.spawn_npcs(traffic_config_static)
#         # world.tick()
#         # time.sleep(2)
#         # print("Destroying NPCs...")
#         # traffic_manager.destroy_npcs()
#         # world.tick()

#         # print("Setting traffic lights to RED")
#         # traffic_manager.set_global_traffic_light_state(carla.TrafficLightState.Red, duration_ms=5000)
#         # world.tick()
#         # time.sleep(6) # Wait for more than duration
#         # print("Unfreezing traffic lights")
#         # traffic_manager.unfreeze_all_traffic_lights()
#         # world.tick()

#     except Exception as e:
#         print(f"Error in TrafficManager test: {e}")
#     finally:
#         if 'world' in locals() and world is not None:
#             settings = world.get_settings()
#             if settings.synchronous_mode:
#                 settings.synchronous_mode = False
#                 settings.fixed_delta_seconds = None
#                 world.apply_settings(settings)
#                 world.tick()
#         print("Test finished.") 