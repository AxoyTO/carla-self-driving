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
        self.walker_controllers: List[carla.Actor] = []  # Changed to Actor type for clarity

        self.vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        # Filter out bicycles and motorcycles for better traffic behavior
        self.vehicle_blueprints = [bp for bp in self.vehicle_blueprints if int(bp.get_attribute('number_of_wheels')) >= 4]
        self.walker_blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
        
        # Initialize CARLA's built-in traffic manager
        self.carla_tm = self.client.get_trafficmanager() # Get the TM instance
        self.tm_port = self.carla_tm.get_port() # Get the actual port the TM instance is running on
        
        # Set global traffic manager parameters
        self.carla_tm.set_global_distance_to_leading_vehicle(2.5)  # Keep reasonable distance between vehicles
        self.carla_tm.set_synchronous_mode(True)  # Match our synchronous environment

    def spawn_npcs(self, traffic_config: Dict):
        num_vehicles = traffic_config.get("num_vehicles", 0)
        num_walkers = traffic_config.get("num_walkers", 0)
        spawn_type = traffic_config.get("type", "none")

        if spawn_type == "none" or (num_vehicles == 0 and num_walkers == 0):
            self.logger.debug("No NPCs to spawn based on traffic_config.")
            return

        self.logger.debug(f"Spawning NPCs: {num_vehicles} vehicles, {num_walkers} walkers (type: {spawn_type})")
        
        # First, let's spawn vehicles
        if num_vehicles > 0:
            self._spawn_vehicles(num_vehicles, spawn_type)
            
        # Then, spawn walkers
        if num_walkers > 0:
            self._spawn_walkers(num_walkers, spawn_type)
        
        self.logger.info(f"Finished NPC spawning: {len(self.npc_vehicles)} vehicles, {len(self.npc_walkers)} walkers.")
        
        # Tick the world to properly initialize all actors
        if self.world.get_settings().synchronous_mode:
            self.world.tick()

    def _spawn_vehicles(self, num_vehicles: int, spawn_type: str):
        """Helper method to spawn vehicles with proper behavior"""
        spawn_points = self.map.get_spawn_points()
        
        if not spawn_points:
            self.logger.error("No spawn points found for vehicles!")
            return
            
        random.shuffle(spawn_points)
        spawn_actor_batch = []
        
        # Generate spawn commands for batch processing
        for i in range(num_vehicles):
            if i >= len(spawn_points):
                self.logger.warning(f"Not enough spawn points for all vehicles. Spawning {i} instead of {num_vehicles}.")
                break
                
            blueprint = random.choice(self.vehicle_blueprints)
            
            # Set random color
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
                
            # Try not to spawn vehicles as the police or emergency vehicles
            if blueprint.has_attribute('role_name'):
                blueprint.set_attribute('role_name', 'autopilot')
                
            # Set as autopilot if dynamic
            autopilot = spawn_type == "dynamic"
            
            spawn_point = spawn_points[i]
            spawn_cmd = carla.command.SpawnActor(blueprint, spawn_point)
            
            # If dynamic, we'll command the traffic manager to take over
            if autopilot:
                spawn_cmd = spawn_cmd.then(carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port))
                
            spawn_actor_batch.append(spawn_cmd)
        
        # Execute the batch spawn
        results = self.client.apply_batch_sync(spawn_actor_batch, True)
        vehicle_ids = []
        
        for i, result in enumerate(results):
            if result.error:
                self.logger.warning(f"Failed to spawn vehicle: {result.error}")
            else:
                vehicle_ids.append(result.actor_id)
        
        # Get the spawned vehicle actors
        vehicles = self.world.get_actors(vehicle_ids)
        self.npc_vehicles.extend(vehicles)
        
        # Configure traffic manager behavior for all vehicles
        if spawn_type == "dynamic":
            self.logger.info(f"Setting up {len(vehicles)} vehicles with autopilot")
            
            for vehicle in vehicles:
                # Randomize driving behavior slightly for each vehicle
                # Lower percentage = faster driving speed
                speed_factor = random.uniform(80.0, 120.0)
                
                # Adjust for time scale
                if self.time_scale != 1.0:
                    speed_factor = speed_factor / self.time_scale  # Faster with higher time_scale
                
                # Set vehicle behavior parameters
                self.carla_tm.vehicle_percentage_speed_difference(vehicle, speed_factor)
                self.carla_tm.update_vehicle_lights(vehicle, True)  # Use vehicle lights
                
                # Randomize lane changing behavior
                lane_change = random.choice([carla.LaneChange.NONE, carla.LaneChange.RIGHT, carla.LaneChange.LEFT, carla.LaneChange.BOTH])
                self.carla_tm.set_path_finding_algorithm(vehicle, True)  # Use path finding
                self.carla_tm.set_lane_change(vehicle, lane_change)
                
                # Some vehicles should respect traffic lights, others might not
                self.carla_tm.ignore_lights_percentage(vehicle, random.randint(0, 20))
                
                # Some vehicles keep more distance than others
                self.carla_tm.distance_to_leading_vehicle(vehicle, random.uniform(1.0, 4.0))
                
                self.logger.debug(f"Vehicle {vehicle.id} set up with speed factor {speed_factor:.1f}%")

    def _spawn_walkers(self, num_walkers: int, spawn_type: str):
        """Helper method to spawn walker pedestrians with proper behavior"""
        # First, we'll spawn all the walker actors
        spawn_points = []
        
        # Try to find valid spawn locations for walkers
        for i in range(num_walkers * 2):  # Try more locations than needed
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
                
        if not spawn_points:
            self.logger.error("Could not find any walker spawn points!")
            return
            
        # Limit to the actual number we want
        spawn_points = spawn_points[:num_walkers]
        
        # Create walker blueprints
        walker_batch = []
        walker_speed = []
        
        for i, spawn_point in enumerate(spawn_points):
            walker_bp = random.choice(self.walker_blueprints)
            
            # Set not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
                
            # Set the max speed
            if walker_bp.has_attribute('speed'):
                # Walker base speed - adjust based on walker type
                base_speed = float(walker_bp.get_attribute('speed').recommended_values[1])
                walker_speed.append(base_speed * self.time_scale)  # Scale with time_scale
            else:
                walker_speed.append(1.4 * self.time_scale)  # Default speed
                
            # Create spawn command
            walker_batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
            
        # Apply the batch and get walker actors
        results = self.client.apply_batch_sync(walker_batch, True)
        walker_ids = []
        
        for i, result in enumerate(results):
            if result.error:
                self.logger.warning(f"Failed to spawn walker: {result.error}")
            else:
                walker_ids.append(result.actor_id)
                
        # Get the actual walker actors
        walkers = self.world.get_actors(walker_ids)
        self.npc_walkers.extend(walkers)
        
        # If not dynamic, we're done - static walkers don't need controllers
        if spawn_type != "dynamic" or len(walkers) == 0:
            return
            
        # Now spawn the walker controllers for dynamic walkers
        controller_batch = []
        controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        
        for i, walker in enumerate(walkers):
            controller_batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), walker))
            
        # Apply the batch and get controller actors
        results = self.client.apply_batch_sync(controller_batch, True)
        controller_ids = []
        
        for i, result in enumerate(results):
            if result.error:
                self.logger.warning(f"Failed to spawn walker controller: {result.error}")
            else:
                controller_ids.append(result.actor_id)
                
        # Get the actual controller actors
        controllers = self.world.get_actors(controller_ids)
        self.walker_controllers.extend(controllers)
        
        # Wait for a world tick for proper initialization
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
            
        # Set walker controller behavior
        for i, controller in enumerate(controllers):
            # Set the walker to random walk
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            
            # Set the max speed
            if i < len(walker_speed):
                controller.set_max_speed(walker_speed[i])
                
        self.logger.info(f"Successfully spawned {len(controllers)} walker controllers for {len(walkers)} walkers")

    def destroy_npcs(self):
        self.logger.debug(f"Destroying {len(self.npc_vehicles)} vehicles, {len(self.npc_walkers)} walkers, and {len(self.walker_controllers)} controllers.")
        
        # First, stop all walker controllers
        for controller in self.walker_controllers:
            if controller and controller.is_alive:
                controller.stop()
                
        # Wait for a tick to let controllers fully stop
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
            
        # Create batch destroy commands for all actors
        destroy_actors = []
        
        # First add controllers
        for controller in self.walker_controllers:
            if controller and controller.is_alive:
                destroy_actors.append(controller)
                
        # Then add walkers
        for walker in self.npc_walkers:
            if walker and walker.is_alive:
                destroy_actors.append(walker)
                
        # Finally add vehicles
        for vehicle in self.npc_vehicles:
            if vehicle and vehicle.is_alive:
                destroy_actors.append(vehicle)
                
        # Batch destroy all actors
        if destroy_actors:
            destroy_commands = [carla.command.DestroyActor(actor) for actor in destroy_actors]
            
            try:
                self.client.apply_batch_sync(destroy_commands, True)
                self.logger.info(f"Successfully destroyed {len(destroy_commands)} NPC actors.")
            except Exception as e:
                self.logger.error(f"Error during NPC destruction: {e}", exc_info=True)
                
                # Fallback: try to destroy individually
                self.logger.info("Falling back to individual destruction...")
                for actor in destroy_actors:
                    try:
                        if actor and actor.is_alive:
                            actor.destroy()
                    except Exception as e_ind:
                        self.logger.error(f"Failed to destroy actor {actor.id}: {e_ind}")
        
        # Clear our lists
        self.npc_vehicles = []
        self.npc_walkers = []
        self.walker_controllers = []
        
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