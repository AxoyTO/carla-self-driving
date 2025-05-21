import pygame
import numpy as np
import carla # For type hinting carla.Image and carla.ColorConverter
from .hud import HUD # Import the new HUD class
import weakref
from collections import OrderedDict
import logging # Import logging
import math # For sin/cos in RADAR, and general math

logger = logging.getLogger(__name__) # Logger for this module

class PygameVisualizer:
    def __init__(self, window_width, window_height, caption="CARLA RL Agent View", carla_env_ref=None):
        pygame.init()
        self.display_surface = pygame.display.set_mode(
            (window_width, window_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        pygame.display.set_caption(caption)
        
        # Font settings to be passed to HUD
        # User previously set font_size to 22 in the file for PygameVisualizer
        self.base_font_size = 28 
        self.fallback_font_size = self.base_font_size + 4 # e.g., 26
        self.font_preferences = ['Ubuntu Mono', 'Consolas', 'Lucida Console', 'monospace', 'mono', 'courier', None]

        self.hud = HUD(
            font_preferences=self.font_preferences,
            base_font_size=self.base_font_size,
            fallback_font_size=self.fallback_font_size
        )
            
        self.clock = pygame.time.Clock()
        self.is_active = True
        self.hud_visible = True 
        self.show_sensor_info_hud = False # Default off
        self.carla_env_ref = carla_env_ref # Weak reference or direct reference to CarlaEnv
        self.logger = logger # Assign module-level logger to instance

        # View cycling for Pygame display
        self.view_source_keys = [
            'spectator_camera',   # RENAMED - The default chase cam
            'rgb_camera',         # Agent's main RGB
            'depth_camera',
            'semantic_camera',
            'lidar',            # Added LIDAR view
            # 'radar'           # RADAR can be added later
        ]
        self.current_view_idx = 0

    def get_current_view_key(self) -> str:
        return self.view_source_keys[self.current_view_idx]

    def add_notification(self, text: str, duration_seconds: float = 3.0, color=None):
        if self.hud and self.hud_visible: # Only add if HUD exists and is visible
            self.hud.add_notification(text, duration_seconds, color=color)

    def _create_lidar_surface(self, lidar_data: carla.LidarMeasurement, surface_size, actual_lidar_sensor_range: float) -> pygame.Surface:
        lidar_surface = pygame.Surface(surface_size)
        lidar_surface.fill((5,5,15)) # Very dark blue background

        if lidar_data is None or len(lidar_data) == 0:
            return lidar_surface
        
        points_buffer = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        if points_buffer.shape[0] == 0 or points_buffer.shape[0] % 4 != 0:
            return lidar_surface
        points = points_buffer.reshape((int(points_buffer.shape[0] / 4), 4))

        # Max visualization range based on sensor's actual range, capped for sanity
        vis_display_range = actual_lidar_sensor_range if 0 < actual_lidar_sensor_range <= 100 else 50.0

        # Filter points by XY plane distance (circular)
        distances_xy = np.sqrt(points[:,0]**2 + points[:,1]**2)
        indices = distances_xy < vis_display_range
        points_to_process = points[indices]

        if points_to_process.shape[0] == 0:
            return lidar_surface

        # --- Aggressive Sub-sampling for FPS ---
        max_lidar_points_to_draw = 1500  # Drastically reduced for FPS test
        sample_rate = 1
        if points_to_process.shape[0] > max_lidar_points_to_draw:
            sample_rate = int(points_to_process.shape[0] / max_lidar_points_to_draw) or 1
        
        points_to_draw = points_to_process[::sample_rate]
        # self.logger.debug(f"LIDAR Vis: Original in range: {points_to_process.shape[0]}, Sampled to: {points_to_draw.shape[0]}")

        if points_to_draw.shape[0] == 0:
            return lidar_surface

        lidar_x = points_to_draw[:,0] # Forward
        lidar_y = points_to_draw[:,1] # Left
        # lidar_z = points_to_draw[:,2] # Up
        intensities = points_to_draw[:,3] # Intensity

        # Autoscaling based on the sampled points
        y_min_world = 0 
        y_max_world = vis_display_range 
        x_min_world = np.min(lidar_y) 
        x_max_world = np.max(lidar_y) 
        if x_max_world == x_min_world: x_max_world += 1.0; x_min_world -=1.0

        padding_px = 10 
        drawable_width = surface_size[0] - 2 * padding_px
        drawable_height = surface_size[1] - 2 * padding_px
        if drawable_width <= 0 or drawable_height <=0: return lidar_surface

        scale_x = drawable_width / (x_max_world - x_min_world + 1e-6) 
        scale_y = drawable_height / (y_max_world - y_min_world + 1e-6) 
        
        origin_x_screen = surface_size[0] / 2
        origin_y_screen = surface_size[1] - padding_px 

        drawn_points_count = 0
        for i in range(len(lidar_x)):
            lx, ly, intensity_val = lidar_x[i], lidar_y[i], intensities[i]
            screen_x = int(origin_x_screen - ly * scale_x) 
            screen_y = int(origin_y_screen - lx * scale_y)

            if 0 <= screen_x < surface_size[0] and 0 <= screen_y < surface_size[1]:
                # Color by intensity (simple yellow/orange mapping)
                norm_intensity = np.clip(intensity_val, 0.0, 1.0) # Assuming intensity is 0-1
                r_col = int(norm_intensity * 255)            # Full red component based on intensity
                g_col = int(norm_intensity * 150 + 50)     # Greenish-yellow component
                b_col = int(norm_intensity * 50)             # Low blue
                color = (np.clip(r_col,0,255), np.clip(g_col,0,255), np.clip(b_col,0,255))
                
                pygame.draw.circle(lidar_surface, color, (screen_x, screen_y), 1) 
                drawn_points_count +=1
        return lidar_surface

    def render(self, raw_sensor_data_item: any, current_view_key: str, debug_info_from_env: dict, lidar_sensor_range_from_env: float = 0.0):
        if not self.is_active:
            return False

        current_display_size = self.display_surface.get_size()
        self.clock.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_active = False
                return False
            elif event.type == pygame.VIDEORESIZE:
                self.display_surface = pygame.display.set_mode(
                    event.size, pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
                )
                current_display_size = event.size 
                self.logger.info(f"Pygame window resized to: {event.size}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    self.hud_visible = not self.hud_visible
                elif event.key == pygame.K_s: # Toggle for sensor info
                    self.show_sensor_info_hud = not self.show_sensor_info_hud
                elif event.key == pygame.K_c: # Cycle camera view
                    self.current_view_idx = (self.current_view_idx + 1) % len(self.view_source_keys)
                    new_view_name = self.get_current_view_key()
                    self.logger.info(f"Switched Pygame display to: {new_view_name}")

        # Process and render the selected sensor data
        surface_to_display = None
        if raw_sensor_data_item is not None:
            if current_view_key == 'lidar':
                surface_to_display = self._create_lidar_surface(raw_sensor_data_item, current_display_size, lidar_sensor_range_from_env)
            elif isinstance(raw_sensor_data_item, carla.Image):
                processed_image_for_display = raw_sensor_data_item
                if current_view_key == 'depth_camera':
                    processed_image_for_display.convert(carla.ColorConverter.LogarithmicDepth)
                elif current_view_key == 'semantic_camera':
                    processed_image_for_display.convert(carla.ColorConverter.CityScapesPalette)
                
                img_bgra = np.array(processed_image_for_display.raw_data).reshape(
                    (processed_image_for_display.height, processed_image_for_display.width, 4))
                img_rgb_h_w_c = img_bgra[:, :, :3][:, :, ::-1]
                temp_surface = pygame.surfarray.make_surface(img_rgb_h_w_c)
                if current_view_key == 'spectator_camera': 
                    temp_surface = pygame.transform.rotate(temp_surface, -90)
                elif current_view_key in ['rgb_camera', 'depth_camera', 'semantic_camera']:
                    temp_surface = pygame.transform.rotate(temp_surface, -90) # Apply -90 to all non-spectator cameras too based on last working config
                surface_to_display = temp_surface # This surface is small (sensor_res)
            # else: other sensor types not yet handled for direct full-screen viz

        if surface_to_display:
            # If it's an image sensor, it needs scaling. LIDAR surface is already screen_size.
            if current_view_key != 'lidar':
                scaled_surface = pygame.transform.scale(surface_to_display, current_display_size)
                self.display_surface.blit(scaled_surface, (0,0))
            else:
                self.display_surface.blit(surface_to_display, (0,0))
        else:
            self.display_surface.fill((30, 30, 30)) # Dark grey if no image (e.g., sensor not ready)

        # Render the main HUD (left panel) if visible
        if self.hud_visible:
            # Pass only the environment debug info to the main HUD render method
            self.hud.render(self.display_surface, self.clock, debug_info_from_env)

        # Render the sensor panel HUD (right panel) if toggled
        if self.show_sensor_info_hud:
            sensor_summary_data = None
            if self.carla_env_ref:
                env_instance = self.carla_env_ref() if isinstance(self.carla_env_ref, weakref.ref) else self.carla_env_ref
                if env_instance:
                    sensor_summary_data = env_instance.get_sensor_summary()
            
            if sensor_summary_data:
                self.hud.render_sensor_panel(self.display_surface, sensor_summary_data)
            else: # Fallback if data couldn't be fetched
                fallback_sensor_info = OrderedDict()
                fallback_sensor_info["Sensor Info"] = "(Unavailable)"
                self.hud.render_sensor_panel(self.display_surface, fallback_sensor_info)

        pygame.display.flip()
        return True

    def close(self):
        if pygame.get_init(): 
            pygame.quit()
        self.is_active = False
        # print("PygameVisualizer closed.") # Optional: log close 