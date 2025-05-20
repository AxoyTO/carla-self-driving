import pygame
import numpy as np
import carla # For type hinting carla.Image
from .hud import HUD # Import the new HUD class
import weakref
from collections import OrderedDict
import logging # Import logging

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

    def add_notification(self, text: str, duration_seconds: float = 3.0, color=None):
        if self.hud and self.hud_visible: # Only add if HUD exists and is visible
            self.hud.add_notification(text, duration_seconds, color=color)

    def render(self, carla_image: carla.Image, debug_info_from_env: dict):
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

        # Render main camera image
        if carla_image is not None:
            img_bgra = np.array(carla_image.raw_data).reshape(
                (carla_image.height, carla_image.width, 4)
            )
            img_rgb = img_bgra[:, :, :3][:, :, ::-1]  
            surface = pygame.surfarray.make_surface(img_rgb)
            surface = pygame.transform.rotate(surface, -90) 
            scaled_surface = pygame.transform.scale(surface, current_display_size)
            self.display_surface.blit(scaled_surface, (0, 0))
        else:
            self.display_surface.fill((30, 30, 30)) 

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