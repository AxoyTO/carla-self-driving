import pygame
import numpy as np
import carla # For type hinting carla.Image and carla.ColorConverter
from .hud import HUD # Import the new HUD class
import weakref
from collections import OrderedDict
import logging # Import logging
import math # For sin/cos in RADAR, and general math
from typing import Tuple, Optional
import random # For subsampling in semantic lidar view
import pygame.gfxdraw  # For anti-aliased circles
import time # For time-based operations
import os
import sys

# Add project root to Python path for local development
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from utils/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import app.config as config

logger = logging.getLogger(__name__) # Logger for this module

# Semantic Label Colors (matches CARLA's default, normalized for Pygame 0-255)
PYGAME_LABEL_COLORS = np.array([
    (0, 0, 0),       # 0 Unlabeled (often black or a distinct color)
    (70, 70, 70),    # 1 Building
    (100, 40, 40),   # 2 Fences
    (55, 90, 80),    # 3 Other
    (220, 20, 60),   # 4 Pedestrian
    (153, 153, 153), # 5 Pole
    (157, 234, 50),  # 6 RoadLines
    (128, 64, 128),  # 7 Road
    (244, 35, 232),  # 8 Sidewalk
    (107, 142, 35),  # 9 Vegetation
    (0, 0, 142),     # 10 Vehicle
    (102, 102, 156), # 11 Wall
    (220, 220, 0),   # 12 TrafficSign
    (70, 130, 180),  # 13 Sky (less relevant for LIDAR)
    (81, 0, 81),     # 14 Ground
    (150, 100, 100), # 15 Bridge
    (230, 150, 140), # 16 RailTrack
    (180, 165, 180), # 17 GuardRail
    (250, 170, 30),  # 18 TrafficLight
    (110, 190, 160), # 19 Static
    (170, 120, 50),  # 20 Dynamic
    (45, 60, 150),   # 21 Water
    (145, 170, 100), # 22 Terrain
    # CARLA docs mention up to 28 classes, ensure enough colors if more are used
    (128,128,0),(0,128,128),(128,0,128),(64,0,0),(0,64,0),(0,0,64) # Extra placeholders
], dtype=np.uint8)

class PygameVisualizer:
    def __init__(self, window_width, window_height, caption="CARLA RL Agent View", carla_env_ref=None, disable_sensor_views_flag=False):
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

        default_view_source_keys = [
            'spectator_camera',
            'display_rgb_camera',
            'display_left_rgb_camera',
            'display_right_rgb_camera',
            'display_rear_rgb_camera',
            'display_depth_camera',
            'display_semantic_camera',
            'lidar',            
            'semantic_lidar',
            'radar'               
        ]

        default_view_display_names = {
            'spectator_camera': "Spectator",
            'display_rgb_camera': "Front RGB Camera",
            'display_left_rgb_camera': "Left RGB Camera",
            'display_right_rgb_camera': "Right RGB Camera",
            'display_rear_rgb_camera': "Rear RGB Camera",
            'display_depth_camera': "Depth Camera",
            'display_semantic_camera': "Semantic Camera",
            'lidar': "LIDAR (Top-Down)",
            'semantic_lidar': "Semantic LIDAR (Top-Down)",
            'radar': "RADAR (Top-Down)"
        }

        if disable_sensor_views_flag:
            self.logger.info("Sensor views are DISABLED for Pygame display. Only Spectator view will be available.")
            self.view_source_keys = ['spectator_camera']
            self.view_display_names = {'spectator_camera': "Spectator"}
        else:
            self.view_source_keys = default_view_source_keys
            self.view_display_names = default_view_display_names

        self.current_view_idx = 0
        self.notifications = [] # List to store active notifications: [{"text": str, "color": tuple, "expire_time": float, "surface": pygame.Surface}]
        self.notification_font_size = 24
        try:
            self.notification_font = pygame.font.Font(None, self.notification_font_size)
        except pygame.error:
            self.logger.warning("Failed to load notification font. Falling back to default system font.")
            self.notification_font = pygame.font.Font(None, self.notification_font_size)

    def get_current_view_key(self) -> str:
        """Returns the raw key of the current view source."""
        return self.view_source_keys[self.current_view_idx]

    def get_current_view_display_name(self) -> str:
        """Returns the user-friendly display name for the current view."""
        raw_key = self.get_current_view_key()
        return self.view_display_names.get(raw_key, raw_key) # Fallback to raw key if no pretty name

    def add_notification(self, text: str, duration_seconds: float = 3.0, color: Tuple[int, int, int] = (255, 255, 0)):
        """Adds a notification message to be displayed on the HUD."""
        if not self.display_surface:
            return # Pygame not initialized
        try:
            text_surface = self.notification_font.render(text, True, color)
            expire_time = time.time() + duration_seconds
            self.notifications.append({
                "text": text,
                "color": color,
                "expire_time": expire_time,
                "surface": text_surface
            })
            # Keep notifications list sorted by expiry time or manage max notifications if needed
            self.notifications = self.notifications[-5:] # Keep max 5 notifications
        except Exception as e:
            self.logger.error(f"Error rendering notification text '{text}': {e}")

    def _render_notifications(self, screen_surface):
        """Renders active notifications onto the screen_surface."""
        if not self.notifications:
            return

        current_time = time.time()
        # Filter out expired notifications
        self.notifications = [n for n in self.notifications if n["expire_time"] > current_time]

        start_y = self.display_surface.get_height() - 10 # Start from bottom
        for i, notification in enumerate(reversed(self.notifications)):
            text_surface = notification["surface"]
            text_rect = text_surface.get_rect(bottomright=(self.display_surface.get_width() - 10, start_y - (i * (self.notification_font_size + 5))))
            
            # Add a semi-transparent background for better readability
            bg_rect = text_rect.inflate(10, 6) # Inflate for padding
            bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, 150)) # Black with alpha
            screen_surface.blit(bg_surface, bg_rect.topleft)
            screen_surface.blit(text_surface, text_rect.topleft)

    def reset_notifications(self):
        """Clears all active notifications."""
        self.notifications = []
        self.logger.debug("Cleared all on-screen notifications.")

    def _create_lidar_surface(self, lidar_points_local: np.ndarray, 
                              surface_size: Tuple[int, int], 
                              debug_info: dict) -> pygame.Surface:
        
        pixel_array_rgb = np.full((surface_size[1], surface_size[0], 3), [5, 5, 10], dtype=np.uint8)

        if lidar_points_local is None or lidar_points_local.shape[0] == 0 or debug_info is None:
            return pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))

        view_range_m = 30.0
        pixels_per_meter = min(surface_size[0] / (2 * view_range_m), surface_size[1] / (2 * view_range_m))
        if pixels_per_meter <= 0: 
            return pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))
        
        center_x_screen, center_y_screen = surface_size[0] // 2, surface_size[1] // 2

        vehicle_x_world = debug_info.get("_vehicle_world_x", 0.0)
        vehicle_y_world = debug_info.get("_vehicle_world_y", 0.0)
        vehicle_z_world = debug_info.get("_vehicle_world_z", 0.0)
        vehicle_yaw_rad_map_frame = debug_info.get("_vehicle_world_yaw_rad", 0.0)
        vehicle_pitch_rad = 0.0
        vehicle_roll_rad = 0.0

        vehicle_world_location = carla.Location(x=vehicle_x_world, y=vehicle_y_world, z=vehicle_z_world)
        vehicle_world_rotation = carla.Rotation(yaw=math.degrees(vehicle_yaw_rad_map_frame), 
                                              pitch=math.degrees(vehicle_pitch_rad), 
                                              roll=math.degrees(vehicle_roll_rad))
        vehicle_world_transform = carla.Transform(vehicle_world_location, vehicle_world_rotation)

        lidar_relative_transform = carla.Transform(carla.Location(x=0.0, z=2.5)) # Default LIDAR pose relative to vehicle

        # Manually compose the transforms: T_world_sensor = T_world_vehicle * T_vehicle_sensor
        # Step 1: Rotate the relative location of the sensor by the vehicle's world rotation
        # This gives the offset vector from vehicle origin to sensor origin, in world coordinates.
        # CARLA rotations are R_y(pitch) * R_z(yaw) * R_x(roll) (intrinsic)
        # Or R_x(roll) * R_y(pitch) * R_z(yaw) (extrinsic if applied in that order)
        # Simpler: get vehicle's forward, right, up vectors
        fwd = vehicle_world_transform.get_forward_vector()
        rgt = vehicle_world_transform.get_right_vector()
        up  = vehicle_world_transform.get_up_vector()

        # Sensor's location in world = vehicle_world_loc + R_world_vehicle * sensor_local_loc
        # sensor_local_loc is lidar_relative_transform.location (x=0, y=0, z=2.5 in vehicle frame)
        sensor_world_location = vehicle_world_location + \
                                lidar_relative_transform.location.x * fwd + \
                                lidar_relative_transform.location.y * rgt + \
                                lidar_relative_transform.location.z * up
        
        # Sensor's rotation in world = vehicle_world_rot combined with sensor_relative_rot
        # If lidar_relative_transform has no rotation, sensor_world_rotation is vehicle_world_rotation
        sensor_world_rotation = vehicle_world_rotation # Assuming lidar_relative_transform has no rotation for now
        # If lidar_relative_transform.rotation is non-zero, combine it: 
        # sensor_world_rotation = vehicle_world_rotation * lidar_relative_transform.rotation (if * was supported)
        # Manual: sensor_world_rotation.pitch += lidar_relative_transform.rotation.pitch (etc. for yaw, roll - careful with frames)
        # For simple relative transforms without much rotation, this is often sufficient:
        # sensor_world_rotation = carla.Rotation(\
        #     pitch = vehicle_world_rotation.pitch + lidar_relative_transform.rotation.pitch,\
        #     yaw   = vehicle_world_rotation.yaw   + lidar_relative_transform.rotation.yaw,\
        #     roll  = vehicle_world_rotation.roll  + lidar_relative_transform.rotation.roll\
        # )\n

        sensor_world_transform = carla.Transform(sensor_world_location, sensor_world_rotation)
        
        # Subsample LIDAR points early for better performance
        max_lidar_points = 5000  # Increased for more frequent rays
        if lidar_points_local.shape[0] > max_lidar_points:
            indices = np.random.choice(lidar_points_local.shape[0], size=max_lidar_points, replace=False)
            lidar_points_local_subsampled = lidar_points_local[indices]
        else:
            lidar_points_local_subsampled = lidar_points_local
        
        # Convert numpy array to list of carla.Location objects - use .item() for efficient numpy->Python conversion
        points_to_transform_carla_loc = [carla.Location(x=p[0].item(), y=p[1].item(), z=p[2].item()) for p in lidar_points_local_subsampled]
        
        world_points_carla = [sensor_world_transform.transform(p) for p in points_to_transform_carla_loc]
        world_points_np = np.array([[p.x, p.y, p.z] for p in world_points_carla])

        # Calculate offsets from vehicle center in world frame for projection (as before)
        dx_world_all = world_points_np[:, 0] - vehicle_x_world
        dy_world_all = world_points_np[:, 1] - vehicle_y_world

        screen_x_offset_all = dx_world_all * pixels_per_meter
        screen_y_offset_all = dy_world_all * pixels_per_meter 

        screen_x_all = (center_x_screen + screen_x_offset_all).astype(int)
        screen_y_all = (center_y_screen + screen_y_offset_all).astype(int)

        valid_indices = (screen_x_all >= 0) & (screen_x_all < surface_size[0]) & \
                        (screen_y_all >= 0) & (screen_y_all < surface_size[1])

        screen_x_valid = screen_x_all[valid_indices]
        screen_y_valid = screen_y_all[valid_indices]
        # Intensities are not available in the processed lidar_points_local (N,3) array
        # So, color points by height (z value) or a fixed color.
        z_values_world_valid = world_points_np[valid_indices, 2]

        lidar_surface = pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))

        if screen_x_valid.size > 0:
            point_size = 3 # Make points larger for a bolder look
            
            # Use a fixed, bright color for all LIDAR points for maximum impact
            fixed_color = (255, 255, 0) # Changed to Yellow

            for i in range(len(screen_x_valid)):
                x, y = screen_x_valid[i], screen_y_valid[i]
                pygame.gfxdraw.filled_circle(lidar_surface, x, y, point_size, fixed_color)
        
        # --- Draw ego vehicle marker (remains similar) ---
        # ... (vehicle marker drawing code using center_x_screen, center_y_screen, vehicle_yaw_rad_map_frame, pixels_per_meter)
        ego_color = (0, 255, 0); vw_m, vl_m = 1.8, 4.5 # vehicle width/length in meters
        vw_px, vl_px = int(vw_m * pixels_per_meter), int(vl_m * pixels_per_meter)
        hl, hw = vl_px / 2, vw_px / 2
        rect_pts_local = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        angle_pygame = -vehicle_yaw_rad_map_frame # Assuming yaw=0 (East) means point right on screen
        cos_a, sin_a = math.cos(angle_pygame), math.sin(angle_pygame)
        rot_rect_pts_screen = [(center_x_screen + int(px*cos_a - py*sin_a), center_y_screen + int(px*sin_a + py*cos_a)) for px,py in rect_pts_local]
        pygame.draw.polygon(lidar_surface, ego_color, rot_rect_pts_screen, 2)
        pygame.draw.circle(lidar_surface, (255,0,0), (center_x_screen, center_y_screen), 2) # Center dot

        return lidar_surface

    def _create_semantic_lidar_surface(self, sem_lidar_data: any, 
                                       surface_size: Tuple[int, int], debug_info: dict) -> pygame.Surface:
        """Create a surface for semantic LIDAR data.
        Supports BOTH raw carla.SemanticLidarMeasurement and pre-processed NumPy arrays with
        shape (N,4) where last column is object_tag."""
        
        # Dark background for contrast
        pixel_array_rgb = np.full((surface_size[1], surface_size[0], 3), [5, 5, 10], dtype=np.uint8)

        if sem_lidar_data is None or len(sem_lidar_data) == 0 or debug_info is None: 
            return pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))

        # Determine input type
        is_numpy = isinstance(sem_lidar_data, np.ndarray)

        if is_numpy and sem_lidar_data.ndim == 2 and sem_lidar_data.shape[1] >= 4:
            # Already processed NumPy array: columns 0..2 => x,y,z , column 3 => tag
            local_points_np = sem_lidar_data[:, :3].astype(np.float32)
            object_tags_np = sem_lidar_data[:, 3].astype(np.uint32)
            sensor_transform = None  # We'll compute transform manually similar to regular LIDAR
        else:
            # Assume raw carla.SemanticLidarMeasurement
            sensor_transform = sem_lidar_data.transform
            # Extract point data and object tags
            local_points_carla = [det.point for det in sem_lidar_data]
            object_tags_np = np.array([det.object_tag for det in sem_lidar_data], dtype=np.uint32)
            local_points_np = np.array([[p.x, p.y, p.z] for p in local_points_carla], dtype=np.float32)

        num_detections = local_points_np.shape[0]
        if num_detections == 0:
            return pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))

        # Subsample to keep performance reasonable
        max_disp_pts = 2000
        if num_detections > max_disp_pts:
            indices = np.random.choice(num_detections, size=max_disp_pts, replace=False)
            local_points_np = local_points_np[indices]
            object_tags_np = object_tags_np[indices]

        # Determine screen scaling based on desired view range
        view_range_m = 30.0  # Same as regular LIDAR view
        pixels_per_meter = min(surface_size[0] / (2 * view_range_m), surface_size[1] / (2 * view_range_m))
        if pixels_per_meter <= 0:
            return pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))

        center_x_screen, center_y_screen = surface_size[0] // 2, surface_size[1] // 2

        # Figure out transform from sensor (local) to world frame
        vehicle_x_world = debug_info.get("_vehicle_world_x", 0.0)
        vehicle_y_world = debug_info.get("_vehicle_world_y", 0.0)
        vehicle_yaw_rad_map_frame = debug_info.get("_vehicle_world_yaw_rad", 0.0)

        if sensor_transform is None:
            # Assume sensor is mounted on roof similar to regular LIDAR (x=0,y=0,z=2.5)
            sensor_world_x = vehicle_x_world + math.cos(vehicle_yaw_rad_map_frame) * 0.0 - math.sin(vehicle_yaw_rad_map_frame)*0.0
            sensor_world_y = vehicle_y_world + math.sin(vehicle_yaw_rad_map_frame) * 0.0 + math.cos(vehicle_yaw_rad_map_frame)*0.0
            dx_world_all = (local_points_np[:,0]*math.cos(vehicle_yaw_rad_map_frame) - local_points_np[:,1]*math.sin(vehicle_yaw_rad_map_frame))
            dy_world_all = (local_points_np[:,0]*math.sin(vehicle_yaw_rad_map_frame) + local_points_np[:,1]*math.cos(vehicle_yaw_rad_map_frame))
            world_points_np = np.column_stack((sensor_world_x + dx_world_all,
                                              sensor_world_y + dy_world_all,
                                              local_points_np[:,2]))
        else:
            # Use sensor_transform from CARLA
            world_pts = [sensor_transform.transform(carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2]))) for p in local_points_np]
            world_points_np = np.array([[p.x, p.y, p.z] for p in world_pts], dtype=np.float32)

        # Calculate offsets from vehicle in world frame (vectorized)
        dx_world_all = world_points_np[:, 0] - vehicle_x_world
        dy_world_all = world_points_np[:, 1] - vehicle_y_world

        screen_x_offset_all = dx_world_all * pixels_per_meter
        screen_y_offset_all = dy_world_all * pixels_per_meter # Y-South for CARLA, Y-Down for Pygame

        screen_x_all = (center_x_screen + screen_x_offset_all).astype(int)
        screen_y_all = (center_y_screen + screen_y_offset_all).astype(int)

        # Filter points outside screen bounds
        valid_indices = (screen_x_all >= 0) & (screen_x_all < surface_size[0]) & \
                        (screen_y_all >= 0) & (screen_y_all < surface_size[1])

        screen_x_valid = screen_x_all[valid_indices]
        screen_y_valid = screen_y_all[valid_indices]
        object_tags_valid = object_tags_np[valid_indices]

        # Create surface initially
        sem_lidar_surface = pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))

        if screen_x_valid.size > 0:
            # Get colors for valid points based on object_tag
            colors_for_points = PYGAME_LABEL_COLORS[object_tags_valid % len(PYGAME_LABEL_COLORS)]
            
            # Enhance brightness of colors (multiply by 2.0 and clip to 255)
            r_valid = np.minimum(colors_for_points[:, 0] * 2.0, 255).astype(np.uint8)
            g_valid = np.minimum(colors_for_points[:, 1] * 2.0, 255).astype(np.uint8)
            b_valid = np.minimum(colors_for_points[:, 2] * 2.0, 255).astype(np.uint8)
            
            point_size = 3  # Increased from 2 to 3 for boldness
            
            max_points_to_draw = min(len(screen_x_valid), 1500) # This was from semantic, keep it or match regular lidar's 5000?
                                                              # For now, keeping its own subsampling limit, focus on boldness.
            step = max(1, len(screen_x_valid) // max_points_to_draw)
            
            for i in range(0, len(screen_x_valid), step):
                x, y = screen_x_valid[i], screen_y_valid[i]
                color = (r_valid[i], g_valid[i], b_valid[i])
                pygame.gfxdraw.filled_circle(sem_lidar_surface, x, y, point_size, color)

        # --- Draw ego vehicle marker --- 
        vehicle_yaw_rad_map_frame = debug_info.get("_vehicle_world_yaw_rad",0.0)
        ego_col=(0,255,0); vw,vl=1.8,4.5; vwp,vlp=int(vw*pixels_per_meter),int(vl*pixels_per_meter); hl,hw=vlp/2,vwp/2
        pts_loc=[(-hw,-hl),(hw,-hl),(hw,hl),(-hw,hl)]; ang = -vehicle_yaw_rad_map_frame + math.pi / 2
        cos_a,sin_a=math.cos(ang),math.sin(ang)
        rot_pts=[(center_x_screen+int(x*cos_a-y*sin_a),center_y_screen+int(x*sin_a+y*cos_a)) for x,y in pts_loc]
        pygame.draw.polygon(sem_lidar_surface,ego_col,rot_pts,2); pygame.draw.circle(sem_lidar_surface,(255,0,0),(center_x_screen,center_y_screen),2)
        return sem_lidar_surface

    def _create_radar_surface(self, radar_data: any, surface_size: Tuple[int, int], 
                              actual_radar_sensor_range: float, debug_info: dict) -> pygame.Surface:
        radar_surface = pygame.Surface(surface_size)
        radar_surface.fill((12, 8, 10))  # Darker background for better contrast with radar points

        if radar_data is None or debug_info is None:
            return radar_surface
        
        # Determine if radar_data is raw CARLA measurement or processed NumPy array
        is_numpy_array = isinstance(radar_data, np.ndarray)

        if is_numpy_array and radar_data.shape[0] == 0:
             return radar_surface # Empty numpy array
        elif not is_numpy_array and len(radar_data) == 0:
             return radar_surface # Empty raw measurement

        view_range_m = actual_radar_sensor_range if actual_radar_sensor_range > 0 else 70.0
        pixels_per_meter = min(surface_size[0] / (2 * view_range_m), surface_size[1] / (2 * view_range_m))
        if pixels_per_meter <= 0: return radar_surface

        center_x_screen = surface_size[0] // 2
        center_y_screen = surface_size[1] // 2

        vehicle_x_world = debug_info.get("_vehicle_world_x", 0.0)
        vehicle_y_world = debug_info.get("_vehicle_world_y", 0.0)
        vehicle_yaw_rad_map_frame = debug_info.get("_vehicle_world_yaw_rad", 0.0)
        radar_to_vehicle_transform = debug_info.get("_radar_to_vehicle_transform", carla.Transform())

        # Draw ego vehicle marker (same as LIDAR view)
        ego_color = (0, 255, 0) 
        vehicle_width_m = 1.8; vehicle_length_m = 4.5
        vehicle_width_px = int(vehicle_width_m * pixels_per_meter)
        vehicle_length_px = int(vehicle_length_m * pixels_per_meter)
        hl, hw = vehicle_length_px / 2, vehicle_width_px / 2
        rect_points_local = [(-hw, -hl), (hw, -hl), (hw, hl), (-hw, hl)]
        angle_to_rotate_pygame = -vehicle_yaw_rad_map_frame 
        cos_a = math.cos(angle_to_rotate_pygame); sin_a = math.sin(angle_to_rotate_pygame)
        rotated_rect_points_screen = []
        for x, y in rect_points_local:
            rotated_x = x * cos_a - y * sin_a
            rotated_y = x * sin_a + y * cos_a
            rotated_rect_points_screen.append((center_x_screen + int(rotated_x), center_y_screen + int(rotated_y)))
        pygame.draw.polygon(radar_surface, ego_color, rotated_rect_points_screen, 2)
        pygame.draw.circle(radar_surface, (255,0,0), (center_x_screen, center_y_screen), 2)

        # Draw RADAR detections
        detections_to_process = []
        if is_numpy_array:
            # radar_data is (N, 4) with [depth, azimuth, altitude, velocity]
            for row in radar_data:
                # Only process if depth is not zero (padded entries are all zeros)
                if row[0] > 1e-6: # Check depth (index 0 in processed array)
                    detections_to_process.append({
                        'depth': row[0],
                        'azimuth': row[1],
                        'altitude': row[2],
                        'velocity': row[3]
                    })
        else: # Assume raw carla.RadarMeasurement (list of carla.RadarDetection)
            for det in radar_data:
                detections_to_process.append({
                    'depth': det.depth,
                    'azimuth': det.azimuth,
                    'altitude': det.altitude,
                    'velocity': det.velocity
                })

        for detection_dict in detections_to_process:
            depth = detection_dict['depth']
            azimuth = detection_dict['azimuth']
            altitude = detection_dict['altitude']
            velocity = detection_dict['velocity']

            x_sensor_local = depth * math.cos(altitude) * math.cos(azimuth)
            y_sensor_local = depth * math.cos(altitude) * math.sin(azimuth)
            
            p_sensor_local = carla.Location(x=float(x_sensor_local), y=float(y_sensor_local), z=0.0) 
            p_vehicle_local = radar_to_vehicle_transform.transform(p_sensor_local)

            dx_world = p_vehicle_local.x * math.cos(vehicle_yaw_rad_map_frame) - p_vehicle_local.y * math.sin(vehicle_yaw_rad_map_frame)
            dy_world = p_vehicle_local.x * math.sin(vehicle_yaw_rad_map_frame) + p_vehicle_local.y * math.cos(vehicle_yaw_rad_map_frame)

            screen_x_offset = dx_world * pixels_per_meter
            screen_y_offset = -dy_world * pixels_per_meter

            screen_x = center_x_screen + int(screen_x_offset)
            screen_y = center_y_screen + int(screen_y_offset)

            if velocity < -0.1: color = (255, 120, 120) 
            elif velocity > 0.1: color = (120, 120, 255) 
            else: color = (230, 230, 230) 
            
            if 0 <= screen_x < surface_size[0] and 0 <= screen_y < surface_size[1]:
                pygame.draw.circle(radar_surface, color, (screen_x, screen_y), 5) 
        
        return radar_surface

    def render(self, raw_sensor_data_item: any, current_view_key: str, debug_info_from_env: dict, lidar_sensor_range_from_env: float = 0.0):
        if not self.is_active:
            return False

        current_display_size = self.display_surface.get_size()
        self.clock.tick(30) # Try to cap at 30 FPS

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
                elif event.key == pygame.K_l: # Toggle Open3D LIDAR visualizer
                    if self.carla_env_ref:
                        env_instance = self.carla_env_ref() if isinstance(self.carla_env_ref, weakref.ref) else self.carla_env_ref
                        if env_instance and hasattr(env_instance, 'toggle_o3d_lidar_visualization'):
                            env_instance.toggle_o3d_lidar_visualization()
                            self.logger.info("Toggle Open3D LIDAR visualizer command sent.")
                        else:
                            self.logger.warning("CarlaEnv reference or toggle method not available for Open3D LIDAR.")

        surface_to_display = None
        if raw_sensor_data_item is not None:
            if current_view_key == 'lidar':
                surface_to_display = self._create_lidar_surface(raw_sensor_data_item, current_display_size, debug_info_from_env)
            elif current_view_key == 'semantic_lidar':
                surface_to_display = self._create_semantic_lidar_surface(raw_sensor_data_item, current_display_size, debug_info_from_env)
            elif current_view_key == 'radar':
                radar_sensor_range = 0.0 
                if self.carla_env_ref:
                    env_instance = self.carla_env_ref() if isinstance(self.carla_env_ref, weakref.ref) else self.carla_env_ref
                    if env_instance and hasattr(env_instance, 'radar_config') and 'range' in env_instance.radar_config:
                        radar_sensor_range = float(env_instance.radar_config['range'])
                surface_to_display = self._create_radar_surface(raw_sensor_data_item, current_display_size, radar_sensor_range, debug_info_from_env)
            elif isinstance(raw_sensor_data_item, carla.Image):
                temp_surface = self._process_carla_image_for_display(raw_sensor_data_item, current_view_key)
                surface_to_display = temp_surface
            elif isinstance(raw_sensor_data_item, np.ndarray) and current_view_key.endswith('_camera'):
                # Handle pre-processed NumPy array from agent sensors if they are selected for display
                if raw_sensor_data_item.ndim == 3 and raw_sensor_data_item.shape[0] in [1, 3]: # CHW format
                    if raw_sensor_data_item.shape[0] == 3: # RGB CHW
                        img_hwc = np.transpose(raw_sensor_data_item, (1, 2, 0)).astype(np.uint8)
                        temp_surface = pygame.surfarray.make_surface(img_hwc)
                    elif raw_sensor_data_item.shape[0] == 1: # Depth or Semantic (single channel CHW)
                        single_channel_hw = raw_sensor_data_item[0] # Extract H,W array
                        
                        if current_view_key == 'depth_camera' or current_view_key == 'display_depth_camera':
                            # Depth data is expected to be normalized [0,1]
                            depth_display_hw = (np.clip(single_channel_hw, 0, 1) * 255).astype(np.uint8)
                            # Create a 3-channel grayscale image for Pygame display
                            depth_display_hwc = np.stack((depth_display_hw,) * 3, axis=-1)
                            temp_surface = pygame.surfarray.make_surface(depth_display_hwc)
                        
                        elif current_view_key == 'semantic_camera' or current_view_key == 'display_semantic_camera':
                            # Semantic data contains class labels (uint8)
                            labels_hw = single_channel_hw.astype(np.uint8)
                            # Map labels to colors
                            colored_semantic_hwc = PYGAME_LABEL_COLORS[labels_hw % len(PYGAME_LABEL_COLORS)]
                            temp_surface = pygame.surfarray.make_surface(colored_semantic_hwc) # Already HWC RGB
                        else: 
                            self.logger.warning(f"PygameVisualizer: Unhandled single-channel NumPy camera view '{current_view_key}'")
                            temp_surface = pygame.Surface(current_display_size); temp_surface.fill((40,40,40))
                    else: # Should not happen if CHW with C=1 or C=3
                        self.logger.warning(f"PygameVisualizer: NumPy camera view '{current_view_key}' has unexpected channel count: {raw_sensor_data_item.shape[0]}")
                        temp_surface = pygame.Surface(current_display_size); temp_surface.fill((50,50,50))
                    
                    # Common transformations for camera views
                    temp_surface = pygame.transform.rotate(temp_surface, -90)
                    temp_surface = pygame.transform.flip(temp_surface, True, False)
                    surface_to_display = temp_surface
                else:
                     self.logger.warning(f"PygameVisualizer: Received NumPy array for camera view '{current_view_key}' with unexpected shape or ndim: {raw_sensor_data_item.shape}, ndim={raw_sensor_data_item.ndim}")

        if surface_to_display:
            # If it's an image sensor or RADAR/LIDAR (already screen_size), blit directly.
            # Only scale if it's a small camera sensor image.
            if current_view_key not in ['lidar', 'radar', 'semantic_lidar']:
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

        self._render_notifications(self.display_surface) # Call this after other elements are drawn

        pygame.display.flip()
        return True

    def close(self):
        if pygame.get_init(): 
            pygame.quit()
        self.is_active = False
        # print("PygameVisualizer closed.") # Optional: log close 

    def update_goal_waypoint_debug(self, target_waypoint: Optional[carla.Waypoint]):
        """Passes the target waypoint to the HUD for display/use."""
        if self.hud:
            self.hud.update_target_waypoint(target_waypoint)
        else:
            self.logger.warning("HUD not initialized in PygameVisualizer, cannot update goal waypoint.") 

    def _process_carla_image_for_display(self, carla_image: carla.Image, view_key: str) -> pygame.Surface:
        """Processes a raw carla.Image into a Pygame surface for display, applying necessary conversions."""
        # The carla.Image.convert() method modifies the image in-place.
        # If the original image in latest_sensor_data needs to be preserved unmodified,
        # the callback storing it there should store a copy, or this method should receive a copy.
        # For now, we operate on the passed image directly.

        if view_key == 'depth_camera' or view_key == 'display_depth_camera':
            carla_image.convert(carla.ColorConverter.LogarithmicDepth)
        elif view_key == 'semantic_camera' or view_key == 'display_semantic_camera':
            carla_image.convert(carla.ColorConverter.CityScapesPalette)
        # else: RGB or spectator_camera, raw_data is BGRA, no .convert() needed before manual BGRA to RGB

        img_bgra = np.array(carla_image.raw_data).reshape((carla_image.height, carla_image.width, 4))
        img_rgb_h_w_c = img_bgra[:, :, :3][:, :, ::-1] # BGRA to BGR, then BGR to RGB
        surface = pygame.surfarray.make_surface(img_rgb_h_w_c)
        
        # Apply standard transformations for display
        surface = pygame.transform.rotate(surface, -90)
        surface = pygame.transform.flip(surface, True, False)
        return surface 