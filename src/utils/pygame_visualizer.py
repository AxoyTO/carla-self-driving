import pygame
import numpy as np
import carla # For type hinting carla.Image and carla.ColorConverter
from .hud import HUD # Import the new HUD class
import weakref
from collections import OrderedDict
import logging # Import logging
import math # For sin/cos in RADAR, and general math
from typing import Tuple
import random # For subsampling in semantic lidar view
import pygame.gfxdraw  # For anti-aliased circles

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
            'spectator_camera',       # Already high-res
            'display_rgb_camera',     # New display key for front RGB
            'display_left_rgb_camera',# New display key for left RGB
            'display_right_rgb_camera',# New display key for right RGB
            'display_rear_rgb_camera', # New display key for rear RGB
            'display_depth_camera',   # New display key for depth
            'display_semantic_camera',# New display key for semantic
            'lidar',            
            'semantic_lidar',
            'radar'               
        ]
        self.current_view_idx = 0
        # Mapping for pretty display names
        self.view_display_names = {
            'spectator_camera': "Spectator",
            'display_rgb_camera': "Front RGB Camera",
            'display_left_rgb_camera': "Left RGB Camera",
            'display_right_rgb_camera': "Right RGB Camera",
            'display_rear_rgb_camera': "Rear RGB Camera",
            'display_depth_camera': "Depth Camera",
            'display_semantic_camera': "Semantic Camera",
            'lidar': "LIDAR (Top-Down)", # Updated to be more descriptive
            'semantic_lidar': "Semantic LIDAR (Top-Down)",
            'radar': "RADAR (Top-Down)"  # Updated to be more descriptive
        }

    def get_current_view_key(self) -> str:
        """Returns the raw key of the current view source."""
        return self.view_source_keys[self.current_view_idx]

    def get_current_view_display_name(self) -> str:
        """Returns the user-friendly display name for the current view."""
        raw_key = self.get_current_view_key()
        return self.view_display_names.get(raw_key, raw_key) # Fallback to raw key if no pretty name

    def add_notification(self, text: str, duration_seconds: float = 3.0, color=None):
        if self.hud and self.hud_visible: # Only add if HUD exists and is visible
            self.hud.add_notification(text, duration_seconds, color=color)

    def _create_lidar_surface(self, lidar_data: carla.LidarMeasurement, surface_size: Tuple[int, int], 
                              actual_lidar_sensor_range: float, debug_info: dict) -> pygame.Surface:
        
        # Create a NumPy array for the surface pixel data (H, W, C for RGB)
        # Even darker background for better contrast with bright points
        pixel_array_rgb = np.full((surface_size[1], surface_size[0], 3), [5, 5, 10], dtype=np.uint8)

        if lidar_data is None or len(lidar_data) == 0 or debug_info is None:
            # If no data, create surface from the initial background
            lidar_surface = pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))
            return lidar_surface

        # Bird\'s-eye view settings
        view_range_m = 30.0  # Display a 60x60 meter area (view_range_m in each direction from center)
        pixels_per_meter = min(surface_size[0] / (2 * view_range_m), surface_size[1] / (2 * view_range_m))
        if pixels_per_meter <= 0: 
            lidar_surface = pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))
            return lidar_surface
        
        center_x_screen = surface_size[0] // 2
        center_y_screen = surface_size[1] // 2

        vehicle_x_world = debug_info.get("_vehicle_world_x", 0.0)
        vehicle_y_world = debug_info.get("_vehicle_world_y", 0.0)
        # vehicle_yaw_rad_map_frame = debug_info.get("_vehicle_world_yaw_rad", 0.0) # For ego vehicle marker

        sensor_transform_at_capture = lidar_data.transform
        
        # Get raw points: x,y,z,intensity
        local_points_xyzi = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')).reshape((-1, 4))
        
        if local_points_xyzi.shape[0] == 0:
            lidar_surface = pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))
            return lidar_surface

        max_points_to_process = 2500  # Reduce points to process for better performance
        if local_points_xyzi.shape[0] > max_points_to_process:
            step = local_points_xyzi.shape[0] // max_points_to_process or 1
            local_points_xyzi = local_points_xyzi[::step, :]

        # Extract X, Y, Z for carla.Location list
        points_to_transform = [carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in local_points_xyzi]
        
        # Transform all points to world coordinates
        world_points_carla = [sensor_transform_at_capture.transform(p) for p in points_to_transform]
        
        # Convert world points back to NumPy array for vectorized calculations
        world_points_np = np.array([[float(p.x), float(p.y), float(p.z)] for p in world_points_carla]) # Shape (N, 3)

        # Calculate offsets from vehicle in world frame (vectorized)
        dx_world_all = world_points_np[:, 0] - vehicle_x_world
        dy_world_all = world_points_np[:, 1] - vehicle_y_world
        # dz_world_all = world_points_np[:, 2] - vehicle_z_world # If needed for height coloring

        # Convert world offsets to screen offsets (vectorized)
        # Map North (CARLA +X vehicle default) is Pygame screen Up (-Y)
        # Map East (CARLA -Y vehicle default) is Pygame screen Right (+X)
        # Assuming standard CARLA world coordinates: X-East, Y-South, Z-Up
        # And vehicle forward is along its X-axis.
        # For top-down view where vehicle forward is "up" on screen (negative Y in pygame):
        # We need to rotate points relative to vehicle yaw if vehicle_yaw_rad_map_frame is 0 when vehicle faces screen-up.
        # For now, simple projection assuming vehicle at center, its X axis is world X, Y is world Y.
        # This means world X (East) maps to screen X (Right), world Y (South) maps to screen Y (Down)
        
        screen_x_offset_all = dx_world_all * pixels_per_meter
        screen_y_offset_all = dy_world_all * pixels_per_meter # Y-South for CARLA, Y-Down for Pygame, so sign matches

        # Rotate points by negative vehicle yaw to align vehicle forward with screen "up"
        # This is a simplification. A full transformation from world to vehicle-centric-view-aligned screen coords is more robust.
        # For now, let's stick to the simpler direct world to screen as before, assuming map North = screen Up.
        # screen_x_offset_all = dx_world_all * pixels_per_meter
        # screen_y_offset_all = -dy_world_all * pixels_per_meter # Y-South for CARLA, -Y for Pygame-Up with North-Up map

        screen_x_all = (center_x_screen + screen_x_offset_all).astype(int)
        screen_y_all = (center_y_screen + screen_y_offset_all).astype(int) # if using Y-South for Pygame as well
        # screen_y_all = (center_y_screen - screen_y_offset_all).astype(int) # if using Y-North for Pygame

        # Filter points outside screen bounds
        valid_indices = (screen_x_all >= 0) & (screen_x_all < surface_size[0]) & \
                        (screen_y_all >= 0) & (screen_y_all < surface_size[1])

        screen_x_valid = screen_x_all[valid_indices]
        screen_y_valid = screen_y_all[valid_indices]
        intensities_valid = local_points_xyzi[valid_indices, 3]

        # Create surface initially
        lidar_surface = pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))

        if screen_x_valid.size > 0: # If there are any points to draw
            # Calculate colors for valid points (vectorized) with increased brightness
            norm_intensity_valid = np.clip(intensities_valid, 0.0, 1.0)
            color_val_valid = (norm_intensity_valid * 250 + 75).astype(int)  # Increase brightness further
            
            r_valid = np.clip(color_val_valid, 0, 255)
            g_valid = np.clip(color_val_valid, 0, 255)
            b_valid = np.clip(color_val_valid // 2, 0, 255)

            # Use a more efficient approach: draw points directly on the surface
            point_size = 3  # Larger points for regular LIDAR for better visibility
            
            # Balance between visibility (more points) and performance
            max_points_to_draw = min(len(screen_x_valid), 1500)
            step = max(1, len(screen_x_valid) // max_points_to_draw)
            
            for i in range(0, len(screen_x_valid), step):
                x, y = screen_x_valid[i], screen_y_valid[i]
                color = (r_valid[i], g_valid[i], b_valid[i])
                # Use anti-aliased circle for smoother appearance
                pygame.gfxdraw.filled_circle(lidar_surface, x, y, point_size, color)
        
        # --- Draw ego vehicle marker (rotated rectangle) ---
        # This part remains largely the same, using the now created lidar_surface
        vehicle_yaw_rad_map_frame = debug_info.get("_vehicle_world_yaw_rad", 0.0)
        ego_color = (0, 255, 0) 
        vehicle_width_m = 1.8
        vehicle_length_m = 4.5
        vehicle_width_px = int(vehicle_width_m * pixels_per_meter)
        vehicle_length_px = int(vehicle_length_m * pixels_per_meter)
        hl, hw = vehicle_length_px / 2, vehicle_width_px / 2
        rect_points_local = [
            (-hw, -hl), ( hw, -hl), ( hw,  hl), (-hw,  hl)
        ]
        # The angle needs to be relative to Pygame\'s coordinate system for rotation.
        # If map North is screen Up, and CARLA yaw is 0 for East, 90 for North:
        # Pygame rotation: 0 is screen Right, 90 is screen Down.
        # angle_to_rotate_pygame = -vehicle_yaw_rad_map_frame # If yaw=0 (East) means no rotation from Pygame X-axis
        # Let's adjust based on common top-down view where vehicle "up" is aligned with screen "up"
        # If vehicle_yaw_rad_map_frame is 0 when car points East (positive world X)
        # and we want vehicle front to be "up" on screen (negative Pygame Y)
        # A rotation of vehicle_yaw_rad_map_frame should align its local X with world X.
        # Then to point "up" on screen, this world X needs to map to screen -Y.
        # For polygon drawing, it's simpler to rotate points relative to screen axes.
        # If vehicle "forward" on screen is Pygame\'s negative Y direction:
        angle_to_rotate_pygame = -vehicle_yaw_rad_map_frame + math.pi / 2 # Add pi/2 so 0 world yaw (East) points right on screen
                                                                    # then negate for pygame rotation direction.
                                                                    # This is often tricky to get right.
                                                                    # The previous version had: angle_to_rotate_pygame = -vehicle_yaw_rad_map_frame

        cos_a = math.cos(angle_to_rotate_pygame)
        sin_a = math.sin(angle_to_rotate_pygame)
        rotated_rect_points_screen = []
        for x, y in rect_points_local: # These are vehicle-local, centered at (0,0) for the rect
            rotated_x = x * cos_a - y * sin_a
            rotated_y = x * sin_a + y * cos_a
            rotated_rect_points_screen.append((center_x_screen + int(rotated_x), center_y_screen + int(rotated_y)))
        pygame.draw.polygon(lidar_surface, ego_color, rotated_rect_points_screen, 2)
        pygame.draw.circle(lidar_surface, (255,0,0), (center_x_screen, center_y_screen), 2) # Center dot

        return lidar_surface

    def _create_semantic_lidar_surface(self, sem_lidar_data: carla.SemanticLidarMeasurement, 
                                       surface_size: Tuple[int, int], debug_info: dict) -> pygame.Surface:
        
        # Even darker background for better contrast with bright points
        pixel_array_rgb = np.full((surface_size[1], surface_size[0], 3), [5, 5, 10], dtype=np.uint8)

        if sem_lidar_data is None or len(sem_lidar_data) == 0 or debug_info is None: 
            sem_lidar_surface = pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))
            return sem_lidar_surface
        
        view_range_m = 30.0
        pixels_per_meter = min(surface_size[0] / (2 * view_range_m), surface_size[1] / (2 * view_range_m))
        if pixels_per_meter <= 0: 
            sem_lidar_surface = pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))
            return sem_lidar_surface
            
        center_x_screen, center_y_screen = surface_size[0]//2, surface_size[1]//2
        vehicle_x_world = debug_info.get("_vehicle_world_x",0.0)
        vehicle_y_world = debug_info.get("_vehicle_world_y",0.0)
        # vehicle_yaw_rad_map_frame = debug_info.get("_vehicle_world_yaw_rad",0.0) # For ego marker

        sensor_transform = sem_lidar_data.transform

        # Extract point data and object tags
        # carla.SemanticLidarDetection: point (carla.Location), object_idx, object_tag
        num_detections = len(sem_lidar_data)
        if num_detections == 0:
            sem_lidar_surface = pygame.surfarray.make_surface(np.transpose(pixel_array_rgb, (1,0,2)))
            return sem_lidar_surface

        local_points_carla = [det.point for det in sem_lidar_data]
        object_tags_np = np.array([det.object_tag for det in sem_lidar_data], dtype=np.uint32)

        # More aggressive subsampling for better performance
        max_disp_pts = 2500
        if num_detections > max_disp_pts:
            indices_to_keep = np.random.choice(num_detections, size=max_disp_pts, replace=False)
            local_points_carla = [local_points_carla[i] for i in indices_to_keep]
            object_tags_np = object_tags_np[indices_to_keep]
        
        # Transform all points to world coordinates
        world_points_carla = [sensor_transform.transform(p) for p in local_points_carla]
        world_points_np = np.array([[float(p.x), float(p.y), float(p.z)] for p in world_points_carla])

        # Calculate offsets from vehicle in world frame (vectorized)
        dx_world_all = world_points_np[:, 0] - vehicle_x_world
        dy_world_all = world_points_np[:, 1] - vehicle_y_world

        # Convert world offsets to screen offsets (vectorized)
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
            # PYGAME_LABEL_COLORS is (NumColors, 3)
            # Ensure tags are within bounds of PYGAME_LABEL_COLORS
            colors_for_points = PYGAME_LABEL_COLORS[object_tags_valid % len(PYGAME_LABEL_COLORS)]
            
            # Enhance brightness of colors (multiply by 2.0 and clip to 255)
            r_valid = np.minimum(colors_for_points[:, 0] * 2.0, 255).astype(np.uint8)
            g_valid = np.minimum(colors_for_points[:, 1] * 2.0, 255).astype(np.uint8)
            b_valid = np.minimum(colors_for_points[:, 2] * 2.0, 255).astype(np.uint8)
            
            # Use a more efficient approach: draw points directly on the surface
            point_size = 2  # Keep semantic LIDAR points a bit smaller than regular LIDAR
            
            # Balance between visibility (more points) and performance
            max_points_to_draw = min(len(screen_x_valid), 1500)
            step = max(1, len(screen_x_valid) // max_points_to_draw)
            
            for i in range(0, len(screen_x_valid), step):
                x, y = screen_x_valid[i], screen_y_valid[i]
                color = (r_valid[i], g_valid[i], b_valid[i])
                # Use anti-aliased circle for smoother appearance
                pygame.gfxdraw.filled_circle(sem_lidar_surface, x, y, point_size, color)

        # --- Draw ego vehicle marker --- 
        vehicle_yaw_rad_map_frame = debug_info.get("_vehicle_world_yaw_rad",0.0)
        ego_col=(0,255,0); vw,vl=1.8,4.5; vwp,vlp=int(vw*pixels_per_meter),int(vl*pixels_per_meter); hl,hw=vlp/2,vwp/2
        pts_loc=[(-hw,-hl),(hw,-hl),(hw,hl),(-hw,hl)]; ang = -vehicle_yaw_rad_map_frame + math.pi / 2
        cos_a,sin_a=math.cos(ang),math.sin(ang)
        rot_pts=[(center_x_screen+int(x*cos_a-y*sin_a),center_y_screen+int(x*sin_a+y*cos_a)) for x,y in pts_loc]
        pygame.draw.polygon(sem_lidar_surface,ego_col,rot_pts,2); pygame.draw.circle(sem_lidar_surface,(255,0,0),(center_x_screen,center_y_screen),2)
        return sem_lidar_surface

    def _create_radar_surface(self, radar_data: carla.RadarMeasurement, surface_size: Tuple[int, int], 
                              actual_radar_sensor_range: float, debug_info: dict) -> pygame.Surface:
        radar_surface = pygame.Surface(surface_size)
        radar_surface.fill((12, 8, 10))  # Darker background for better contrast with radar points

        if radar_data is None or len(radar_data) == 0 or debug_info is None:
            return radar_surface

        view_range_m = actual_radar_sensor_range if actual_radar_sensor_range > 0 else 70.0
        pixels_per_meter = min(surface_size[0] / (2 * view_range_m), surface_size[1] / (2 * view_range_m))
        if pixels_per_meter <= 0: return radar_surface

        center_x_screen = surface_size[0] // 2
        center_y_screen = surface_size[1] // 2

        vehicle_x_world = debug_info.get("_vehicle_world_x", 0.0)
        vehicle_y_world = debug_info.get("_vehicle_world_y", 0.0)
        vehicle_yaw_rad_map_frame = debug_info.get("_vehicle_world_yaw_rad", 0.0)
        # Get the RADAR sensor's transform relative to the vehicle
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
        for detection in radar_data:
            # detection.depth, detection.azimuth (rad), detection.altitude (rad), detection.velocity
            # 1. Convert radar detection from sensor spherical to sensor Cartesian
            # Azimuth is relative to sensor's forward X-axis. Altitude is angle with XY plane.
            # x = R * cos(alt) * cos(az_sensor)
            # y = R * cos(alt) * sin(az_sensor)
            # z = R * sin(alt)
            x_sensor_local = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
            y_sensor_local = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
            # z_sensor_local = detection.depth * math.sin(detection.altitude) # Not used for 2D top-down
            
            p_sensor_local = carla.Location(x=float(x_sensor_local), y=float(y_sensor_local), z=0.0) # Use z=0 for 2D projection

            # 2. Transform point from RADAR sensor local frame to VEHICLE local frame
            p_vehicle_local = radar_to_vehicle_transform.transform(p_sensor_local)

            # p_vehicle_local.x is now distance in front(+) / behind(-) vehicle center
            # p_vehicle_local.y is now distance to left(+) / right(-) of vehicle center

            # 3. Rotate this vehicle-local point by vehicle's world yaw to get world-frame offsets
            # (dx_world, dy_world) from the vehicle's current world position.
            # Vehicle yaw (vehicle_yaw_rad_map_frame) is from X-East, CCW.
            # If vehicle points East (yaw=0), p_vehicle_local.x is East offset, p_vehicle_local.y is SOUTH offset (CARLA Y=South)
            # If vehicle points North (yaw=pi/2), p_vehicle_local.x is North offset, p_vehicle_local.y is EAST offset.
            
            # Rotate (p_vehicle_local.x, p_vehicle_local.y) by vehicle_yaw_rad_map_frame
            dx_world = p_vehicle_local.x * math.cos(vehicle_yaw_rad_map_frame) - p_vehicle_local.y * math.sin(vehicle_yaw_rad_map_frame)
            dy_world = p_vehicle_local.x * math.sin(vehicle_yaw_rad_map_frame) + p_vehicle_local.y * math.cos(vehicle_yaw_rad_map_frame)

            # 4. Convert world-oriented offsets to screen offsets (map North is Pygame screen Up)
            screen_x_offset = dx_world * pixels_per_meter  # East = Right
            screen_y_offset = -dy_world * pixels_per_meter # South = Down, so -Y for North up

            screen_x = center_x_screen + int(screen_x_offset)
            screen_y = center_y_screen + int(screen_y_offset)

            # Color based on velocity
            velocity = detection.velocity
            if velocity < -0.1: color = (255, 120, 120) # Approaching (brighter red)
            elif velocity > 0.1: color = (120, 120, 255) # Receding (brighter blue)
            else: color = (230, 230, 230) # Static (brighter grey)
            
            # Draw larger circles for radar detections
            if 0 <= screen_x < surface_size[0] and 0 <= screen_y < surface_size[1]:
                pygame.draw.circle(radar_surface, color, (screen_x, screen_y), 5) # Increased from 4 to 5
        
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
                surface_to_display = self._create_lidar_surface(raw_sensor_data_item, current_display_size, lidar_sensor_range_from_env, debug_info_from_env)
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
                processed_image_for_display = raw_sensor_data_item
                if current_view_key == 'depth_camera' or current_view_key == 'display_depth_camera':
                    processed_image_for_display.convert(carla.ColorConverter.LogarithmicDepth)
                elif current_view_key == 'semantic_camera' or current_view_key == 'display_semantic_camera':
                    processed_image_for_display.convert(carla.ColorConverter.CityScapesPalette)
                
                img_bgra = np.array(processed_image_for_display.raw_data).reshape(
                    (processed_image_for_display.height, processed_image_for_display.width, 4))
                img_rgb_h_w_c = img_bgra[:, :, :3][:, :, ::-1] # Ensure it's HWC, RGB
                temp_surface = pygame.surfarray.make_surface(img_rgb_h_w_c)
                
                # Consistently rotate all carla.Image derived views by -90 degrees (clockwise 90)
                # This seems to be required to correct their orientation for Pygame display.
                temp_surface = pygame.transform.rotate(temp_surface, -90)
                # Flip horizontally to correct mirrored view
                temp_surface = pygame.transform.flip(temp_surface, True, False)

                surface_to_display = temp_surface
            # else: other sensor types not yet handled for direct full-screen viz

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

        pygame.display.flip()
        return True

    def close(self):
        if pygame.get_init(): 
            pygame.quit()
        self.is_active = False
        # print("PygameVisualizer closed.") # Optional: log close 