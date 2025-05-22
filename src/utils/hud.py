import pygame
import os # For OS-specific font names
import carla # For carla.TrafficLightState

class HUD:
    def __init__(self, font_preferences: list, base_font_size: int, fallback_font_size: int):
        """
        Initializes the HUD.
        Args:
            font_preferences (list): A list of font names to try, e.g., ['Consolas', 'monospace', None].
            base_font_size (int): The preferred font size.
            fallback_font_size (int): Font size to use if preferred fonts/size fail.
        """
        self.font = None
        self.line_height = 0
        self.base_font_size = base_font_size
        actual_font_name_main = "Unknown"
        actual_font_size_main = 0
        font_to_load = None

        print("HUD: Attempting to load main font...")
        try:
            self.font = pygame.font.Font(pygame.font.match_font(font_preferences[0]), base_font_size)
            actual_font_name_main = font_preferences[0]
            actual_font_size_main = base_font_size
        except:
            self.font = pygame.font.Font(None, fallback_font_size)
            actual_font_name_main = "DefaultPygame"
            actual_font_size_main = fallback_font_size
        self.line_height = actual_font_size_main + 6
        print(f"HUD: Successfully loaded main font: '{actual_font_name_main}' with size {actual_font_size_main}")
        
        self.notification_font = self.font # Default notification to main font for simplicity
        actual_font_name_notif = "Unknown"
        target_notif_font_size = int(base_font_size * 1.2) 
        fallback_notif_font_size = int(fallback_font_size * 1.2)
        if target_notif_font_size < 16: target_notif_font_size = 16 # Ensure minimum practical size
        if fallback_notif_font_size < 16: fallback_notif_font_size = 16
        actual_loaded_notif_font_size = 0

        print("HUD: Attempting to load notification font...")
        try:
            # Try to use the same source that worked for the main font, but at the smaller size
            if font_to_load: # If a specific file/matched name was successfully used for main font
                try:
                    self.notification_font = pygame.font.Font(font_to_load, target_notif_font_size)
                    actual_font_name_notif = f"{actual_font_name_main.split(' (')[0]} (source same as main)" # Use original preferred name part
                    actual_loaded_notif_font_size = target_notif_font_size
                    print(f"HUD: Successfully loaded notification font (from main font source): '{actual_font_name_notif}' with size {actual_loaded_notif_font_size}")
                except (pygame.error, OSError):
                    self.notification_font = None # Failed, will try full logic below
            
            if not self.notification_font: # If loading from main font source failed or wasn't available
                font_name_os_specific_notif = 'courier' if os.name == 'nt' else 'mono' # Same OS specific logic
                preferred_fonts_available_notif = [x for x in pygame.font.get_fonts() if x is not None]
                font_to_load_notif = None
                for pref_name_notif in font_preferences:
                    if pref_name_notif is None: continue
                    try:
                        matched_font_notif = pygame.font.match_font(pref_name_notif)
                        if matched_font_notif:
                            self.notification_font = pygame.font.Font(matched_font_notif, target_notif_font_size)
                            actual_font_name_notif = f"{pref_name_notif} (matched as {os.path.basename(matched_font_notif) if matched_font_notif else 'N/A'})"
                            font_to_load_notif = matched_font_notif; break
                        elif os.path.exists(pref_name_notif):
                            self.notification_font = pygame.font.Font(pref_name_notif, target_notif_font_size)
                            actual_font_name_notif = f"{os.path.basename(pref_name_notif)} (direct path)"
                            font_to_load_notif = pref_name_notif; break
                    except (pygame.error, OSError): continue
                
                if not self.notification_font:
                    system_fonts_lower_notif = [f.lower() for f in preferred_fonts_available_notif if f]
                    chosen_font_name_notif = None
                    if 'ubuntumono' in system_fonts_lower_notif: chosen_font_name_notif = 'ubuntumono' 
                    elif 'ubuntu mono' in system_fonts_lower_notif: chosen_font_name_notif = 'ubuntu mono'
                    else:
                        fonts_with_os_specific_name_notif = [f for f in preferred_fonts_available_notif if font_name_os_specific_notif in f.lower()]
                        if fonts_with_os_specific_name_notif: chosen_font_name_notif = fonts_with_os_specific_name_notif[0]
                        elif preferred_fonts_available_notif: chosen_font_name_notif = preferred_fonts_available_notif[0]
                    
                    if chosen_font_name_notif:
                        matched_font_path_notif = pygame.font.match_font(chosen_font_name_notif)
                        if matched_font_path_notif:
                            self.notification_font = pygame.font.Font(matched_font_path_notif, target_notif_font_size)
                            actual_font_name_notif = f"{chosen_font_name_notif} (matched as {os.path.basename(matched_font_path_notif)})"
                        else: raise pygame.error("match_font failed for chosen notification font")
                    else: raise pygame.error("No suitable notification font found by name")
                actual_loaded_notif_font_size = target_notif_font_size
                print(f"HUD: Successfully loaded notification font: '{actual_font_name_notif}' with size {actual_loaded_notif_font_size}")
        
        except pygame.error:
            pygame.font.init()
            self.notification_font = pygame.font.Font(None, fallback_notif_font_size)
            actual_loaded_notif_font_size = fallback_notif_font_size
            actual_font_name_notif = "Pygame Default (Fallback)"
            print(f"HUD: Fell back to notification font: '{actual_font_name_notif}' with size {actual_loaded_notif_font_size}")

        if self.notification_font is None: # Ultimate fallback for notification font
             self.notification_font = self.font # Use main font if all else failed for notification
             actual_font_name_notif = actual_font_name_main # Copy main font details
             actual_loaded_notif_font_size = actual_font_size_main 
             print("HUD: Notification font defaulted to main HUD font.")
        
        # Print final loaded fonts (can be enabled for debugging)
        if actual_font_name_main != "Unknown": print(f"HUD Main Font: '{actual_font_name_main}', Size: {actual_font_size_main}")
        if actual_font_name_notif != "Unknown": print(f"HUD Notification Font: '{actual_font_name_notif}', Size: {actual_loaded_notif_font_size}")

        # Define the HUD layout structure:
        # Each item is a list of keys for that block.
        # "Current View" and "Client FPS" are handled specially before these blocks.
        self.hud_layout = [
            ["Server FPS"], # Block 1 (after Current View & Client FPS)
            ["Vehicle Model", "Map", "Simulation Time", "Speed (km/h)", 
             "Location (X,Y,Z)", "Compass", "Throttle", "Steer", "Brake", "Gear"], #, "Acceleration", "Gyroscope"],
            ["Episode | Step", "Step Reward", "Episode Score", "Dist to Goal (m)", "Action"],
            ["Traffic Light", "Collision", "Proximity Penalty", "Term Reason"]
        ]
        # Key for Current View, handled separately at the top
        self.current_view_key_hud = "Current View"
        # Key for Client FPS, handled separately after Current View
        self.client_fps_key_hud = "Client FPS"

        self.text_color = (255, 255, 255)  
        self.notification_default_color = (255, 255, 0) 
        self.background_color = (0, 0, 0) 
        self.background_alpha = int(255 * 0.4) 
        self.notifications = []

        # Colors for traffic lights
        self.traffic_light_colors = {
            carla.TrafficLightState.Red: (255, 0, 0),
            carla.TrafficLightState.Yellow: (255, 255, 0),
            carla.TrafficLightState.Green: (0, 255, 0),
            carla.TrafficLightState.Off: (128, 128, 128), # Grey for off
            carla.TrafficLightState.Unknown: (100, 100, 100) # Dark grey for unknown
        }
        self.boolean_box_size = self.line_height - 8 # Smaller than line height
        self.boolean_box_true_color = (0, 200, 0) # Green for true
        self.boolean_box_false_color = (150, 0, 0) # Dark Red for false / (100,100,100) for grey outline

    def add_notification(self, text: str, duration_seconds: float = 3.0, color=None):
        if not self.font: return # Should not happen
        font_to_use = self.notification_font if self.notification_font else self.font
        text_color_to_use = color if color else self.notification_default_color
        
        text_surface = font_to_use.render(text, True, text_color_to_use)
        expiry_time = pygame.time.get_ticks() + int(duration_seconds * 1000)
        self.notifications.append({"surface": text_surface, "expiry": expiry_time, "alpha": 255, "original_color": text_color_to_use})
        # Limit number of notifications to avoid clutter
        max_notifications = 5
        if len(self.notifications) > max_notifications:
            self.notifications = self.notifications[-max_notifications:]

    def render(self, surface: pygame.Surface, clock: pygame.time.Clock, debug_info: dict):
        """
        Renders the HUD (text and background) onto the provided surface.
        Args:
            surface (pygame.Surface): The Pygame surface to draw on.
            clock (pygame.time.Clock): The Pygame clock for FPS calculation.
            debug_info (dict): A dictionary of information to display.
        """
        if not self.font: # Should not happen if __init__ is correct
            return

        current_display_size = surface.get_size()
        y_offset = 10; x_offset = 10
        key_value_gap = 15 
        box_text_gap = 5
        block_spacer_height = self.line_height // 2

        # --- Calculate dynamic widths for alignment --- 
        # Keys that will be rendered with a label
        keys_for_labels = [self.current_view_key_hud, self.client_fps_key_hud]
        for block in self.hud_layout:
            for key in block:
                if key in debug_info and not key.startswith('_'): # Only consider keys actually in debug_info
                    keys_for_labels.append(key)
        
        max_key_width = 0
        if self.font:
            for key_text in keys_for_labels:
                key_surf = self.font.render(f"{key_text}:", True, self.text_color)
                max_key_width = max(max_key_width, key_surf.get_width())
        
        # Calculate actual widest value component based on current debug_info
        widest_value_actual_px = 0
        if self.font:
            # Check Current View value width
            cv_val_str = str(debug_info.get(self.current_view_key_hud, "N/A"))
            cv_val_surf = self.font.render(cv_val_str, True, self.text_color)
            widest_value_actual_px = max(widest_value_actual_px, cv_val_surf.get_width())
            # Check Client FPS value width
            fps_val_str = f"{clock.get_fps():.1f}"
            fps_val_surf = self.font.render(fps_val_str, True, self.text_color)
            widest_value_actual_px = max(widest_value_actual_px, fps_val_surf.get_width())

            # Check values for all other keys in the layout
            for block_keys in self.hud_layout:
                for key_text in block_keys:
                    if key_text in debug_info and not key_text.startswith('_'):
                        value = debug_info[key_text]
                        current_val_width = 0
                        if key_text == "Traffic Light" and isinstance(value, carla.TrafficLightState):
                            val_str_display = str(value).split('.')[-1]
                            base_width = self.font.size(val_str_display)[0] if val_str_display else 0
                            current_val_width = self.boolean_box_size + box_text_gap + base_width
                        elif key_text in ["Collision", "Proximity Penalty"] and isinstance(value, str) and value in ["True", "False"]:
                            current_val_width = self.boolean_box_size 
                        else:
                            val_str_display = str(value)
                            current_val_width = self.font.size(val_str_display)[0]
                        widest_value_actual_px = max(widest_value_actual_px, current_val_width)

        content_width = max_key_width + key_value_gap + widest_value_actual_px
        bg_width = content_width + 20 # +20 for L/R padding

        # --- Determine background height --- 
        num_display_lines = 0
        # Current View
        num_display_lines += 1 
        # Spacer after Current View
        num_display_lines += 0.5 
        # Client FPS
        num_display_lines += 1 

        for block_idx, block_keys in enumerate(self.hud_layout):
            # Spacer before each block defined in hud_layout (including the first one after FPS)
            num_display_lines += 0.5 
            actual_keys_in_block = 0
            for key in block_keys:
                if key in debug_info and not key.startswith('_'):
                    actual_keys_in_block += 1
            if actual_keys_in_block == 0 and block_idx == 0 and self.hud_layout[0] == ["Server FPS", "Episode | Step"] : # Special case for first block if empty
                pass # Don't add spacer if first block is empty
            elif actual_keys_in_block == 0: # if a whole block is empty (no keys from it in debug_info), don't add its spacer nor lines
                num_display_lines -= 0.5 # Remove the pre-added spacer for this empty block
                continue
            num_display_lines += actual_keys_in_block
        
        bg_height = int(num_display_lines * self.line_height + 10) # +10 for top/bottom padding

        if bg_width > 0 and bg_height > 0 and self.font:
            try:
                bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
                bg_surface.fill((*self.background_color, self.background_alpha))
                surface.blit(bg_surface, (x_offset - 5, y_offset - 5))
            except pygame.error as e: print(f"HUD Error: background surface: {e}")

        # --- Render items based on defined layout --- 
        
        # 1. Current View
        key_text = self.current_view_key_hud
        value_str = str(debug_info.get(key_text, "N/A"))
        key_surf = self.font.render(f"{key_text}:", True, self.text_color)
        val_surf = self.font.render(value_str, True, self.text_color)
        surface.blit(key_surf, (x_offset, y_offset))
        surface.blit(val_surf, (x_offset + max_key_width + key_value_gap, y_offset))
        y_offset += self.line_height + block_spacer_height

        # 2. Client FPS
        key_text = self.client_fps_key_hud
        value_str = f"{clock.get_fps():.1f}"
        key_surf = self.font.render(f"{key_text}:", True, self.text_color)
        val_surf = self.font.render(value_str, True, self.text_color)
        surface.blit(key_surf, (x_offset, y_offset))
        surface.blit(val_surf, (x_offset + max_key_width + key_value_gap, y_offset))
        y_offset += self.line_height

        # 3. Iterate through defined blocks and their keys
        for block_idx, block_keys in enumerate(self.hud_layout):
            if block_idx > 0 or self.client_fps_key_hud in self.hud_layout[0]: # Add spacer if not the very first block after CurrentView/FPS special handling
                 y_offset += block_spacer_height
            
            for key_text in block_keys:
                if key_text.startswith('_'): continue # Should not be in hud_layout, but safeguard
                if key_text not in debug_info: continue # Skip if data not provided

                value = debug_info[key_text]
                key_surf = self.font.render(f"{key_text}:", True, self.text_color)
                surface.blit(key_surf, (x_offset, y_offset))
                current_val_x_start = x_offset + max_key_width + key_value_gap

                if key_text == "Traffic Light" and isinstance(value, carla.TrafficLightState):
                    color = self.traffic_light_colors.get(value, self.traffic_light_colors[carla.TrafficLightState.Unknown])
                    pygame.draw.rect(surface, color, (current_val_x_start, y_offset + (self.line_height - self.boolean_box_size)//2, self.boolean_box_size, self.boolean_box_size))
                    val_str_display = str(value).split('.')[-1] 
                    if val_str_display:
                        val_surf = self.font.render(val_str_display, True, self.text_color)
                        surface.blit(val_surf, (current_val_x_start + self.boolean_box_size + box_text_gap, y_offset))
                elif key_text in ["Collision", "Proximity Penalty"] and isinstance(value, str) and value in ["True", "False"]:
                    bool_val = True if value == "True" else False
                    box_color = self.boolean_box_true_color if bool_val else self.boolean_box_false_color
                    pygame.draw.rect(surface, box_color, (current_val_x_start, y_offset + (self.line_height - self.boolean_box_size)//2, self.boolean_box_size, self.boolean_box_size))
                else:
                    val_str_display = str(value)
                    val_surf = self.font.render(val_str_display, True, self.text_color)
                    surface.blit(val_surf, (current_val_x_start, y_offset))
                y_offset += self.line_height
                if y_offset > current_display_size[1] - self.line_height * 3 : break # Early exit if going off screen
            if y_offset > current_display_size[1] - self.line_height * 3 : break

        # --- Render Notifications --- 
        current_time = pygame.time.get_ticks()
        active_notifications = []
        for n in self.notifications:
            if n["expiry"] > current_time:
                notif_surface = n["surface"]
                time_left = n["expiry"] - current_time
                fade_duration = 1000 
                if time_left < fade_duration:
                    alpha = int(255 * (time_left / fade_duration))
                    notif_surface.set_alpha(alpha)
                else:
                    notif_surface.set_alpha(255)
                active_notifications.append(notif_surface)
        self.notifications = [n for n in self.notifications if n["expiry"] > current_time] 
        notification_y_offset = current_display_size[1] - 10 
        for i, notif_surface in enumerate(reversed(active_notifications)):
            notification_y_offset -= notif_surface.get_height() + 2 
            # Check against main HUD y_offset AND potential right panel width
            if notification_y_offset < y_offset + 10 : break 
            surface.blit(notif_surface, (x_offset, notification_y_offset))
            if i >= 4 : break

    def render_sensor_panel(self, surface: pygame.Surface, sensor_info: dict):
        if not self.font or not sensor_info:
            return

        y_offset_initial = 10
        padding = 10
        key_value_gap = 10

        # 1. Calculate max key and value widths for alignment and background sizing
        max_sensor_key_len_px = 0
        max_sensor_val_len_px = 0
        num_lines_for_bg_calc = 0

        for key, value in sensor_info.items():
            key_s = self.font.render(f"{key}:", True, self.text_color)
            if key_s.get_width() > max_sensor_key_len_px:
                max_sensor_key_len_px = key_s.get_width()
            
            val_s_width = 0
            if not (key == "Attached Sensors" and value == "---"): 
                val_s_render_temp = self.font.render(str(value), True, self.text_color)
                val_s_width = val_s_render_temp.get_width()
            if val_s_width > max_sensor_val_len_px:
                max_sensor_val_len_px = val_s_width
            num_lines_for_bg_calc +=1
            if key == "Attached Sensors" and value == "---": # Account for extra space after this header
                num_lines_for_bg_calc += 0.33 # Approximate extra third of a line for spacing

        panel_content_width = max_sensor_key_len_px + key_value_gap + max_sensor_val_len_px
        panel_bg_width = panel_content_width + 2 * padding
        panel_bg_height = (num_lines_for_bg_calc * self.line_height) + (2 * padding) 
        
        panel_x_offset = surface.get_width() - panel_bg_width - (padding - 5) 

        if panel_bg_width > 0 and panel_bg_height > 0:
            try:
                sensor_bg_surface = pygame.Surface((panel_bg_width, panel_bg_height), pygame.SRCALPHA)
                sensor_bg_surface.fill((*self.background_color, self.background_alpha))
                surface.blit(sensor_bg_surface, (panel_x_offset, y_offset_initial - 5))
            except pygame.error as e:
                print(f"HUD Error: Failed to create sensor panel background: {e}")

        # 3. Render the actual sensor text
        y_offset = y_offset_initial
        text_x_start = panel_x_offset + padding

        for key, value in sensor_info.items():
            key_render_surface = self.font.render(f"{key}:", True, self.text_color)
            surface.blit(key_render_surface, (text_x_start, y_offset))
            
            if not (key == "Attached Sensors" and value == "---"): 
                value_render_surface = self.font.render(str(value), True, self.text_color)
                surface.blit(value_render_surface, (text_x_start + max_sensor_key_len_px + key_value_gap, y_offset))
            
            y_offset += self.line_height
            if key == "Attached Sensors" and value == "---": # Add extra space after this header
                y_offset += self.line_height // 3

            if y_offset > surface.get_height() - self.line_height: 
                break
    # ... (close method) ... 