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

        print("HUD: Attempting to load main font...")
        try:
            font_name_os_specific = 'courier' if os.name == 'nt' else 'mono'
            preferred_fonts_available = [x for x in pygame.font.get_fonts() if x is not None]
            font_to_load = None # Stores the name/path of the successfully loaded main font source
            for pref_name in font_preferences:
                if pref_name is None: continue
                try:
                    matched_font = pygame.font.match_font(pref_name)
                    if matched_font:
                        self.font = pygame.font.Font(matched_font, base_font_size)
                        actual_font_name_main = f"{pref_name} (matched as {os.path.basename(matched_font) if matched_font else 'N/A'})"
                        font_to_load = matched_font; break
                    elif os.path.exists(pref_name):
                         self.font = pygame.font.Font(pref_name, base_font_size)
                         actual_font_name_main = f"{os.path.basename(pref_name)} (direct path)"
                         font_to_load = pref_name; break
                except (pygame.error, OSError): continue
            if not self.font:
                system_fonts_lower = [f.lower() for f in preferred_fonts_available if f]
                chosen_font_name = None
                if 'ubuntumono' in system_fonts_lower: chosen_font_name = 'ubuntumono' 
                elif 'ubuntu mono' in system_fonts_lower: chosen_font_name = 'ubuntu mono'
                else:
                    fonts_with_os_specific_name = [f for f in preferred_fonts_available if font_name_os_specific in f.lower()]
                    if fonts_with_os_specific_name: chosen_font_name = fonts_with_os_specific_name[0]
                    elif preferred_fonts_available: chosen_font_name = preferred_fonts_available[0]
                if chosen_font_name:
                    matched_font_path = pygame.font.match_font(chosen_font_name)
                    if matched_font_path:
                        self.font = pygame.font.Font(matched_font_path, base_font_size)
                        actual_font_name_main = f"{chosen_font_name} (matched as {os.path.basename(matched_font_path)})"
                        font_to_load = matched_font_path # Store the path used
                    else: raise pygame.error("match_font failed for chosen main font")
                else: raise pygame.error("No suitable main font found by name")
            actual_font_size_main = base_font_size
            print(f"  HUD: Successfully loaded main font: '{actual_font_name_main}' with size {actual_font_size_main}")
            self.line_height = actual_font_size_main + 6
        except pygame.error:
            pygame.font.init() 
            self.font = pygame.font.Font(None, fallback_font_size)
            actual_font_size_main = fallback_font_size
            self.line_height = actual_font_size_main + 4
            actual_font_name_main = "Pygame Default (Fallback)"
            print(f"  HUD: Fell back to main font: '{actual_font_name_main}' with size {actual_font_size_main}")
        
        # --- Notification Font Loading (mirroring main font logic but for smaller size) ---
        self.notification_font = None
        actual_font_name_notif = "Unknown"
        target_notif_font_size = int(base_font_size * 0.9) 
        fallback_notif_font_size = int(fallback_font_size * 0.9)
        if target_notif_font_size < 8: target_notif_font_size = 8 # Ensure minimum practical size
        if fallback_notif_font_size < 8: fallback_notif_font_size = 8
        actual_loaded_notif_font_size = 0

        print("HUD: Attempting to load notification font...")
        try:
            # Try to use the same source that worked for the main font, but at the smaller size
            if font_to_load: # If a specific file/matched name was successfully used for main font
                try:
                    self.notification_font = pygame.font.Font(font_to_load, target_notif_font_size)
                    actual_font_name_notif = f"{actual_font_name_main.split(' (')[0]} (source same as main)" # Use original preferred name part
                    actual_loaded_notif_font_size = target_notif_font_size
                    print(f"  HUD: Successfully loaded notification font (from main font source): '{actual_font_name_notif}' with size {actual_loaded_notif_font_size}")
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
                print(f"  HUD: Successfully loaded notification font: '{actual_font_name_notif}' with size {actual_loaded_notif_font_size}")
        
        except pygame.error:
            pygame.font.init()
            self.notification_font = pygame.font.Font(None, fallback_notif_font_size)
            actual_loaded_notif_font_size = fallback_notif_font_size
            actual_font_name_notif = "Pygame Default (Fallback)"
            print(f"  HUD: Fell back to notification font: '{actual_font_name_notif}' with size {actual_loaded_notif_font_size}")

        if self.notification_font is None: # Ultimate fallback for notification font
             self.notification_font = self.font # Use main font if all else failed for notification
             actual_font_name_notif = actual_font_name_main # Copy main font details
             actual_loaded_notif_font_size = actual_font_size_main 
             print("  HUD: Notification font defaulted to main HUD font.")
        
        # Print final loaded fonts (can be enabled for debugging)
        if actual_font_name_main != "Unknown": print(f"HUD Main Font: '{actual_font_name_main}', Size: {actual_font_size_main}")
        if actual_font_name_notif != "Unknown": print(f"HUD Notification Font: '{actual_font_name_notif}', Size: {actual_loaded_notif_font_size}")

        self.section_break_keys = [
            "Vehicle Model", 
            "Speed (km/h)",
            "RL Action",
            "Dist to Goal (m)",
            "--- Sensors ---"  # Add the new key for sensor section break
        ]
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
        y_offset_initial = 10; x_offset_initial = 10
        key_value_gap = 10; box_text_gap = 5

        all_keys_for_sizing = ["Client FPS"] + list(debug_info.keys())
        max_key_len_px = 0
        for key_text in all_keys_for_sizing:
            key_surface = self.font.render(f"{key_text}:", True, self.text_color)
            if key_surface.get_width() > max_key_len_px:
                max_key_len_px = key_surface.get_width()
        
        widest_value_component_px = 0
        temp_combined_info = {"Client FPS": f"{clock.get_fps():.2f}", **debug_info}
        for key, value in temp_combined_info.items():
            current_value_width = 0
            if key == "Traffic Light" and isinstance(value, carla.TrafficLightState):
                val_str_tl_debug = str(value).split('.')[-1]
                current_value_width = self.boolean_box_size + box_text_gap + self.font.size(val_str_tl_debug)[0]
            elif key in ["Collision", "Proximity Penalty"] and isinstance(value, bool):
                current_value_width = self.boolean_box_size 
            else: 
                val_str_std_debug = str(value)
                current_value_width = self.font.size(val_str_std_debug)[0]
            if current_value_width > widest_value_component_px:
                widest_value_component_px = current_value_width
        content_width = max_key_len_px + key_value_gap + widest_value_component_px 
        bg_width = content_width + 20
        num_main_hud_lines = 1 
        processed_lines_for_sizing = 0
        for key_iter in debug_info.keys():
            if key_iter in self.section_break_keys and processed_lines_for_sizing > 0:
                num_main_hud_lines +=1 
            num_main_hud_lines +=1
            processed_lines_for_sizing +=1
        num_spacers = sum(1 for key_iter in debug_info if key_iter in self.section_break_keys and key_iter != list(debug_info.keys())[0] if processed_lines_for_sizing > 0) 
        bg_height = (num_main_hud_lines - num_spacers) * self.line_height + num_spacers * (self.line_height // 2) + 10

        if bg_width > 0 and bg_height > 0 and self.font:
            try:
                bg_surface = pygame.Surface((bg_width, bg_height), pygame.SRCALPHA)
                bg_surface.fill((*self.background_color, self.background_alpha))
                surface.blit(bg_surface, (x_offset_initial - 5, y_offset_initial - 5))
            except pygame.error as e: print(f"HUD Error: background surface: {e}")

        y_offset = y_offset_initial
        key_s = self.font.render("Client FPS:", True, self.text_color)
        val_s = self.font.render(f"{clock.get_fps():.2f}", True, self.text_color)
        surface.blit(key_s, (x_offset_initial, y_offset))
        surface.blit(val_s, (x_offset_initial + max_key_len_px + key_value_gap, y_offset))
        y_offset += self.line_height
        first_item_after_fps = True
        for key, value in debug_info.items():
            if key in self.section_break_keys and not first_item_after_fps:
                y_offset += self.line_height // 2  
            key_render_surface = self.font.render(f"{key}:", True, self.text_color)
            surface.blit(key_render_surface, (x_offset_initial, y_offset))
            current_x_for_value = x_offset_initial + max_key_len_px + key_value_gap
            if key == "Traffic Light":
                if isinstance(value, carla.TrafficLightState):
                    color = self.traffic_light_colors.get(value, self.traffic_light_colors[carla.TrafficLightState.Unknown])
                    pygame.draw.rect(surface, color, (current_x_for_value, y_offset + (self.line_height - self.boolean_box_size)//2, self.boolean_box_size, self.boolean_box_size))
                    val_str = str(value).split('.')[-1] 
                    if val_str:
                        val_render_surface = self.font.render(val_str, True, self.text_color)
                        surface.blit(val_render_surface, (current_x_for_value + self.boolean_box_size + box_text_gap, y_offset))
                else: 
                    val_str = str(value)
                    val_render_surface = self.font.render(val_str, True, self.text_color)
                    surface.blit(val_render_surface, (current_x_for_value, y_offset))
            elif key in ["Collision", "Proximity Penalty"] and isinstance(value, bool):
                box_color = self.boolean_box_true_color if value else self.boolean_box_false_color
                pygame.draw.rect(surface, box_color, (current_x_for_value, y_offset + (self.line_height - self.boolean_box_size)//2, self.boolean_box_size, self.boolean_box_size))
            else:
                val_str = str(value) 
                val_render_surface = self.font.render(val_str, True, self.text_color)
                surface.blit(val_render_surface, (current_x_for_value, y_offset))
            y_offset += self.line_height
            first_item_after_fps = False
            if y_offset > current_display_size[1] - self.line_height * (len(self.notifications) + 3): break 
        
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
            surface.blit(notif_surface, (x_offset_initial, notification_y_offset))
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