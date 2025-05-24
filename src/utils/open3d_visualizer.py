import open3d as o3d
import numpy as np
import carla # For type hints
import logging
from matplotlib import cm # For colormaps
import threading # Added for threading
import queue # Added for thread-safe communication
import time # For sleep in thread loop

logger = logging.getLogger(__name__)

# Colormap for LIDAR intensity, similar to CARLA examples
VIRIDIS_CMAP = np.array(cm.get_cmap('plasma').colors)
VID_RANGE_CMAP = np.linspace(0.0, 1.0, VIRIDIS_CMAP.shape[0])

# Semantic Label Colors for Open3D (0-1 range)
O3D_LABEL_COLORS = np.array([
    [0, 0, 0],       # 0 Unlabeled
    [0.27, 0.27, 0.27],    # 1 Building (70/255)
    [0.39, 0.15, 0.15],   # 2 Fences (100/255, 40/255, 40/255)
    [0.21, 0.35, 0.31],    # 3 Other
    [0.86, 0.07, 0.23],   # 4 Pedestrian
    [0.60, 0.60, 0.60], # 5 Pole
    [0.61, 0.91, 0.19],  # 6 RoadLines
    [0.50, 0.25, 0.50],  # 7 Road
    [0.95, 0.13, 0.91],  # 8 Sidewalk
    [0.42, 0.55, 0.13],  # 9 Vegetation
    [0, 0, 0.55],     # 10 Vehicle
    [0.40, 0.40, 0.61], # 11 Wall
    [0.86, 0.86, 0],   # 12 TrafficSign
    [0.27, 0.51, 0.70],  # 13 Sky
    [0.31, 0, 0.31],     # 14 Ground
    [0.58, 0.39, 0.39], # 15 Bridge
    [0.90, 0.58, 0.54], # 16 RailTrack
    [0.70, 0.64, 0.70], # 17 GuardRail
    [0.98, 0.66, 0.11],  # 18 TrafficLight
    [0.43, 0.74, 0.62], # 19 Static
    [0.66, 0.47, 0.19],  # 20 Dynamic
    [0.17, 0.23, 0.58],   # 21 Water
    [0.56, 0.66, 0.39], # 22 Terrain
    [0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5],[0.25,0,0],[0,0.25,0],[0,0,0.25] # Placeholders
])

class Open3DLidarVisualizer:
    def __init__(self, window_name="CARLA Open3D LIDAR", width=1280, height=720):
        """
        Initializes the Open3D LIDAR Visualizer.
        Note: This visualizer runs its own window and event loop.
        Make sure 'open3d' is installed (e.g., pip install open3d).
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        
        self.vis = None # Will be created in the thread
        self.point_cloud_o3d = o3d.geometry.PointCloud()
        self.is_initialized_in_thread = False # Renamed for clarity
        self.view_control = None
        self._active = False
        self.camera_distance = 15.0
        self.camera_pitch_deg = -30.0
        self.camera_yaw_offset_deg = 0.0

        self._vis_thread = None
        self._data_queue = queue.Queue(maxsize=1) # Store only the latest data
        self._stop_event = threading.Event()

        # logger.info("Open3D LIDAR Visualizer created. Call activate() to show window.") # Log moved to activate

    def _visualization_thread_target(self):
        """Target function for the Open3D visualization thread."""
        try:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.window_name, width=self.width, height=self.height, visible=True)
            render_option = self.vis.get_render_option()
            render_option.background_color = np.asarray([0.05, 0.05, 0.05])
            render_option.point_size = 1.5
            logger.info("Open3D Lidar window created in thread.")
            self.is_initialized_in_thread = False # Reset for add_geometry

            while not self._stop_event.is_set():
                try:
                    lidar_data, ego_transform = self._data_queue.get(timeout=0.01) # Non-blocking with timeout
                    if lidar_data is None and ego_transform is None: # Sentinel for stopping
                        break
                    
                    points_world, colors = self._process_lidar_data(lidar_data)
                    if points_world is not None and colors is not None:
                        self.point_cloud_o3d.points = o3d.utility.Vector3dVector(points_world)
                        self.point_cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
                    else:
                        self.point_cloud_o3d.points = o3d.utility.Vector3dVector()
                        self.point_cloud_o3d.colors = o3d.utility.Vector3dVector()

                    if not self.is_initialized_in_thread:
                        self.vis.add_geometry(self.point_cloud_o3d)
                        self.is_initialized_in_thread = True
                    else:
                        self.vis.update_geometry(self.point_cloud_o3d)
                    
                    if ego_transform:
                        self._set_camera_view(ego_transform) # self.vis must be valid here

                except queue.Empty:
                    pass # No new data, just continue polling events
                except Exception as e:
                    logger.error(f"Error in O3D vis thread data processing: {e}", exc_info=True)
                    # Continue loop unless it's a critical error with self.vis

                if not self.vis.poll_events(): # Returns False if window closed
                    logger.info("Open3D window closed by user (detected by poll_events).")
                    break # Exit loop if window closed
                self.vis.update_renderer()
                # time.sleep(0.01) # Small sleep to yield thread if CPU usage is high

        except Exception as e:
            logger.error(f"Exception in Open3D visualization thread: {e}", exc_info=True)
        finally:
            if self.vis:
                self.vis.destroy_window()
            self._active = False # Ensure this is set when thread exits
            self.is_initialized_in_thread = False
            self.vis = None # Clear vis object
            logger.info("Open3D visualization thread finished.")

    def activate(self):
        if not self._active:
            self._stop_event.clear()
            self._vis_thread = threading.Thread(target=self._visualization_thread_target, daemon=True)
            self._vis_thread.start()
            self._active = True # Mark as active immediately, thread will manage window
            logger.info("Open3D LIDAR visualizer thread started.")
            return True
        return True # Already considered active

    def is_active(self) -> bool:
        return self._active

    def update_data(self, lidar_data: carla.SensorData, ego_transform: carla.Transform):
        if not self._active or lidar_data is None or ego_transform is None:
            return False # Return False if not active or data invalid, but thread handles actual updates
        
        try:
            # Non-blocking put, discard old data if queue is full (maxsize=1)
            self._data_queue.put_nowait((lidar_data, ego_transform))
        except queue.Full:
            try: # Try to clear and put again
                self._data_queue.get_nowait()
                self._data_queue.put_nowait((lidar_data, ego_transform))
            except queue.Empty:
                pass # Should not happen if Full was raised
            except queue.Full:
                logger.warning("O3D data queue still full after trying to clear. Skipping update.")
                return False
        return True # Signifies data was successfully queued

    def _process_lidar_data(self, lidar_data: carla.SensorData):
        """Processes CARLA LidarData (standard or semantic) to Open3D points and colors."""
        if lidar_data is None:
            return None, None

        points_world_np = None
        colors_rgb = None

        if isinstance(lidar_data, carla.LidarMeasurement):
            data = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4')))
            data = np.reshape(data, (int(data.shape[0] / 4), 4))
            points_local = data[:, :-1]
            intensity = data[:, -1]
            sensor_world_transform = lidar_data.transform
            world_coords = []
            for i in range(points_local.shape[0]):
                p_local = carla.Location(x=float(points_local[i,0]), y=float(points_local[i,1]), z=float(points_local[i,2]))
                p_world = sensor_world_transform.transform(p_local)
                world_coords.append([p_world.x, p_world.y, p_world.z])
            points_world_np = np.array(world_coords)
            intensity_col = np.clip(1.0 - np.log(intensity + 1e-6) / np.log(np.exp(-0.004 * 100.0)), 0.0, 1.0)
            colors_rgb = np.zeros((points_world_np.shape[0], 3))
            colors_rgb[:, 0] = np.interp(intensity_col, VID_RANGE_CMAP, VIRIDIS_CMAP[:, 0])
            colors_rgb[:, 1] = np.interp(intensity_col, VID_RANGE_CMAP, VIRIDIS_CMAP[:, 1])
            colors_rgb[:, 2] = np.interp(intensity_col, VID_RANGE_CMAP, VIRIDIS_CMAP[:, 2])
        
        elif isinstance(lidar_data, carla.SemanticLidarMeasurement):
            sensor_world_transform = lidar_data.transform
            world_coords = []
            tags = []
            for detection in lidar_data:
                p_world = sensor_world_transform.transform(detection.point)
                world_coords.append([p_world.x, p_world.y, p_world.z])
                tags.append(detection.object_tag)
            points_world_np = np.array(world_coords)
            tags_np = np.array(tags, dtype=np.uint32)
            colors_rgb = O3D_LABEL_COLORS[tags_np % len(O3D_LABEL_COLORS)]
        
        else:
            logger.warning(f"Open3D Visualizer: Unknown LIDAR data type: {type(lidar_data)}")
            return None, None
            
        return points_world_np, colors_rgb

    def _set_camera_view(self, ego_transform: carla.Transform):
        if not self.vis: return # Called from thread, self.vis might be None if window creation failed or closing
        if not self.view_control:
            self.view_control = self.vis.get_view_control()
            if not self.view_control: return

        ego_location = ego_transform.location
        ego_forward_vector = ego_transform.get_forward_vector() # Normalized
        ego_up_vector = ego_transform.get_up_vector()         # Normalized
        # ego_right_vector = ego_transform.get_right_vector()   # Normalized

        # Target point for the camera to look at (e.g., slightly in front of the vehicle center)
        look_at_point = ego_location + ego_forward_vector * 2.0 # Look 2m ahead of vehicle center

        # Camera position: behind, above, and slightly offset if desired.
        # Start by going backwards from the vehicle along its forward vector.
        # Then go up.
        # Using fixed offsets relative to the vehicle's orientation.
        
        # Parameters for camera positioning (can be tuned)
        distance_behind = 10.0  # How far behind the car
        height_above = 5.0    # How high above the car
        # lateral_offset = 2.0 # Optional: offset to the side for a more angled view

        # Calculate camera position:
        # 1. Move backwards from ego location
        cam_pos = ego_location - distance_behind * ego_forward_vector
        # 2. Move upwards from that point
        cam_pos.z += height_above
        # 3. Optional: Add lateral offset using the vehicle's right vector
        # cam_pos += lateral_offset * ego_right_vector
        
        # Get the current camera parameters
        cam_params = self.view_control.convert_to_pinhole_camera_parameters()

        # Set the extrinsic parameters [eye, lookat, up]
        # Eye: camera_position
        # Lookat: look_at_point
        # Up: world up vector (0,0,1) for CARLA should be fine for a stable view, 
        # or ego_up_vector if you want camera to tilt with car roll (usually not for this type of view)
        world_up_vector = [0.0, 0.0, 1.0]

        # Construct the new extrinsic matrix for Open3D
        # Open3D's extrinsic is [R | t], where R is rotation, t is translation.
        # It represents camera pose in world. Inverse of view matrix.
        # Simpler: use set_look_at, set_up_vector, set_eye_position if available directly on params or view_control.
        # Open3D's set_lookat, set_front, set_up are for ViewControl, let's try to set extrinsic directly
        # if we want a fixed relative position to the car for that angled view.

        # Build view matrix (camera looking FROM eye TO look_at, with UP direction)
        # This is a standard lookAt matrix calculation.
        # Z_cam = normalize(eye - lookat)
        # X_cam = normalize(cross(up, Z_cam))
        # Y_cam = cross(Z_cam, X_cam)
        # Rotation_part = [X_cam, Y_cam, Z_cam]^T. Translation_part = -R * eye
        # Open3D extrinsic is [R | t] where t = camera_world_position, R = camera_orientation_in_world

        eye = np.array([cam_pos.x, cam_pos.y, cam_pos.z])
        lookat = np.array([look_at_point.x, look_at_point.y, look_at_point.z])
        up = np.array(world_up_vector) # Use world Z up for stability

        # Camera coordinate system (OpenGL style: Z away from viewer, Y up, X right)
        # Open3D usually aligns with this. Let's assume default intrinsic for now.
        # Forward vector of camera (from eye to lookat, then negated for camera Z axis)
        f = lookat - eye
        f = f / np.linalg.norm(f)
        # Right vector of camera
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        # Up vector of camera (recalculated to be orthogonal)
        u = np.cross(s, f) # Open3D up is often negative of this depending on convention
        # For Open3D, up vector for camera can often be just the world up if not rolling camera

        # Extrinsic matrix: first 3 rows, 4 columns [R | t]
        # R is orientation of camera in world. t is position of camera in world.
        extrinsic = np.identity(4)
        extrinsic[0, :3] = s # Camera X axis (right)
        extrinsic[1, :3] = u # Camera Y axis (up for Open3D default view)
        extrinsic[2, :3] = -f # Camera Z axis (pointing into the scene from camera)
        extrinsic[:3, 3] = eye # Camera position in world

        cam_params.extrinsic = extrinsic 
        self.view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        
        # Fallback using direct view control methods if the above is problematic
        # self.view_control.set_lookat([look_at_point.x, look_at_point.y, look_at_point.z])
        # self.view_control.set_up(world_up_vector) # Use world Z up
        # self.view_control.set_front(-f) # Camera's forward vector (direction it's pointing)
        # Zoom and other parameters can be set once during activate() or here if needed
        # self.view_control.set_zoom(0.7) # Adjust zoom for desired field of view

    def close(self):
        if self._active:
            self._stop_event.set() # Signal thread to stop
            # Put a sentinel on the queue to ensure the thread wakes up if blocked on get()
            try: self._data_queue.put_nowait((None, None)) 
            except queue.Full: pass # If full, thread will eventually timeout or see stop_event
            
            if self._vis_thread and self._vis_thread.is_alive():
                self._vis_thread.join(timeout=1.0) # Wait for thread to finish
                if self._vis_thread.is_alive():
                    logger.warning("Open3D visualization thread did not terminate in time.")
            self._active = False # Explicitly set inactive here
            self.is_initialized_in_thread = False
            self.view_control = None
            self.vis = None # Thread should have called destroy_window and cleared this
            logger.info("Open3D Lidar visualizer close requested, thread signaled/joined.")
        # Ensure queue is clear for next activation if any stale data
        while not self._data_queue.empty():
            try: self._data_queue.get_nowait()
            except queue.Empty: break

# Example of how this might be used (external to the class):
# if __name__ == '__main__':
#     # Dummy CARLA LidarData and Transform for testing
#     # This requires a running CARLA server if you want to spawn an actual sensor.
#     # For standalone test, you'd mock these.
#     class MockLidarData:
#         def __init__(self, num_points=1000):
#             # x, y, z, intensity
#             self.raw_data = (np.random.rand(num_points * 4) * 50 - 25).astype(np.float32).tobytes()
#             self.transform = carla.Transform(carla.Location(x=1,y=1,z=1), carla.Rotation())
#         def __len__(self):
#             return np.frombuffer(self.raw_data, dtype=np.float32).shape[0] // 4

#     o3d_vis = Open3DLidarVisualizer()
#     if o3d_vis.activate():
#         try:
#             for i in range(200):
#                 mock_lidar = MockLidarData(20000)
#                 # Simulate vehicle moving
#                 mock_ego_transform = carla.Transform(
#                     carla.Location(x=1 + i*0.1, y=1 + i*0.05, z=0.5),
#                     carla.Rotation(yaw = i * 2)
#                 )
#                 if not o3d_vis.update_data(mock_lidar, mock_ego_transform):
#                     print("Open3D window was closed by user.")
#                     break
#                 time.sleep(0.05) # Simulate game loop delay
#         finally:
#             o3d_vis.close()
#     else:
#         print("Could not activate Open3D window.") 