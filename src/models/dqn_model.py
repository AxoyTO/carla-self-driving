import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym # To access space types if needed for introspection
from collections import OrderedDict
from typing import Dict, Tuple, Optional # Added Tuple, Optional

# Helper nn.Module for permuting tensor dimensions
class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class DQNModel(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, n_actions: int):
        """
        Initialize the DQN Model.
        Args:
            observation_space (gym.spaces.Dict): Observation space of the environment.
            n_actions (int): Number of possible actions.
        """
        super(DQNModel, self).__init__()
        self.observation_space = observation_space
        self.n_actions = n_actions

        self.feature_extractors = nn.ModuleDict()
        current_concat_size = 0

        # --- Create Feature Extractors for each modality ---
        if 'rgb_camera' in observation_space.spaces:
            extractor, size = self._create_rgb_extractor(observation_space.spaces['rgb_camera'].shape, name_prefix="FrontRGB")
            if extractor: self.feature_extractors['rgb_camera'] = extractor; current_concat_size += size
        
        if 'left_rgb_camera' in observation_space.spaces:
            extractor, size = self._create_rgb_extractor(observation_space.spaces['left_rgb_camera'].shape, name_prefix="LeftRGB")
            if extractor: self.feature_extractors['left_rgb_camera'] = extractor; current_concat_size += size

        if 'right_rgb_camera' in observation_space.spaces:
            extractor, size = self._create_rgb_extractor(observation_space.spaces['right_rgb_camera'].shape, name_prefix="RightRGB")
            if extractor: self.feature_extractors['right_rgb_camera'] = extractor; current_concat_size += size

        if 'rear_rgb_camera' in observation_space.spaces:
            extractor, size = self._create_rgb_extractor(observation_space.spaces['rear_rgb_camera'].shape, name_prefix="RearRGB")
            if extractor: self.feature_extractors['rear_rgb_camera'] = extractor; current_concat_size += size

        if 'depth_camera' in observation_space.spaces:
            extractor, size = self._create_depth_extractor(observation_space.spaces['depth_camera'].shape)
            if extractor: self.feature_extractors['depth_camera'] = extractor; current_concat_size += size

        if 'semantic_camera' in observation_space.spaces:
            extractor, size = self._create_semantic_extractor(observation_space.spaces['semantic_camera'].shape)
            if extractor: self.feature_extractors['semantic_camera'] = extractor; current_concat_size += size

        if 'lidar' in observation_space.spaces:
            extractor, size = self._create_lidar_extractor(observation_space.spaces['lidar'].shape)
            if extractor: self.feature_extractors['lidar'] = extractor; current_concat_size += size

        if 'gnss' in observation_space.spaces:
            extractor, size = self._create_gnss_extractor(observation_space.spaces['gnss'].shape[0])
            if extractor: self.feature_extractors['gnss'] = extractor; current_concat_size += size

        if 'imu' in observation_space.spaces:
            extractor, size = self._create_imu_extractor(observation_space.spaces['imu'].shape[0])
            if extractor: self.feature_extractors['imu'] = extractor; current_concat_size += size

        if 'radar' in observation_space.spaces:
            extractor, size = self._create_radar_extractor(observation_space.spaces['radar'].shape)
            if extractor: self.feature_extractors['radar'] = extractor; current_concat_size += size

        if 'semantic_lidar' in observation_space.spaces:
            extractor, size = self._create_semantic_lidar_extractor(observation_space.spaces['semantic_lidar'].shape)
            if extractor: self.feature_extractors['semantic_lidar'] = extractor; current_concat_size += size

        if current_concat_size == 0:
            raise ValueError("No feature extractors were created. Observation space might be empty or not recognized.")

        # Fully connected layers for Q-value estimation
        self.fc_q = nn.Sequential(
            nn.Linear(current_concat_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        print(f"Total concatenated feature size before final FC layers: {current_concat_size}")

    def _get_conv_out_size(self, shape_chw, kernels, strides, channels, padding=0):
        """ Helper to calculate output size of a series of Conv2d layers. """
        c, h, w = shape_chw
        for i in range(len(kernels)):
            h = (h + 2 * padding - (kernels[i] - 1) - 1) // strides[i] + 1
            w = (w + 2 * padding - (kernels[i] - 1) - 1) // strides[i] + 1
        return h * w * channels[-1]
    
    def _get_1d_conv_out_size(self, length, kernels, strides, padding=0):
        """ Helper to calculate output size of a series of Conv1d layers. """
        for i in range(len(kernels)):
            length = (length + 2 * padding - (kernels[i] - 1) - 1) // strides[i] + 1
        return length

    def _create_rgb_extractor(self, rgb_shape: Tuple[int, int, int], name_prefix: str = "RGB") -> Tuple[Optional[nn.Module], int]:
        c, h, w = rgb_shape
        extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, *rgb_shape)
            feature_size = extractor(dummy_input).shape[1]
        print(f"DQNModel: {name_prefix} Camera feature extractor created, output size: {feature_size}")
        return extractor, feature_size

    def _create_depth_extractor(self, depth_shape: Tuple[int, int, int]) -> Tuple[Optional[nn.Module], int]:
        c, h, w = depth_shape
        conv_out_features = self._get_conv_out_size(depth_shape, kernels=[5,3], strides=[2,2], channels=[c,16,32])
        output_feature_size = 128
        extractor = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out_features, output_feature_size),
            nn.ReLU()
        )
        print(f"DQNModel: Depth Camera feature extractor created, output size: {output_feature_size}")
        return extractor, output_feature_size

    def _create_semantic_extractor(self, semantic_shape: Tuple[int, int, int]) -> Tuple[Optional[nn.Module], int]:
        c, h, w = semantic_shape
        conv_out_features = self._get_conv_out_size(semantic_shape, kernels=[5,3], strides=[2,2], channels=[c,16,32])
        output_feature_size = 128
        extractor = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_out_features, output_feature_size),
            nn.ReLU()
        )
        print(f"DQNModel: Semantic Camera feature extractor created, output size: {output_feature_size}")
        return extractor, output_feature_size

    def _create_lidar_extractor(self, lidar_shape: Tuple[int, int]) -> Tuple[Optional[nn.Module], int]:
        # lidar_shape is (N_points, 3) or similar
        output_feature_size = 128
        extractor = nn.Sequential(
            Permute((0, 2, 1)), # (B, N_points, Channels) -> (B, Channels, N_points)
            nn.Conv1d(in_channels=lidar_shape[1], out_channels=32, kernel_size=5, stride=2), # Assuming lidar_shape[1] is num_features_per_point
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(output_feature_size),
            nn.ReLU()
        )
        print(f"DQNModel: LiDAR feature extractor created, output size: {output_feature_size} (using 1D Conv + LazyLinear)")
        return extractor, output_feature_size

    def _create_gnss_extractor(self, gnss_shape_flat: int) -> Tuple[Optional[nn.Module], int]:
        output_feature_size = 32
        extractor = nn.Sequential(
            nn.Linear(gnss_shape_flat, 64),
            nn.ReLU(),
            nn.Linear(64, output_feature_size),
            nn.ReLU()
        )
        print(f"DQNModel: GNSS feature extractor created, output size: {output_feature_size}")
        return extractor, output_feature_size

    def _create_imu_extractor(self, imu_shape_flat: int) -> Tuple[Optional[nn.Module], int]:
        output_feature_size = 32
        extractor = nn.Sequential(
            nn.Linear(imu_shape_flat, 64),
            nn.ReLU(),
            nn.Linear(64, output_feature_size),
            nn.ReLU()
        )
        print(f"DQNModel: IMU feature extractor created, output size: {output_feature_size}")
        return extractor, output_feature_size

    def _create_radar_extractor(self, radar_shape: Tuple[int, int]) -> Tuple[Optional[nn.Module], int]:
        # radar_shape is (M_detections, 4)
        output_feature_size = 64
        extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(radar_shape[0] * radar_shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, output_feature_size),
            nn.ReLU()
        )
        print(f"DQNModel: RADAR feature extractor created, output size: {output_feature_size}")
        return extractor, output_feature_size

    def _create_semantic_lidar_extractor(self, sem_lidar_shape: Tuple[int, int]) -> Tuple[Optional[nn.Module], int]:
        # sem_lidar_shape is (N_points, 4) where 4 = (x,y,z,object_tag)
        # We can use a similar 1D Conv architecture as for standard LIDAR,
        # but now with 4 input channels per point if object_tag is treated as a channel.
        # Or, embed object_tag and concatenate with (x,y,z) features.
        # For simplicity, let's treat all 4 as input channels to Conv1D.
        output_feature_size = 128 
        # Input to Conv1d: (Batch, Channels, Length) -> (Batch, 4, N_points)
        num_input_features_per_point = sem_lidar_shape[1] # Should be 4

        extractor = nn.Sequential(
            Permute((0, 2, 1)), # (B, N_points, Channels) -> (B, Channels, N_points)
            nn.Conv1d(in_channels=num_input_features_per_point, out_channels=32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(output_feature_size),
            nn.ReLU()
        )
        print(f"DQNModel: Semantic LIDAR feature extractor created, input_channels={num_input_features_per_point}, output_size: {output_feature_size}")
        return extractor, output_feature_size

    def forward(self, obs_dict: Dict[str, torch.Tensor]):
        """
        Forward pass through the network.
        Args:
            obs_dict (Dict[str, torch.Tensor]): Dictionary of observation tensors.
        Returns:
            torch.Tensor: Q-values for each action (batch_size, n_actions).
        """
        extracted_features = []

        if 'rgb_camera' in self.feature_extractors:
            # Normalize RGB image (assuming 0-255 range)
            rgb_tensor = obs_dict['rgb_camera'].float() / 255.0
            extracted_features.append(self.feature_extractors['rgb_camera'](rgb_tensor))

        # Process new cameras (Left, Right, Rear)
        for cam_key in ['left_rgb_camera', 'right_rgb_camera', 'rear_rgb_camera']:
            if cam_key in self.feature_extractors:
                cam_tensor = obs_dict[cam_key].float() / 255.0 # Normalize
                extracted_features.append(self.feature_extractors[cam_key](cam_tensor))

        if 'depth_camera' in self.feature_extractors:
            # Depth data is already normalized to [0,1] by the handler
            depth_tensor = obs_dict['depth_camera'].float() 
            extracted_features.append(self.feature_extractors['depth_camera'](depth_tensor))
        
        if 'semantic_camera' in self.feature_extractors:
            # Semantic data is class labels, ensure it's float for conv layers
            # Normalization might not be standard here, but conv layers expect float.
            semantic_tensor = obs_dict['semantic_camera'].float() # / 28.0 if normalization needed
            extracted_features.append(self.feature_extractors['semantic_camera'](semantic_tensor))

        if 'lidar' in self.feature_extractors:
            lidar_tensor = obs_dict['lidar'].float()
            extracted_features.append(self.feature_extractors['lidar'](lidar_tensor))

        if 'gnss' in self.feature_extractors:
            gnss_tensor = obs_dict['gnss'].float()
            extracted_features.append(self.feature_extractors['gnss'](gnss_tensor))

        if 'imu' in self.feature_extractors:
            imu_tensor = obs_dict['imu'].float()
            extracted_features.append(self.feature_extractors['imu'](imu_tensor))

        if 'radar' in self.feature_extractors:
            radar_tensor = obs_dict['radar'].float()
            extracted_features.append(self.feature_extractors['radar'](radar_tensor))
        
        if 'semantic_lidar' in self.feature_extractors:
            sem_lidar_tensor = obs_dict['semantic_lidar'].float() # Tags are already float from handler
            extracted_features.append(self.feature_extractors['semantic_lidar'](sem_lidar_tensor))

        # Concatenate all features
        # print([f.shape for f in extracted_features]) # Debug shapes
        concatenated_features = torch.cat(extracted_features, dim=1)
        
        q_values = self.fc_q(concatenated_features)
        return q_values

    def get_constructor_args(self):
        """
        Returns the arguments needed to reconstruct this model.
        Used by DQNAgent to create the target network.
        """
        return self.observation_space, self.n_actions

# Example Usage (for testing the model dimensions)
# if __name__ == '__main__':
#     from gymnasium import spaces
#     import numpy as np
#     # Define a dummy observation space similar to CarlaEnv
#     dummy_obs_space = spaces.Dict(OrderedDict([
#         ('rgb_camera', spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)),
#         ('depth_camera', spaces.Box(low=0, high=1.0, shape=(1, 84, 84), dtype=np.float32)),
#         ('semantic_camera', spaces.Box(low=0, high=28, shape=(1, 84, 84), dtype=np.uint8)),
#         ('lidar', spaces.Box(low=-np.inf, high=np.inf, shape=(720, 3), dtype=np.float32)), # N_points x 3
#         ('gnss', spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)),
#         ('imu', spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)),
#         ('radar', spaces.Box(low=-np.inf, high=np.inf, shape=(20, 4), dtype=np.float32)) # M_detections x 4
#     ]))
#     dummy_n_actions = 5
#     model = DQNModel(observation_space=dummy_obs_space, n_actions=dummy_n_actions)
#     print("\nModel Structure:")
#     print(model)

#     # Create a dummy input observation dictionary
#     batch_size = 2
#     dummy_input_dict = OrderedDict()
#     for key, space in dummy_obs_space.spaces.items():
#         dummy_input_dict[key] = torch.from_numpy(np.array([space.sample() for _ in range(batch_size)])).float()
#         if key == 'rgb_camera' or key == 'semantic_camera': # uint8 types
#             dummy_input_dict[key] = dummy_input_dict[key].to(torch.uint8).float() # Keep as float for model input after potential /255
    
#     print("\nInput shapes:")
#     for key, val in dummy_input_dict.items():
#         print(f"  {key}: {val.shape}")
    
#     # Pass the dummy input through the model
#     output = model(dummy_input_dict)
#     print("\nOutput shape (Q-values per action):", output.shape)
#     assert output.shape == (batch_size, dummy_n_actions)

#     args = model.get_constructor_args()
#     # print("\nConstructor args:", args) # This will be large due to obs_space
#     reconstructed_model = DQNModel(*args)
#     # Comparing str(model) might not work well due to LazyLinear resolved sizes etc.
#     # print("Reconstructed model structure is the same:", str(model) == str(reconstructed_model)) 