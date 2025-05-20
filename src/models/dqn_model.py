import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym # To access space types if needed for introspection
from collections import OrderedDict
from typing import Dict # For type hinting

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

        # --- RGB Camera Feature Extractor (CNN) ---
        if 'rgb_camera' in observation_space.spaces:
            rgb_shape = observation_space.spaces['rgb_camera'].shape # (C, H, W)
            c, h, w = rgb_shape
            self.feature_extractors['rgb_camera'] = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            # Calculate flattened size dynamically
            with torch.no_grad():
                dummy_rgb = torch.zeros(1, *rgb_shape)
                rgb_feature_size = self.feature_extractors['rgb_camera'](dummy_rgb).shape[1]
            current_concat_size += rgb_feature_size
            print(f"RGB Camera feature size: {rgb_feature_size}")

        # --- Depth Camera Feature Extractor (Simple CNN or MLP) ---
        if 'depth_camera' in observation_space.spaces:
            depth_shape = observation_space.spaces['depth_camera'].shape # (1, H, W)
            c, h, w = depth_shape
            # Using a simpler CNN for depth, or could be an MLP after flatten
            self.feature_extractors['depth_camera'] = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self._get_conv_out_size(depth_shape, 
                                                 kernels=[5,3], strides=[2,2], channels=[c,16,32]), 128),
                nn.ReLU()
            )
            current_concat_size += 128
            print(f"Depth Camera feature size: 128")


        # --- Semantic Segmentation Camera Feature Extractor ---
        if 'semantic_camera' in observation_space.spaces:
            semantic_shape = observation_space.spaces['semantic_camera'].shape # (1, H, W)
            c, h, w = semantic_shape
            # Similar to depth or even simpler
            self.feature_extractors['semantic_camera'] = nn.Sequential(
                nn.Conv2d(c, 16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self._get_conv_out_size(semantic_shape,
                                                 kernels=[5,3], strides=[2,2], channels=[c,16,32]), 128),
                nn.ReLU()
            )
            current_concat_size += 128
            print(f"Semantic Camera feature size: 128")

        # --- LIDAR Feature Extractor (1D Conv + MLP or just MLP) ---
        if 'lidar' in observation_space.spaces:
            lidar_shape = observation_space.spaces['lidar'].shape # (N_points, 3)
            # Option 1: Flatten and MLP
            # self.feature_extractors['lidar'] = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(lidar_shape[0] * lidar_shape[1], 256),
            #     nn.ReLU(),
            #     nn.Linear(256, 128),
            #     nn.ReLU()
            # )
            # current_concat_size += 128
            # Option 2: 1D Convolutions (treat points as sequence, features as channels)
            # Input shape for Conv1d: (Batch, Channels, Length) -> (Batch, 3, N_points)
            self.feature_extractors['lidar'] = nn.Sequential(
                Permute((0, 2, 1)), # (B, N_points, 3) -> (B, 3, N_points)
                nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                # Need to calculate output size of Conv1d sequence
                # For now, let's use a placeholder and then an MLP.
                # Or calculate it dynamically, assuming fixed lidar_shape[0]
                # fc_lidar_in = self._get_1d_conv_out_size(lidar_shape[0], kernels=[5,3], strides=[2,2]) * 64
                # nn.Linear(fc_lidar_in, 128), # This calculation is tricky without knowing lidar_shape[0] at init
                nn.LazyLinear(128), # Use LazyLinear to infer input size
                nn.ReLU()
            )
            current_concat_size += 128
            print(f"LIDAR feature size: 128 (using 1D Conv + LazyLinear)")


        # --- GNSS Feature Extractor (Simple MLP) ---
        if 'gnss' in observation_space.spaces:
            gnss_shape = observation_space.spaces['gnss'].shape[0] # (3,)
            self.feature_extractors['gnss'] = nn.Sequential(
                nn.Linear(gnss_shape, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            current_concat_size += 32
            print(f"GNSS feature size: 32")

        # --- IMU Feature Extractor (Simple MLP) ---
        if 'imu' in observation_space.spaces:
            imu_shape = observation_space.spaces['imu'].shape[0] # (6,)
            self.feature_extractors['imu'] = nn.Sequential(
                nn.Linear(imu_shape, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            current_concat_size += 32
            print(f"IMU feature size: 32")

        # --- RADAR Feature Extractor (Simple MLP after flatten) ---
        if 'radar' in observation_space.spaces:
            radar_shape = observation_space.spaces['radar'].shape # (M_detections, 4)
            self.feature_extractors['radar'] = nn.Sequential(
                nn.Flatten(),
                nn.Linear(radar_shape[0] * radar_shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            current_concat_size += 64
            print(f"Radar feature size: 64")

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

# Helper nn.Module for permuting tensor dimensions
class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

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