import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNModel(nn.Module):
    def __init__(self, state_shape, n_actions):
        """
        Initialize the DQN Model.
        Args:
            state_shape (tuple): Shape of the input state (e.g., (C, H, W) for images).
            n_actions (int): Number of possible actions.
        """
        super(DQNModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions

        # Assuming state_shape is (C, H, W)
        # Example for an 84x84 input image, similar to DeepMind's Nature DQN paper
        # Input channels C should be part of state_shape[0]
        c, h, w = state_shape

        # Convolutional layers
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the flattened size after conv layers
        # This is a common way to do it dynamically
        def conv_out_size(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        conv_h = conv_out_size(conv_out_size(conv_out_size(h, 8, 4), 4, 2), 3, 1)
        conv_w = conv_out_size(conv_out_size(conv_out_size(w, 8, 4), 4, 2), 3, 1)
        flattened_size = conv_h * conv_w * 64 # 64 is the number of output channels from conv3

        if flattened_size <= 0:
            raise ValueError(f"Calculated flattened size is not positive: {flattened_size}. Check input dimensions and conv layers.")

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor (batch_size, C, H, W).
        Returns:
            torch.Tensor: Q-values for each action (batch_size, n_actions).
        """
        # Ensure input is float (common practice)
        x = x.float() / 255.0 # Normalize pixel values if they are 0-255

        # Convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output from conv layers before passing to FC layers
        x = x.view(x.size(0), -1) 
        
        # Fully connected layers with ReLU activation for the hidden layer
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values

    def get_constructor_args(self):
        """
        Returns the arguments needed to reconstruct this model.
        Used by DQNAgent to create the target network.
        """
        return self.state_shape, self.n_actions

# Example Usage (for testing the model dimensions)
# if __name__ == '__main__':
#     # Assuming input is a batch of 1 image, 3 channels (RGB), 84x84 pixels
#     dummy_state_shape = (3, 84, 84) 
#     dummy_n_actions = 3
#     model = DQNModel(state_shape=dummy_state_shape, n_actions=dummy_n_actions)
#     print(model)

#     # Create a dummy input tensor (batch_size, C, H, W)
#     dummy_input = torch.randn(1, dummy_state_shape[0], dummy_state_shape[1], dummy_state_shape[2])
#     print("\nInput shape:", dummy_input.shape)
    
#     # Pass the dummy input through the model
#     output = model(dummy_input)
#     print("Output shape (Q-values per action):", output.shape)
#     assert output.shape == (1, dummy_n_actions)

#     args = model.get_constructor_args()
#     print("\nConstructor args:", args)
#     reconstructed_model = DQNModel(*args)
#     print("Reconstructed model structure is the same:", str(model) == str(reconstructed_model)) 