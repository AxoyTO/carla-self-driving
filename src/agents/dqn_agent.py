import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .base_agent import BaseAgent
# from ..models.dqn_model import DQNModel # Assuming you have a DQNModel class
# from ..replay_buffers.uniform_replay_buffer import UniformReplayBuffer # Or any other buffer

class DQNAgent(BaseAgent):
    def __init__(self, observation_space, action_space, model, replay_buffer,
                 lr=1e-4, gamma=0.99, tau=1e-3, batch_size=64, update_every=4,
                 device='cpu'):
        super().__init__(observation_space, action_space)

        self.model = model.to(device) # Q-Network
        self.target_model = type(model)(*model.get_constructor_args()).to(device) # Target Q-Network
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.replay_buffer = replay_buffer
        
        self.lr = lr
        self.gamma = gamma # Discount factor
        self.tau = tau # For soft update of target parameters
        self.batch_size = batch_size
        self.update_every = update_every # How often to update the network
        self.device = device

        self.timesteps = 0 # Counter for update_every

    def select_action(self, observation, epsilon=0.0):
        """Selects an action using an epsilon-greedy policy."""
        self.timesteps +=1
        if random.random() > epsilon:
            # Exploit: select the action with max Q-value
            # Ensure observation is a PyTorch tensor and on the correct device
            if not isinstance(observation, torch.Tensor):
                # Assuming observation is a numpy array, convert it
                # Add batch dimension if it's a single observation: (H, W, C) -> (1, H, W, C)
                # And permute if necessary for PyTorch (N, C, H, W) if using Conv2D
                # This depends heavily on your observation format and model input
                observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
                # Example for image: observation = observation.permute(0, 3, 1, 2) 
            
            self.model.eval() # Set model to evaluation mode
            with torch.no_grad():
                action_values = self.model(observation)
            self.model.train() # Set model back to train mode
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Explore: select a random action
            # Assumes action_space is compatible with random.choice or similar
            # If using gymnasium.spaces.Discrete, then self.action_space.sample() works
            return random.choice(np.arange(self.action_space.n)) # Example for Discrete space

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Q_targets = r + γ * max_a Q_target(s', a)
        delta = Q_targets - Q(s,a)
        loss = MSE(delta)
        
        experiences: (states, actions, rewards, next_states, dones) tuple of tensors
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.model(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self._soft_update(self.model, self.target_model, self.tau)
        
        return loss.item() # Return loss for logging

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    # Placeholder for model saving/loading (can be part of BaseAgent or here)
    # def save(self, filepath): 
    #     torch.save(self.model.state_dict(), filepath + "_qnetwork.pth")
    #     torch.save(self.target_model.state_dict(), filepath + "_target_qnetwork.pth")
    #     torch.save(self.optimizer.state_dict(), filepath + "_optimizer.pth")

    # def load(self, filepath):
    #     self.model.load_state_dict(torch.load(filepath + "_qnetwork.pth", map_location=self.device))
    #     self.target_model.load_state_dict(torch.load(filepath + "_target_qnetwork.pth", map_location=self.device))
    #     self.optimizer.load_state_dict(torch.load(filepath + "_optimizer.pth"))
    #     self.target_model.eval()

    def step_experience_and_learn(self):
        """Called at each step of the agent to add experience and learn if it's time."""
        # If enough samples are available in memory and it's time to update
        loss = None # Initialize loss to None
        if len(self.replay_buffer) > self.batch_size and self.timesteps % self.update_every == 0:
            experiences = self.replay_buffer.sample(self.batch_size)
            # Convert to PyTorch tensors, assuming experiences are tuples of numpy arrays
            # This might need adjustment based on how replay_buffer.sample() returns data
            
            # experiences is a list of Experience namedtuples
            # e.state is a numpy array of shape (C, H, W), e.g. (3, 84, 84)
            # We want to stack them into a single numpy array of shape (batch_size, C, H, W)
            
            states_list = [e.state for e in experiences if e is not None]
            actions_list = [e.action for e in experiences if e is not None]
            rewards_list = [e.reward for e in experiences if e is not None]
            next_states_list = [e.next_state for e in experiences if e is not None]
            dones_list = [e.done for e in experiences if e is not None]

            # Check if lists are empty (e.g. if all experiences were None, though unlikely with current buffer)
            if not states_list: 
                return loss # Cannot learn from empty batch

            states = torch.from_numpy(np.array(states_list)).float().to(self.device)
            actions = torch.from_numpy(np.array(actions_list)).long().unsqueeze(1).to(self.device) # Actions need to be (batch_size, 1) for gather
            rewards = torch.from_numpy(np.array(rewards_list)).float().unsqueeze(1).to(self.device) # Rewards need to be (batch_size, 1)
            next_states = torch.from_numpy(np.array(next_states_list)).float().to(self.device)
            dones = torch.from_numpy(np.array(dones_list).astype(np.uint8)).float().unsqueeze(1).to(self.device) # Dones need to be (batch_size, 1)
            
            # Example for image data that needs permutation: (N, H, W, C) -> (N, C, H, W)
            # Our states and next_states are already (N, C, H, W) from np.array([...])
            # states = states.permute(0, 3, 1, 2)
            # next_states = next_states.permute(0, 3, 1, 2)

            loss = self.learn((states, actions, rewards, next_states, dones))
        return loss # Return the loss (or None if not updated) 