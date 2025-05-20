import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict # For creating dicts for model input
from typing import Dict # For type hinting
import os # For os.makedirs and os.path.join
import logging # For logging save/load actions

from .base_agent import BaseAgent
# from ..models.dqn_model import DQNModel # Assuming you have a DQNModel class
# from ..replay_buffers.uniform_replay_buffer import UniformReplayBuffer # Or any other buffer

logger = logging.getLogger(__name__) # Logger for this module

class DQNAgent(BaseAgent):
    def __init__(self, observation_space, action_space, model, replay_buffer,
                 lr=1e-4, gamma=0.99, tau=1e-3, batch_size=64, update_every=4,
                 device='cpu'):
        super().__init__(observation_space, action_space)

        self.model_constructor_args = model.get_constructor_args() # Store for reconstruction if needed
        self.q_network = model.to(device) # Renamed from self.model for clarity
        self.target_q_network = type(model)(*self.model_constructor_args).to(device) # Reconstruct target
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = replay_buffer
        
        self.lr = lr
        self.gamma = gamma # Discount factor
        self.tau = tau # For soft update of target parameters
        self.batch_size = batch_size
        self.update_every = update_every # How often to update the network
        self.device = device

        self.timesteps = 0 # Counter for update_every

    def _observation_to_tensor_dict(self, observation_dict: dict) -> Dict[str, torch.Tensor]:
        """Converts a dictionary of NumPy observation arrays to a dictionary of PyTorch tensors."""
        tensor_dict = OrderedDict()
        for key, value in observation_dict.items():
            # Ensure value is numpy array first if it isn't (e.g. raw list from some custom env)
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            tensor_dict[key] = torch.from_numpy(value).float().unsqueeze(0).to(self.device)
        return tensor_dict

    def select_action(self, observation: dict, epsilon=0.0):
        """Selects an action using an epsilon-greedy policy.
        Args:
            observation (dict): A dictionary where keys are sensor names and values are NumPy arrays.
            epsilon (float): Epsilon for epsilon-greedy exploration.
        """
        self.timesteps += 1
        if random.random() > epsilon:
            # Exploit: select the action with max Q-value
            obs_tensor_dict = self._observation_to_tensor_dict(observation)
            
            self.q_network.eval()  # Set model to evaluation mode
            with torch.no_grad():
                action_values = self.q_network(obs_tensor_dict)
            self.q_network.train()  # Set model back to train mode
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Explore: select a random action
            return self.action_space.sample() # Assumes gymnasium.spaces.Discrete

    def learn(self, experiences_dict):
        """Update value parameters using given batch of experience tuples.
        Q_targets = r + γ * max_a Q_target(s', a)
        delta = Q_targets - Q(s,a)
        loss = MSE(delta)
        
        experiences_dict: A dictionary containing batched states, actions, rewards, next_states, dones.
                          states and next_states are dictionaries of tensors.
        """
        states_dict = experiences_dict['states']
        actions = experiences_dict['actions']
        rewards = experiences_dict['rewards']
        next_states_dict = experiences_dict['next_states']
        dones = experiences_dict['dones']

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.target_q_network(next_states_dict).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.q_network(states_dict).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0) # Optional: Gradient clipping
        self.optimizer.step()

        self._soft_update(self.q_network, self.target_q_network, self.tau)
        
        return loss.item()  # Return loss for logging

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
    
    def save(self, directory_path, model_name="dqn_agent"):
        """Saves the Q-network, target Q-network, and optimizer states."""
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                logger.info(f"Created directory for saving models: {directory_path}")

            q_network_path = os.path.join(directory_path, f"{model_name}_q_network.pth")
            target_q_network_path = os.path.join(directory_path, f"{model_name}_target_q_network.pth")
            optimizer_path = os.path.join(directory_path, f"{model_name}_optimizer.pth")
            # Optional: Save model constructor args if they are complex or not easily reproducible
            # constructor_args_path = os.path.join(directory_path, f"{model_name}_constructor_args.pkl")

            torch.save(self.q_network.state_dict(), q_network_path)
            torch.save(self.target_q_network.state_dict(), target_q_network_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            # import pickle
            # with open(constructor_args_path, 'wb') as f:
            #    pickle.dump(self.model_constructor_args, f)
            
            logger.info(f"Saved DQN agent models to {directory_path} with prefix '{model_name}'")
        except Exception as e:
            logger.error(f"Error saving DQN agent models: {e}", exc_info=True)

    def load(self, directory_path, model_name="dqn_agent", map_location=None):
        """Loads the Q-network, target Q-network, and optimizer states."""
        try:
            q_network_path = os.path.join(directory_path, f"{model_name}_q_network.pth")
            target_q_network_path = os.path.join(directory_path, f"{model_name}_target_q_network.pth")
            optimizer_path = os.path.join(directory_path, f"{model_name}_optimizer.pth")
            # constructor_args_path = os.path.join(directory_path, f"{model_name}_constructor_args.pkl")

            if not os.path.exists(q_network_path):
                logger.error(f"Q-Network model file not found at {q_network_path}")
                return False # Indicate failure

            # Load constructor args if saved and needed for model reconstruction
            # This DQNAgent already reconstructs target_model using stored args,
            # so primary model reconstruction might be handled by whoever creates DQNAgent instance.
            # import pickle
            # if os.path.exists(constructor_args_path):
            #     with open(constructor_args_path, 'rb') as f:
            #         loaded_constructor_args = pickle.load(f)
            #         # Potentially re-initialize self.q_network here if its structure could change
            #         # self.q_network = type(self.q_network)(*loaded_constructor_args).to(self.device)

            device_to_load_on = map_location if map_location else self.device

            self.q_network.load_state_dict(torch.load(q_network_path, map_location=device_to_load_on))
            if os.path.exists(target_q_network_path):
                self.target_q_network.load_state_dict(torch.load(target_q_network_path, map_location=device_to_load_on))
            else: # Fallback: copy from loaded q_network if target is missing (e.g. older save)
                self.target_q_network.load_state_dict(self.q_network.state_dict())
                logger.warning(f"Target Q-Network model file not found at {target_q_network_path}. Copied from Q-Network.")
            
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=device_to_load_on))
            else:
                logger.warning(f"Optimizer state file not found at {optimizer_path}. Optimizer not loaded.")

            self.q_network.train() # Default to train mode after loading for continued training
            self.target_q_network.eval() # Target network is always in eval mode except during direct state_dict load
            
            logger.info(f"Loaded DQN agent models from {directory_path} with prefix '{model_name}'")
            return True # Indicate success
        except Exception as e:
            logger.error(f"Error loading DQN agent models: {e}", exc_info=True)
            return False # Indicate failure

    def _batch_experiences(self, experiences_list):
        """Converts a list of Experience namedtuples (where state/next_state are dicts)
           into a dictionary of batched tensors, ready for the learning step.
        """
        batch = OrderedDict()
        # Initialize lists for each component of the experience
        states_dict_list = [e.state for e in experiences_list if e is not None]
        actions_list = [e.action for e in experiences_list if e is not None]
        rewards_list = [e.reward for e in experiences_list if e is not None]
        next_states_dict_list = [e.next_state for e in experiences_list if e is not None]
        dones_list = [e.done for e in experiences_list if e is not None]

        if not states_dict_list: # Should not happen if buffer size > batch_size check is done
            return None

        # Batch states and next_states (dictionaries of observations)
        # Get all unique sensor keys from the first valid state dictionary
        sample_obs_keys = states_dict_list[0].keys()
        
        batched_states = OrderedDict()
        for key in sample_obs_keys:
            # Stack all numpy arrays for this key from each state dictionary in the list
            batched_states[key] = torch.from_numpy(np.array([s[key] for s in states_dict_list])).float().to(self.device)

        batched_next_states = OrderedDict()
        for key in sample_obs_keys: # Assume next_states have the same keys
            batched_next_states[key] = torch.from_numpy(np.array([s[key] for s in next_states_dict_list])).float().to(self.device)
        
        batch['states'] = batched_states
        batch['next_states'] = batched_next_states
        batch['actions'] = torch.from_numpy(np.array(actions_list)).long().unsqueeze(1).to(self.device)
        batch['rewards'] = torch.from_numpy(np.array(rewards_list)).float().unsqueeze(1).to(self.device)
        batch['dones'] = torch.from_numpy(np.array(dones_list).astype(np.uint8)).float().unsqueeze(1).to(self.device)
        
        return batch

    def step_experience_and_learn(self):
        loss = None
        if len(self.replay_buffer) > self.batch_size and self.timesteps % self.update_every == 0:
            experiences_list = self.replay_buffer.sample(self.batch_size)
            
            batched_experiences_dict = self._batch_experiences(experiences_list)
            
            if batched_experiences_dict:
                loss = self.learn(batched_experiences_dict)
        return loss 