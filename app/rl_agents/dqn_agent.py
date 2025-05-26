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
from torch.cuda.amp import GradScaler, autocast

from .base_agent import BaseAgent
# from ..models.dqn_model import DQNModel # Assuming you have a DQNModel class
# from ..replay_buffers.uniform_replay_buffer import UniformReplayBuffer # Or any other buffer

logger = logging.getLogger(__name__) # Logger for this module

class DQNAgent(BaseAgent):
    def __init__(self, observation_space, action_space, model, replay_buffer,
                 lr=1e-4, gamma=0.99, tau=1e-3, batch_size=64, update_every=4,
                 device='cpu', use_mixed_precision=True, gradient_accumulation_steps=1,
                 max_grad_norm=1.0, double_dqn=True, dueling_dqn=False, use_n_step=False,
                 use_prioritized_replay=False, use_noisy_nets=False,
                 # C51 parameters
                 use_distributional_rl=False, n_atoms=51, v_min=-10.0, v_max=10.0,
                 use_lstm=False):
        super().__init__(observation_space, action_space)

        # Store the original observation_space object, not from args, as it might be a string there
        self._original_obs_space_for_target_net = observation_space 

        self.model_constructor_args = model.get_constructor_args()
        # Ensure the observation_space in constructor_args is the actual object
        # This is critical if model.get_constructor_args() might not preserve the live object state perfectly
        # or if it was serialized/deserialized elsewhere (though not the case here yet)
        self.model_constructor_args['observation_space'] = self._original_obs_space_for_target_net

        self.q_network = model.to(device)
        # Reconstruct target network using keyword arguments from the stored dict
        self.target_q_network = type(model)(**self.model_constructor_args).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        # Algorithm variants
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.use_n_step = use_n_step # Add N-step flag
        self.use_prioritized_replay = use_prioritized_replay # Store the flag
        self.use_noisy_nets = use_noisy_nets # Store this flag

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = replay_buffer
        
        self.lr = lr
        self.gamma = gamma # Discount factor
        self.tau = tau # For soft update of target parameters
        self.batch_size = batch_size
        self.update_every = update_every # How often to update the network
        self.device = device

        # Enhanced training features
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.accumulation_counter = 0
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.8, patience=10, verbose=True
        )
        
        # Training metrics
        self.training_metrics = {
            'loss_history': [],
            'gradient_norms': [],
            'learning_rates': []
        }

        self.timesteps = 0 # Counter for update_every

        self.use_distributional_rl = use_distributional_rl
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        if self.use_distributional_rl:
            self.delta_z = (v_max - v_min) / (n_atoms - 1)
            self.support = torch.linspace(v_min, v_max, n_atoms).to(device)

        self.use_lstm = use_lstm
        self.lstm_hidden_state = None # To store hidden state for select_action

    def _observation_to_tensor_dict(self, observation_dict: dict) -> Dict[str, torch.Tensor]:
        """Converts a dictionary of NumPy observation arrays to a dictionary of PyTorch tensors."""
        tensor_dict = OrderedDict()
        for key, value in observation_dict.items():
            # Ensure value is numpy array first if it isn't (e.g. raw list from some custom env)
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            tensor_dict[key] = torch.from_numpy(value).float().unsqueeze(0).to(self.device)
        return tensor_dict

    def reset_lstm_hidden_state(self, batch_size: int = 1):
        """Resets the LSTM hidden state to zeros. Called at the start of an episode for select_action."""
        if self.use_lstm:
            # Determine device from model parameters
            device = next(self.q_network.parameters()).device
            # Get lstm_num_layers and lstm_hidden_size from the model itself
            num_layers = self.q_network.lstm_num_layers if hasattr(self.q_network, 'lstm_num_layers') else 1
            hidden_size = self.q_network.lstm_hidden_size if hasattr(self.q_network, 'lstm_hidden_size') else 256 # Default or fetch
            
            h_0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
            c_0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
            self.lstm_hidden_state = (h_0, c_0)
        else:
            self.lstm_hidden_state = None

    def select_action(self, observation: dict, epsilon=0.0):
        """Selects an action.
           If use_noisy_nets is True, epsilon is ignored and exploration is handled by the noisy layers.
           Otherwise, an epsilon-greedy policy is used.
        Args:
            observation (dict): A dictionary where keys are sensor names and values are NumPy arrays.
            epsilon (float): Epsilon for epsilon-greedy exploration (ignored if use_noisy_nets is True).
        """
        self.timesteps += 1
        obs_tensor_dict = self._observation_to_tensor_dict(observation)

        if self.use_lstm and self.lstm_hidden_state is None:
            self.reset_lstm_hidden_state(batch_size=1)

        original_bn_training_states = []

        with torch.no_grad():
            if self.use_noisy_nets:
                self.q_network.train() # Noisy nets need train mode for exploration noise
                
                # Temporarily set BatchNorm layers to eval mode to avoid error with batch_size=1
                for module in self.q_network.modules():
                    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                        original_bn_training_states.append(module.training)
                        module.eval()
                        
                model_output, next_lstm_hidden = self.q_network(obs_tensor_dict, self.lstm_hidden_state)
                
                # Restore BatchNorm layers to their original training state
                idx = 0
                for module in self.q_network.modules():
                    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                        if idx < len(original_bn_training_states):
                            module.train(original_bn_training_states[idx])
                            idx += 1
                        else:
                            # This case should not happen if logic is correct
                            module.train() # Default to train if something went wrong with state tracking
                            
            else: # Epsilon-greedy
                self.q_network.eval() # Standard eval mode for exploitation or epsilon-greedy
                model_output, next_lstm_hidden = self.q_network(obs_tensor_dict, self.lstm_hidden_state)
            
            if self.use_lstm:
                self.lstm_hidden_state = next_lstm_hidden

            if self.use_distributional_rl:
                q_values = (model_output * self.support).sum(dim=2)
            else:
                q_values = model_output
        
        if self.use_noisy_nets:
            # Model was already in train() mode or BNs were handled. If other parts of network need to be eval, adjust here.
            # For NoisyNets, the primary network stays in train for noise, BNs were temp eval.
            pass 
        elif random.random() > epsilon:
            self.q_network.train() # Switch back to train if it was eval and we are exploiting
        else:
            self.q_network.train() # Ensure train mode if coming from eval for exploration
            return self.action_space.sample()

        return np.argmax(q_values.cpu().data.numpy().squeeze())

    def step_experience_and_learn(self):
        loss = None
        if len(self.replay_buffer) > self.batch_size and self.timesteps % self.update_every == 0:
            # Sample experiences
            if self.use_prioritized_replay:
                experiences_list, indices, weights = self.replay_buffer.sample(self.batch_size)
                # Convert weights to tensor
                is_weights_tensor = torch.from_numpy(weights).float().to(self.device)
            else:
                experiences_list = self.replay_buffer.sample(self.batch_size)
                indices = None
                is_weights_tensor = None # No IS weights for uniform sampling
            
            batched_experiences_dict = self._batch_experiences(experiences_list)
            
            if batched_experiences_dict:
                loss = self.learn(batched_experiences_dict, indices, is_weights_tensor)
        return loss

    def _project_distribution(self, next_dist, rewards, dones, gammas):
        """Projects the target distribution onto the support of the current network's atoms for C51."""
        batch_size = next_dist.size(0)
        target_support = rewards + gammas * self.support.unsqueeze(0) * (1 - dones) # (batch_size, n_atoms)
        
        # Clamp target support to [v_min, v_max]
        target_support = target_support.clamp(self.v_min, self.v_max)
        
        # Calculate projection indices and weights (bilinear interpolation)
        b = (target_support - self.v_min) / self.delta_z # (batch_size, n_atoms)
        lower_bound = b.floor().long()
        upper_bound = b.ceil().long()

        # Handle cases where lower_bound == upper_bound (value is exactly on an atom)
        # And cases where they might go out of bounds due to clamping + floating point issues
        lower_bound.clamp_(0, self.n_atoms - 1)
        upper_bound.clamp_(0, self.n_atoms - 1)

        # Initialize projected distribution (batch_size, n_actions, n_atoms)
        # We need to do this for each action in the batch. 
        # next_dist is (batch_size, n_actions, n_atoms)
        # target_support was calculated for each atom of a *single* action's distribution (the one selected by policy)
        # The projection needs to be done for the distribution of the target action.
        # For Double DQN, this target action is selected by the online network.
        
        # Create the projected distribution for the target actions
        # This part is tricky because `next_dist` has shape (batch, n_actions, n_atoms)
        # but the rewards, dones, gammas apply to the chosen action. 
        # The projection should be for the distribution of the *selected* next_actions.
        # Let's assume `next_dist` here is already the distribution of the chosen next actions (batch, n_atoms)

        projected_dist = torch.zeros(batch_size, self.n_atoms, device=self.device)
        
        # Expand next_dist from (batch_size, n_atoms) to (batch_size * n_atoms)
        # This assumes next_dist passed in is already (batch_size, n_atoms) for the chosen action
        next_dist_flat = next_dist.view(-1) # batch_size * n_atoms

        # Expand lower/upper bounds and weights for batch processing
        # lower_bound and upper_bound have shape (batch_size, n_atoms)
        # The original probabilities from next_dist need to be distributed
        
        offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).long()\
                        .unsqueeze(1).expand(batch_size, self.n_atoms).to(self.device)

        # Distribute probabilities
        # For each atom in the original next_dist, its probability p_j is distributed
        # to atoms l and u in the projected_dist.
        # projected_dist[l] += p_j * (u - b_j)
        # projected_dist[u] += p_j * (b_j - l)
        
        # Corrected projection logic using add_ with index_add_ syntax for safety
        # Iterate over atoms of the original distribution
        for j in range(self.n_atoms):
            # Probabilities for the j-th atom across the batch
            p_j = next_dist[:, j] # Shape: (batch_size)
            
            # Target support, lower/upper bounds for this j-th atom calculation
            # target_support_j is effectively target_support[:, j] but this is incorrect context
            # We need to project each atom of the *entire next_dist* distribution of the target action
            # So target_support should be (batch_size, n_atoms) for the target action.
            # b_j, l_j, u_j also (batch_size, n_atoms) referring to the j-th component of target support.
            
            # This means `b` (batch_size, n_atoms) is correct. `lower_bound`, `upper_bound` also correct.
            # `next_dist` passed in should be (batch_size, n_atoms) - dist of selected next action.

            # Weight for lower bound: (upper_bound_atom_index - actual_float_index)
            # Weight for upper bound: (actual_float_index - lower_bound_atom_index)
            weight_lower = (upper_bound[:, j].float() - b[:, j]).clamp(0, 1)
            weight_upper = (b[:, j] - lower_bound[:, j].float()).clamp(0, 1)
            
            # Add to projected distribution
            # projected_dist.scatter_add_(dim=1, index=lower_bound[:,j].unsqueeze(1), src=(p_j * weight_lower).unsqueeze(1))
            # projected_dist.scatter_add_(dim=1, index=upper_bound[:,j].unsqueeze(1), src=(p_j * weight_upper).unsqueeze(1))
            # scatter_add_ is tricky with duplicate indices. Using a loop for clarity first.
            for batch_idx in range(batch_size):
                if lower_bound[batch_idx, j] == upper_bound[batch_idx, j]: # If it falls exactly on an atom
                    projected_dist[batch_idx, lower_bound[batch_idx, j]] += p_j[batch_idx]
                else:
                    projected_dist[batch_idx, lower_bound[batch_idx, j]] += p_j[batch_idx] * weight_lower[batch_idx]
                    projected_dist[batch_idx, upper_bound[batch_idx, j]] += p_j[batch_idx] * weight_upper[batch_idx]
        return projected_dist

    def learn(self, experiences_dict, batch_indices=None, is_weights=None):
        self.q_network.train()
        self.target_q_network.eval()

        initial_hidden_state_batch = None
        if self.use_lstm:
            batch_size_actual = list(experiences_dict['states'].values())[0].size(0)
            num_layers = self.q_network.lstm_num_layers if hasattr(self.q_network, 'lstm_num_layers') else 1
            hidden_size = self.q_network.lstm_hidden_size if hasattr(self.q_network, 'lstm_hidden_size') else 256
            h_0_batch = torch.zeros(num_layers, batch_size_actual, hidden_size).to(self.device)
            c_0_batch = torch.zeros(num_layers, batch_size_actual, hidden_size).to(self.device)
            initial_hidden_state_batch = (h_0_batch, c_0_batch)

        states_dict = experiences_dict['states']
        actions = experiences_dict['actions']
        rewards = experiences_dict['rewards']
        next_states_dict = experiences_dict['next_states']
        dones = experiences_dict['dones']
        
        current_gamma_for_projection = self.gamma
        if self.use_n_step:
            gammas_n = experiences_dict['gammas_n']
            current_gamma_for_projection = gammas_n 

        if self.use_distributional_rl:
            with torch.no_grad():
                next_output_target, _ = self.target_q_network(next_states_dict, initial_hidden_state_batch) # Unpack tuple
                next_action_dist_target = next_output_target.detach() # Now detach the tensor
                
                if self.double_dqn:
                    next_output_online, _ = self.q_network(next_states_dict, initial_hidden_state_batch) # Unpack tuple
                    next_action_dist_online = next_output_online # No detach needed for online net values before .max
                    expected_q_online = (next_action_dist_online * self.support).sum(dim=2)
                    next_actions = expected_q_online.max(1)[1]
                else:
                    expected_q_target = (next_action_dist_target * self.support).sum(dim=2)
                    next_actions = expected_q_target.max(1)[1]
                
                next_best_action_dist = next_action_dist_target[torch.arange(next_action_dist_target.size(0)), next_actions, :]
                target_projected_dist = self._project_distribution(next_best_action_dist, rewards, dones, current_gamma_for_projection)
            
            current_output, _ = self.q_network(states_dict, initial_hidden_state_batch) # Unpack tuple
            current_action_dist = current_output 
            action_indices = actions.unsqueeze(-1).expand(-1, -1, self.n_atoms)
            log_current_action_probs = torch.log(current_action_dist.gather(1, action_indices).squeeze(1) + 1e-8)
            loss = -(target_projected_dist * log_current_action_probs).sum(dim=1)

            if self.use_prioritized_replay and is_weights is not None:
                loss = (loss * is_weights.squeeze(1)).mean()
                td_errors = loss.detach() 
            else:
                loss = loss.mean()
                td_errors = loss.detach()

        else: # Standard (non-distributional) DQN path
            with torch.no_grad(): # Target calculations should not have gradients
                next_q_values_target_output, _ = self.target_q_network(next_states_dict, initial_hidden_state_batch) # Unpack
                next_q_values_target_detached = next_q_values_target_output.detach()

                if self.double_dqn:
                    next_q_online_output, _ = self.q_network(next_states_dict, initial_hidden_state_batch) # Unpack
                    # next_q_online = next_q_online_output # No detach for action selection path
                    next_actions_indices = next_q_online_output.max(1)[1].unsqueeze(1)
                    Q_targets_next = next_q_values_target_detached.gather(1, next_actions_indices)
                else:
                    Q_targets_next = next_q_values_target_detached.max(1)[0].unsqueeze(1)
            
            final_gamma = current_gamma_for_projection if self.use_n_step else self.gamma
            Q_targets = rewards + (final_gamma * Q_targets_next * (1-dones))
            
            current_q_values_output, _ = self.q_network(states_dict, initial_hidden_state_batch) # Unpack
            Q_expected = current_q_values_output.gather(1, actions)
            
            td_errors_per = torch.abs(Q_expected - Q_targets).detach()
            if self.use_prioritized_replay and is_weights is not None:
                loss = (is_weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()
            else:
                loss = F.mse_loss(Q_expected, Q_targets)
            
            if self.use_prioritized_replay: td_errors = td_errors_per
            elif not self.use_distributional_rl : # Ensure td_errors is defined for PER if not distributional and not PER already handled
                td_errors = loss.detach() # Fallback for PER if not set by td_errors_per

        # Ensure td_errors is initialized if not PER, to avoid potential issues if PER logic changes
        if not self.use_prioritized_replay and 'td_errors' not in locals():
            td_errors = loss.detach() # Or some other placeholder if needed for non-PER priority updates (though not standard)

        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        self.accumulation_counter += 1
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulation_counter = 0
            self.scheduler.step(loss.item() * self.gradient_accumulation_steps) # Unscale for scheduler
            self.training_metrics['gradient_norms'].append(grad_norm.item())
            self.training_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

        # Update priorities in PER buffer if used
        if self.use_prioritized_replay and batch_indices is not None:
            priorities = td_errors.cpu().numpy().flatten()
            self.replay_buffer.update_priorities(batch_indices, priorities)

        self._soft_update(self.q_network, self.target_q_network, self.tau)
        actual_loss = loss.item() * self.gradient_accumulation_steps
        self.training_metrics['loss_history'].append(actual_loss)
        return actual_loss

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
        """Saves the Q-network, target Q-network, optimizer states, and current timesteps."""
        try:
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
                logger.info(f"Created directory for saving models: {directory_path}")

            q_network_path = os.path.join(directory_path, f"{model_name}_q_network.pth")
            target_q_network_path = os.path.join(directory_path, f"{model_name}_target_q_network.pth")
            optimizer_path = os.path.join(directory_path, f"{model_name}_optimizer.pth")
            agent_state_path = os.path.join(directory_path, f"{model_name}_agent_state.pth")

            torch.save(self.q_network.state_dict(), q_network_path)
            torch.save(self.target_q_network.state_dict(), target_q_network_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            
            agent_state = {
                'timesteps': self.timesteps
                # Add other agent-specific state variables here if needed in the future
            }
            torch.save(agent_state, agent_state_path)
            
            logger.info(f"Saved DQN agent models and state to {directory_path} with prefix '{model_name}'")
        except Exception as e:
            logger.error(f"Error saving DQN agent models: {e}", exc_info=True)

    def load(self, directory_path, model_name="dqn_agent", map_location=None):
        """Loads the Q-network, target Q-network, optimizer states, and current timesteps."""
        try:
            q_network_path = os.path.join(directory_path, f"{model_name}_q_network.pth")
            target_q_network_path = os.path.join(directory_path, f"{model_name}_target_q_network.pth")
            optimizer_path = os.path.join(directory_path, f"{model_name}_optimizer.pth")
            agent_state_path = os.path.join(directory_path, f"{model_name}_agent_state.pth")

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

            if os.path.exists(agent_state_path):
                agent_state = torch.load(agent_state_path, map_location=device_to_load_on)
                self.timesteps = agent_state.get('timesteps', 0)
                logger.info(f"Loaded agent state (timesteps: {self.timesteps}) from {agent_state_path}")
            else:
                self.timesteps = 0 # Default if not found
                logger.warning(f"Agent state file not found at {agent_state_path}. Timesteps reset to 0.")

            self.q_network.train() # Default to train mode after loading for continued training
            self.target_q_network.eval() # Target network is always in eval mode except during direct state_dict load
            
            logger.info(f"Loaded DQN agent models from {directory_path} with prefix '{model_name}'")
            return True # Indicate success
        except Exception as e:
            logger.error(f"Error loading DQN agent models: {e}", exc_info=True)
            return False # Indicate failure

    def _batch_experiences(self, experiences_list):
        """Converts a list of Experience or NStepExperience namedtuples
           into a dictionary of batched tensors, ready for the learning step.
        """
        batch = OrderedDict()
        
        # Check if we are using N-step experiences
        is_n_step = hasattr(experiences_list[0], 'n_step_reward')

        states_dict_list = [e.state for e in experiences_list if e is not None]
        actions_list = [e.action for e in experiences_list if e is not None]
        
        if is_n_step:
            rewards_list = [e.n_step_reward for e in experiences_list if e is not None]
            next_states_dict_list = [e.next_n_state for e in experiences_list if e is not None]
            dones_list = [e.done for e in experiences_list if e is not None] # 'done' from NStepExperience
            gammas_n_list = [e.gamma_n for e in experiences_list if e is not None]
        else:
            rewards_list = [e.reward for e in experiences_list if e is not None]
            next_states_dict_list = [e.next_state for e in experiences_list if e is not None]
            dones_list = [e.done for e in experiences_list if e is not None]

        if not states_dict_list:
            return None

        sample_obs_keys = states_dict_list[0].keys()
        
        batched_states = OrderedDict()
        for key in sample_obs_keys:
            batched_states[key] = torch.from_numpy(np.array([s[key] for s in states_dict_list])).float().to(self.device)

        batched_next_states = OrderedDict()
        for key in sample_obs_keys: 
            batched_next_states[key] = torch.from_numpy(np.array([s[key] for s in next_states_dict_list])).float().to(self.device)
        
        batch['states'] = batched_states
        batch['next_states'] = batched_next_states
        batch['actions'] = torch.from_numpy(np.array(actions_list)).long().unsqueeze(1).to(self.device)
        batch['rewards'] = torch.from_numpy(np.array(rewards_list)).float().unsqueeze(1).to(self.device)
        batch['dones'] = torch.from_numpy(np.array(dones_list).astype(np.uint8)).float().unsqueeze(1).to(self.device)
        
        if is_n_step:
            batch['gammas_n'] = torch.from_numpy(np.array(gammas_n_list)).float().unsqueeze(1).to(self.device)
            
        return batch 