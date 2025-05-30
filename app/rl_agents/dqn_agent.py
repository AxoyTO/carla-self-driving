import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict, deque
from typing import Dict, Tuple, Optional, Union
import os # For os.makedirs and os.path.join
import logging # For logging save/load actions
from torch.cuda.amp import GradScaler, autocast
import math

from .base_agent import BaseAgent
# from ..models.dqn_model import DQNModel # Assuming you have a DQNModel class
# from ..replay_buffers.uniform_replay_buffer import UniformReplayBuffer # Or any other buffer

logger = logging.getLogger(__name__) # Logger for this module

class EnhancedDQNAgent(BaseAgent):
    """Enhanced DQN Agent with Rainbow DQN improvements and advanced training features."""
    
    def __init__(self, observation_space, action_space, model, replay_buffer,
                 lr=1e-4, gamma=0.99, tau=1e-3, batch_size=64, update_every=4,
                 device='cpu', use_mixed_precision=True, gradient_accumulation_steps=1,
                 max_grad_norm=1.0, double_dqn=True, dueling_dqn=False, use_n_step=False,
                 use_prioritized_replay=False, use_noisy_nets=False,
                 # C51 parameters
                 use_distributional_rl=False, n_atoms=51, v_min=-10.0, v_max=10.0,
                 use_lstm=False,
                 # Enhanced features
                 use_multi_step_bootstrap=True, multi_step_n=3,
                 exploration_strategy='epsilon_greedy', 
                 adaptive_exploration=True,
                 weight_decay=1e-4,
                 use_cosine_annealing=True,
                 curriculum_aware_training=True):
        """
        Initialize Enhanced DQN Agent with Rainbow DQN improvements.
        
        Args:
            All original parameters plus:
            use_multi_step_bootstrap: Enable multi-step TD learning
            multi_step_n: Number of steps for multi-step learning
            exploration_strategy: 'epsilon_greedy', 'noisy_nets', or 'ucb'
            adaptive_exploration: Dynamically adjust exploration based on uncertainty
            weight_decay: L2 regularization coefficient
            use_cosine_annealing: Use cosine annealing LR scheduler
            curriculum_aware_training: Adapt learning based on curriculum phase
        """
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

        # Enhanced features
        self.use_multi_step_bootstrap = use_multi_step_bootstrap
        self.multi_step_n = multi_step_n
        self.exploration_strategy = exploration_strategy
        self.adaptive_exploration = adaptive_exploration
        self.curriculum_aware_training = curriculum_aware_training
        
        # Optimizers with weight decay
        self.optimizer = optim.AdamW(
            self.q_network.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
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
        
        # Mixed precision scaler with better settings
        if self.use_mixed_precision:
            self.scaler = GradScaler(
                init_scale=2**12,  # Start with lower scale (4096 instead of 65536)
                growth_factor=1.5,  # Slower growth to prevent rapid scaling
                backoff_factor=0.25,  # Faster backoff when overflow detected
                growth_interval=200  # Less frequent growth
            )
        else:
            self.scaler = None
        
        # Enhanced learning rate scheduling
        if use_cosine_annealing:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, 
                T_0=1000,  # Restart every 1000 updates
                T_mult=2,  # Double the restart period each time
                eta_min=lr * 0.01
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.8, patience=10, verbose=True
            )
        
        # Enhanced training metrics
        self.training_metrics = {
            'loss_history': deque(maxlen=10000),
            'gradient_norms': deque(maxlen=1000),
            'learning_rates': deque(maxlen=1000),
            'q_value_estimates': deque(maxlen=1000),
            'td_errors': deque(maxlen=1000),
            'exploration_rates': deque(maxlen=1000),
            'uncertainty_estimates': deque(maxlen=1000)
        }

        self.timesteps = 0 # Counter for update_every
        self.training_updates = 0

        self.use_distributional_rl = use_distributional_rl
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        if self.use_distributional_rl:
            self.delta_z = (v_max - v_min) / (n_atoms - 1)
            self.support = torch.linspace(v_min, v_max, n_atoms).to(device)

        self.use_lstm = use_lstm
        self.lstm_hidden_state = None # To store hidden state for select_action

        # Enhanced exploration tracking
        self.exploration_history = deque(maxlen=1000)
        self.action_counts = np.zeros(action_space.n)
        self.state_visitation_counts = {}
        
        # Curriculum-aware features
        self.current_curriculum_phase = 0
        self.phase_adaptation_factor = 1.0
        
        # UCB exploration parameters
        self.ucb_confidence = 2.0
        self.action_ucb_counts = np.zeros(action_space.n)
        self.action_rewards = np.zeros(action_space.n)
        
        logger.info(f"Enhanced DQN Agent initialized with exploration: {exploration_strategy}, "
                   f"multi-step: {use_multi_step_bootstrap}, distributional: {use_distributional_rl}")

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
        """Enhanced action selection with multiple exploration strategies.
        Args:
            observation (dict): A dictionary where keys are sensor names and values are NumPy arrays.
            epsilon (float): Epsilon for epsilon-greedy exploration.
        """
        self.timesteps += 1
        obs_tensor_dict = self._observation_to_tensor_dict(observation)

        if self.use_lstm and self.lstm_hidden_state is None:
            self.reset_lstm_hidden_state(batch_size=1)

        original_bn_training_states = []

        with torch.no_grad():
            if self.use_noisy_nets or self.exploration_strategy == 'noisy_nets':
                self.q_network.train()
                
                # Handle BatchNorm layers for single sample inference
                for module in self.q_network.modules():
                    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                        original_bn_training_states.append(module.training)
                        module.eval()
                        
                model_output, next_lstm_hidden = self.q_network(obs_tensor_dict, self.lstm_hidden_state)
                
                # Restore BatchNorm layers
                idx = 0
                for module in self.q_network.modules():
                    if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                        if idx < len(original_bn_training_states):
                            module.train(original_bn_training_states[idx])
                            idx += 1
                        else:
                            module.train()
                            
            else:
                self.q_network.eval()
                model_output, next_lstm_hidden = self.q_network(obs_tensor_dict, self.lstm_hidden_state)
            
            if self.use_lstm:
                self.lstm_hidden_state = next_lstm_hidden

            if self.use_distributional_rl:
                q_values = (model_output * self.support).sum(dim=2)
            else:
                q_values = model_output
        
        # Enhanced exploration strategies
        if self.exploration_strategy == 'noisy_nets' or self.use_noisy_nets:
            # NoisyNets handle exploration internally
            action = np.argmax(q_values.cpu().data.numpy().squeeze())
            
        elif self.exploration_strategy == 'ucb':
            # Upper Confidence Bound exploration
            action = self._ucb_action_selection(q_values)
            
        else:  # epsilon_greedy (default)
            # Calculate uncertainty for adaptive exploration
            uncertainty = self._calculate_uncertainty_bonus(q_values) if self.adaptive_exploration else 0
            
            # Adapt epsilon based on uncertainty and curriculum phase
            effective_epsilon = self._adaptive_epsilon(uncertainty, epsilon) if self.adaptive_exploration else epsilon
            
            if random.random() > effective_epsilon:
                action = np.argmax(q_values.cpu().data.numpy().squeeze())
            else:
                action = self.action_space.sample()
            
            # Track exploration metrics
            self.exploration_history.append(effective_epsilon)
            self.training_metrics['exploration_rates'].append(effective_epsilon)
            if self.adaptive_exploration:
                self.training_metrics['uncertainty_estimates'].append(uncertainty.item() if torch.is_tensor(uncertainty) else uncertainty)

        # Update action tracking for UCB
        self.action_counts[action] += 1
        
        # Track Q-value estimates for analysis
        max_q = torch.max(q_values).item()
        self.training_metrics['q_value_estimates'].append(max_q)
        
        self.q_network.train()  # Ensure training mode for continued learning
        return action

    def step_experience_and_learn(self):
        loss = None
        # Track whether learning actually occurred this step
        learning_occurred = False
        
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
                learning_occurred = True
        
        # Track learning frequency for analytics
        if hasattr(self, 'training_metrics'):
            self.training_metrics.setdefault('learning_steps', 0)
            self.training_metrics.setdefault('total_steps', 0)
            
            if learning_occurred:
                self.training_metrics['learning_steps'] += 1
            self.training_metrics['total_steps'] += 1
            
            # Track learning frequency ratio
            if self.training_metrics['total_steps'] > 0:
                self.training_metrics['learning_frequency'] = (
                    self.training_metrics['learning_steps'] / self.training_metrics['total_steps']
                )
        
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
        
        # Validate input tensors for NaN/Inf before processing
        if not self._validate_tensor_batch(states_dict, "states"):
            logger.warning("Invalid states detected, skipping learning step")
            return None
        if not self._validate_tensor_batch(next_states_dict, "next_states"):
            logger.warning("Invalid next_states detected, skipping learning step")
            return None
        if not torch.isfinite(rewards).all():
            logger.warning("Invalid rewards detected, skipping learning step")
            return None
        
        current_gamma_for_projection = self.gamma
        if self.use_n_step:
            gammas_n = experiences_dict['gammas_n']
            if not torch.isfinite(gammas_n).all():
                logger.warning("Invalid gammas_n detected, using standard gamma")
                current_gamma_for_projection = self.gamma
            else:
                current_gamma_for_projection = gammas_n
        
        # Apply input stabilization
        rewards = torch.clamp(rewards, -100.0, 100.0)  # Prevent extreme reward values
        
        # Enhanced distributional RL computation with stability checks
        if self.use_distributional_rl:
            with torch.no_grad():
                next_dist_target_output, _ = self.target_q_network(next_states_dict, initial_hidden_state_batch)
                next_dist_target = next_dist_target_output.detach()
                
                if self.double_dqn:
                    next_q_online_output, _ = self.q_network(next_states_dict, initial_hidden_state_batch)
                    next_actions = next_q_online_output.mean(dim=2).max(1)[1].unsqueeze(1).unsqueeze(1).expand(-1, -1, self.n_atoms)
                    next_dist = next_dist_target.gather(1, next_actions).squeeze(1)
                else:
                    next_q_values = next_dist_target.mean(dim=2)
                    next_actions = next_q_values.max(1)[1].unsqueeze(1).unsqueeze(1).expand(-1, -1, self.n_atoms)
                    next_dist = next_dist_target.gather(1, next_actions).squeeze(1)
            
            projected_dist = self._project_distribution(next_dist, rewards, dones, current_gamma_for_projection)
            
            current_dist_output, _ = self.q_network(states_dict, initial_hidden_state_batch)
            current_dist = current_dist_output.gather(1, actions.unsqueeze(2).expand(-1, -1, self.n_atoms)).squeeze(1)
            
            # Enhanced loss computation with stability checks
            current_dist = torch.clamp(current_dist, min=1e-8, max=1.0)  # Prevent log(0)
            projected_dist = torch.clamp(projected_dist, min=1e-8, max=1.0)
            
            loss = -(projected_dist * torch.log(current_dist)).sum(dim=1)
            
            if self.use_prioritized_replay and is_weights is not None:
                loss = (is_weights.squeeze() * loss).mean()
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
            
            # Stabilize target values
            Q_targets = torch.clamp(Q_targets, -1000.0, 1000.0)
            
            current_q_values_output, _ = self.q_network(states_dict, initial_hidden_state_batch) # Unpack
            Q_expected = current_q_values_output.gather(1, actions)
            
            # Validate Q-values before loss computation
            if not torch.isfinite(Q_expected).all():
                logger.warning("NaN/Inf in Q_expected, applying model recovery")
                self._emergency_model_stabilization()
                return None
            
            if not torch.isfinite(Q_targets).all():
                logger.warning("NaN/Inf in Q_targets, clamping values")
                Q_targets = torch.where(torch.isfinite(Q_targets), Q_targets, torch.zeros_like(Q_targets))
            
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

        # Apply comprehensive loss stabilization
        loss = self._apply_loss_stabilization(loss)
        
        # Validate final loss before backpropagation
        if not torch.isfinite(loss):
            logger.warning(f"Gradient explosion detected: nan at update {self.training_updates}")
            self._handle_gradient_explosion_recovery()
            return None

        # Enhanced gradient computation with mixed precision
        scaled_loss = loss / self.gradient_accumulation_steps
        
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
            
        self.accumulation_counter += 1
        
        if self.accumulation_counter >= self.gradient_accumulation_steps:
            # Enhanced gradient handling with NaN detection and recovery
            if self.use_mixed_precision and self.scaler is not None:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                
                # Validate gradients before processing
                if not self._validate_gradients():
                    logger.warning(f"Gradient explosion detected: nan at update {self.training_updates}")
                    self.scaler.update()  # Update scaler to prevent scale getting stuck
                    self.optimizer.zero_grad()
                    self._handle_gradient_explosion_recovery()
                    self.training_metrics['gradient_norms'].append(0.0)
                    self.accumulation_counter = 0
                    self.training_updates += 1
                    return None
                
                grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
                
                # Comprehensive NaN/Inf gradient detection and handling
                if torch.isfinite(grad_norm) and not torch.isnan(grad_norm):
                    # Normal case: finite gradients
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # Update learning rate scheduler
                    if hasattr(self.scheduler, 'step'):
                        if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            self.scheduler.step()
                        else:
                            self.scheduler.step(loss.item() * self.gradient_accumulation_steps)
                            
                    # Track successful gradient update
                    self.training_metrics['gradient_norms'].append(grad_norm.item())
                else:
                    # Handle NaN/Inf gradients
                    if torch.isnan(grad_norm):
                        logger.warning(f"Gradient explosion detected: nan at update {self.training_updates}")
                    else:
                        logger.warning(f"Gradient explosion detected: inf ({grad_norm.item():.2f}) at update {self.training_updates}")
                    
                    # Skip optimizer step but still update scaler to prevent scale getting stuck
                    self.scaler.update()
                    
                    # Reset gradients to prevent accumulation of bad gradients
                    self.optimizer.zero_grad()
                    
                    # Apply gradient recovery strategies
                    self._handle_gradient_explosion_recovery()
                    
                    # Track failed gradient update
                    self.training_metrics['gradient_norms'].append(0.0)
            else:
                # Non-mixed precision gradient handling
                if not self._validate_gradients():
                    logger.warning(f"Gradient explosion detected: nan at update {self.training_updates}")
                    self.optimizer.zero_grad()
                    self._handle_gradient_explosion_recovery()
                    self.training_metrics['gradient_norms'].append(0.0)
                    self.accumulation_counter = 0
                    self.training_updates += 1
                    return None
                
                grad_norm = torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
                
                # Check for NaN/Inf gradients in non-mixed precision mode
                if torch.isfinite(grad_norm) and not torch.isnan(grad_norm):
                    self.optimizer.step()
                    
                    # Update learning rate scheduler
                    if hasattr(self.scheduler, 'step'):
                        if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                            self.scheduler.step()
                        else:
                            self.scheduler.step(loss.item() * self.gradient_accumulation_steps)
                            
                    # Track successful gradient update
                    self.training_metrics['gradient_norms'].append(grad_norm.item())
                else:
                    # Handle NaN/Inf gradients in non-mixed precision
                    if torch.isnan(grad_norm):
                        logger.warning(f"Gradient explosion detected: nan at update {self.training_updates}")
                    else:
                        logger.warning(f"Gradient explosion detected: inf ({grad_norm.item():.2f}) at update {self.training_updates}")
                    
                    # Apply gradient recovery strategies
                    self._handle_gradient_explosion_recovery()
                    
                    # Track failed gradient update
                    self.training_metrics['gradient_norms'].append(0.0)
            
            self.optimizer.zero_grad()
            self.accumulation_counter = 0
            self.training_updates += 1
            
            # Enhanced metrics tracking
            self.training_metrics['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Track TD errors for analysis
            if 'td_errors' in locals():
                # Check for NaN in TD errors as well
                if torch.isfinite(td_errors).all():
                    avg_td_error = torch.mean(torch.abs(td_errors)).item()
                    self.training_metrics['td_errors'].append(avg_td_error)
                else:
                    logger.warning("NaN/Inf detected in TD errors. Skipping TD error tracking.")
                    self.training_metrics['td_errors'].append(0.0)

        # Update priorities in PER buffer if used
        if self.use_prioritized_replay and batch_indices is not None:
            # Ensure td_errors are valid before updating priorities
            if torch.isfinite(td_errors).all():
                priorities = td_errors.cpu().numpy().flatten()
                self.replay_buffer.update_priorities(batch_indices, priorities)
            else:
                logger.warning("Invalid TD errors detected, skipping priority update")

        # Enhanced soft update with curriculum-aware adaptation
        effective_tau = self.tau * self.phase_adaptation_factor if self.curriculum_aware_training else self.tau
        self._soft_update(self.q_network, self.target_q_network, effective_tau)
        
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

    def _calculate_uncertainty_bonus(self, q_values: torch.Tensor) -> torch.Tensor:
        """Calculate uncertainty bonus for adaptive exploration."""
        if self.use_distributional_rl:
            # For distributional RL, use entropy as uncertainty measure
            probs = F.softmax(q_values, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            return entropy.mean(dim=-1)  # Average across atoms
        else:
            # For regular DQN, use variance of Q-values as uncertainty
            return torch.var(q_values, dim=-1)

    def _ucb_action_selection(self, q_values: torch.Tensor) -> int:
        """Upper Confidence Bound action selection."""
        if self.timesteps < len(self.action_counts):
            # Ensure all actions are tried at least once
            return self.timesteps % len(self.action_counts)
        
        # Calculate UCB values
        total_counts = np.sum(self.action_ucb_counts)
        ucb_values = np.zeros(len(self.action_counts))
        
        for a in range(len(self.action_counts)):
            if self.action_ucb_counts[a] > 0:
                confidence_interval = self.ucb_confidence * math.sqrt(
                    math.log(total_counts) / self.action_ucb_counts[a]
                )
                ucb_values[a] = self.action_rewards[a] / self.action_ucb_counts[a] + confidence_interval
            else:
                ucb_values[a] = float('inf')  # Unvisited actions get highest priority
        
        return np.argmax(ucb_values)

    def _adaptive_epsilon(self, uncertainty: float, base_epsilon: float) -> float:
        """Adapt epsilon based on model uncertainty."""
        if not self.adaptive_exploration:
            return base_epsilon
        
        # Scale epsilon based on uncertainty
        uncertainty_normalized = torch.clamp(uncertainty, 0, 1).item()
        adaptive_factor = 1.0 + uncertainty_normalized
        
        # Apply curriculum phase adaptation
        phase_factor = max(0.5, 1.0 - self.current_curriculum_phase * 0.1)
        
        return min(base_epsilon * adaptive_factor * phase_factor, 1.0)

    def update_curriculum_phase(self, phase_info: Dict):
        """Update agent parameters based on curriculum phase."""
        if not self.curriculum_aware_training:
            return
            
        self.current_curriculum_phase = phase_info.get('phase_number', 0)
        phase_difficulty = phase_info.get('difficulty', 1.0)
        
        # Adapt learning parameters based on phase difficulty
        self.phase_adaptation_factor = max(0.5, 1.0 / (1.0 + phase_difficulty * 0.1))
        
        # Adjust exploration based on phase complexity
        if hasattr(self, 'exploration_strategy') and self.exploration_strategy == 'ucb':
            self.ucb_confidence = max(1.0, 2.0 - phase_difficulty * 0.2)
        
        logger.info(f"Updated curriculum phase: {self.current_curriculum_phase}, "
                   f"adaptation factor: {self.phase_adaptation_factor:.3f}")

    def get_training_analytics(self) -> Dict:
        """Get comprehensive training analytics."""
        analytics = {}
        
        # Loss and gradient analysis
        if self.training_metrics['loss_history']:
            recent_losses = list(self.training_metrics['loss_history'])[-100:]
            analytics['avg_recent_loss'] = np.mean(recent_losses)
            analytics['loss_std'] = np.std(recent_losses)
            
            # Calculate loss trend using linear regression (slope indicates trend direction)
            if len(recent_losses) > 1:
                analytics['loss_trend'] = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            else:
                analytics['loss_trend'] = 0.0
                
            # Add loss statistics for better monitoring
            analytics['min_recent_loss'] = np.min(recent_losses)
            analytics['max_recent_loss'] = np.max(recent_losses)
            analytics['total_training_updates'] = len(self.training_metrics['loss_history'])
        
        # Q-value analysis
        if self.training_metrics['q_value_estimates']:
            recent_q_values = list(self.training_metrics['q_value_estimates'])[-100:]
            analytics['avg_q_value'] = np.mean(recent_q_values)
            analytics['q_value_std'] = np.std(recent_q_values)
        
        # Exploration analysis
        if self.training_metrics['exploration_rates']:
            recent_exploration = list(self.training_metrics['exploration_rates'])[-100:]
            analytics['avg_exploration_rate'] = np.mean(recent_exploration)
            analytics['exploration_efficiency'] = self._calculate_exploration_efficiency()
        
        # Action distribution analysis
        total_actions = np.sum(self.action_counts)
        if total_actions > 0:
            action_probs = self.action_counts / total_actions
            analytics['action_entropy'] = -np.sum(action_probs * np.log(action_probs + 1e-8))
            analytics['action_distribution'] = action_probs.tolist()
        
        # Gradient and training stability
        if self.training_metrics['gradient_norms']:
            recent_grads = list(self.training_metrics['gradient_norms'])[-100:]
            analytics['avg_grad_norm'] = np.mean(recent_grads)
            analytics['grad_stability'] = 1.0 / (1.0 + np.std(recent_grads))
        
        analytics['training_updates'] = self.training_updates
        analytics['timesteps'] = self.timesteps
        
        # Learning frequency analytics
        if 'learning_frequency' in self.training_metrics:
            analytics['learning_frequency'] = self.training_metrics['learning_frequency']
            analytics['learning_steps'] = self.training_metrics.get('learning_steps', 0)
            analytics['total_agent_steps'] = self.training_metrics.get('total_steps', 0)
        
        return analytics

    def _calculate_exploration_efficiency(self) -> float:
        """Calculate exploration efficiency based on action diversity and uncertainty."""
        if len(self.exploration_history) < 10:
            return 0.0
        
        # Recent exploration rates
        recent_exploration = list(self.exploration_history)[-50:]
        avg_exploration = np.mean(recent_exploration)
        
        # Action diversity (entropy)
        total_actions = np.sum(self.action_counts)
        if total_actions == 0:
            return 0.0
        
        action_probs = self.action_counts / total_actions
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
        max_entropy = np.log(len(self.action_counts))
        normalized_entropy = action_entropy / max_entropy
        
        # Combine exploration rate and action diversity
        efficiency = (avg_exploration + normalized_entropy) / 2.0
        return min(efficiency, 1.0)

    def reset_exploration_tracking(self):
        """Reset exploration tracking for new curriculum phase."""
        self.action_counts.fill(0)
        self.action_ucb_counts.fill(0)
        self.action_rewards.fill(0)
        self.exploration_history.clear()
        logger.info("Reset exploration tracking for new curriculum phase")

    def _handle_gradient_explosion_recovery(self):
        """
        Implement comprehensive gradient explosion recovery strategies.
        
        This method is called when NaN or infinite gradients are detected,
        applying various recovery techniques to stabilize training.
        """
        # Strategy 1: Reduce learning rate temporarily
        current_lr = self.optimizer.param_groups[0]['lr']
        recovery_lr = current_lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = recovery_lr
        
        logger.info(f"Applied gradient explosion recovery: LR reduced from {current_lr:.6f} to {recovery_lr:.6f}")
        
        # Strategy 2: Reset gradient scaler if using mixed precision
        if self.use_mixed_precision and self.scaler is not None:
            # Reduce scaler scale to prevent future explosions
            current_scale = self.scaler.get_scale()
            new_scale = max(current_scale * 0.25, 2**10)  # Minimum scale of 1024
            self.scaler._scale.fill_(new_scale)
            logger.info(f"Reset gradient scaler from {current_scale:.0f} to {new_scale:.0f}")
        
        # Strategy 3: Soft reset of target network to main network
        # This helps when target network state becomes too different
        self._soft_update(self.q_network, self.target_q_network, 0.1)  # Larger tau for reset
        
        # Strategy 4: Clear problematic states from training metrics
        if len(self.training_metrics['loss_history']) > 10:
            # Remove recent potentially problematic losses
            recent_losses = list(self.training_metrics['loss_history'])[-5:]
            if any(loss > 1000 for loss in recent_losses):  # Very high losses indicate instability
                for _ in range(min(5, len(self.training_metrics['loss_history']))):
                    self.training_metrics['loss_history'].pop()
                logger.info("Removed recent high loss values from training history")
        
        # Strategy 5: Track recovery attempts to prevent infinite loops
        if not hasattr(self, '_gradient_explosion_count'):
            self._gradient_explosion_count = 0
        self._gradient_explosion_count += 1
        
        # Strategy 6: Emergency reinitialization if too many failures
        if self._gradient_explosion_count > 10:
            logger.warning(f"Excessive gradient explosions detected ({self._gradient_explosion_count}). "
                          "Applying emergency stabilization.")
            self._emergency_model_stabilization()
            self._gradient_explosion_count = 0  # Reset counter after emergency intervention
    
    def _emergency_model_stabilization(self):
        """
        Emergency model stabilization when gradient explosions become chronic.
        """
        logger.warning("Applying emergency model stabilization measures")
        
        # 1. Check and fix model parameters first
        if not self._validate_model_state():
            logger.warning("Model parameters are corrupted, reinitializing affected layers")
            self._reinitialize_corrupted_parameters()
        
        # 2. Reinitialize optimizer with much lower learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        emergency_lr = max(current_lr * 0.01, 1e-7)  # Much more aggressive reduction
        
        self.optimizer = optim.AdamW(
            self.q_network.parameters(), 
            lr=emergency_lr,
            weight_decay=1e-5,  # Stronger regularization
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 3. Reinitialize learning rate scheduler with conservative settings
        if hasattr(self, 'scheduler'):
            if hasattr(self.scheduler, 'T_0'):  # CosineAnnealingWarmRestarts
                self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    self.optimizer, 
                    T_0=2000,  # Longer restart period
                    T_mult=2,
                    eta_min=emergency_lr * 0.001  # Much lower minimum
                )
            else:  # ReduceLROnPlateau
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
                )
        
        # 4. Reset mixed precision scaler with very conservative settings
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler = GradScaler(
                init_scale=2**8,  # Very low initial scale (256)
                growth_factor=1.1,  # Very slow growth
                backoff_factor=0.1,  # Aggressive backoff
                growth_interval=500  # Very infrequent growth
            )
        
        # 5. Synchronize target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # 6. Clear training metrics that might be corrupted
        for key in ['loss_history', 'gradient_norms', 'td_errors']:
            if key in self.training_metrics:
                self.training_metrics[key].clear()
        
        # 7. Reduce gradient clipping threshold temporarily
        self.max_grad_norm = min(self.max_grad_norm * 0.1, 0.1)
        
        logger.info(f"Emergency stabilization complete. New LR: {emergency_lr:.6f}, "
                   f"New max_grad_norm: {self.max_grad_norm:.6f}")
    
    def _reinitialize_corrupted_parameters(self):
        """
        Reinitialize parameters that contain NaN or infinite values.
        """
        reinitialized_layers = []
        
        for name, param in self.q_network.named_parameters():
            if not torch.isfinite(param).all():
                # Reinitialize this parameter using Xavier initialization
                if len(param.shape) >= 2:  # Weight matrix
                    torch.nn.init.xavier_uniform_(param)
                else:  # Bias vector
                    torch.nn.init.zeros_(param)
                reinitialized_layers.append(name)
                logger.warning(f"Reinitialized corrupted parameter: {name}")
        
        if reinitialized_layers:
            logger.info(f"Reinitialized {len(reinitialized_layers)} corrupted parameters")
            # Update target network to match
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def _validate_model_state(self) -> bool:
        """
        Validate that the model parameters are in a healthy state.
        
        Returns:
            bool: True if model state is healthy, False otherwise
        """
        # Check for NaN or infinite parameters
        for name, param in self.q_network.named_parameters():
            if not torch.isfinite(param).all():
                logger.warning(f"Invalid parameters detected in {name}")
                return False
                
        # Check parameter magnitudes (detect extreme values)
        for name, param in self.q_network.named_parameters():
            param_norm = torch.norm(param)
            if param_norm > 100.0:  # Very large parameters
                logger.warning(f"Large parameter norm detected in {name}: {param_norm:.2f}")
                return False
                
        return True
    
    def _validate_tensor_batch(self, tensor_dict: dict, tensor_name: str) -> bool:
        """
        Validate a batch of tensors for NaN or Inf values.
        
        Args:
            tensor_dict: Dictionary containing tensors
            tensor_name: Name of the tensor being validated
            
        Returns:
            bool: True if all tensors are valid, False otherwise
        """
        for key, tensor in tensor_dict.items():
            if not torch.isfinite(tensor).all():
                logger.warning(f"Invalid {tensor_name} detected in {key}")
                return False
        return True

    def _validate_gradients(self) -> bool:
        """
        Validate gradients for NaN or Inf values.
        
        Returns:
            bool: True if all gradients are valid, False otherwise
        """
        for name, param in self.q_network.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                logger.warning(f"Invalid gradients detected in {name}")
                return False
        return True

    def _apply_loss_stabilization(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply loss stabilization techniques to prevent gradient explosions.
        
        Args:
            loss: Raw loss tensor
            
        Returns:
            Stabilized loss tensor
        """
        # 1. Clamp loss to reasonable range
        max_loss = 100.0  # Prevent extremely high losses
        stabilized_loss = torch.clamp(loss, max=max_loss)
        
        # 2. Apply adaptive loss scaling based on recent loss history
        if len(self.training_metrics['loss_history']) > 5:
            recent_losses = list(self.training_metrics['loss_history'])[-5:]
            recent_avg = np.mean(recent_losses)
            
            # If current loss is much higher than recent average, scale it down
            if stabilized_loss.item() > recent_avg * 3.0:
                scaling_factor = min(1.0, (recent_avg * 2.0) / stabilized_loss.item())
                stabilized_loss = stabilized_loss * scaling_factor
                logger.debug(f"Applied loss scaling: {scaling_factor:.3f}")
        
        # 3. Check for NaN loss and replace with small value
        if torch.isnan(stabilized_loss):
            logger.warning("NaN loss detected, replacing with small value")
            stabilized_loss = torch.tensor(0.01, device=loss.device, dtype=loss.dtype)
        
        return stabilized_loss

    def _monitor_gradient_health(self) -> Dict[str, float]:
        """
        Monitor gradient health and return statistics for debugging.
        
        Returns:
            Dictionary containing gradient statistics
        """
        grad_stats = {
            'total_norm': 0.0,
            'max_grad': 0.0,
            'min_grad': 0.0,
            'nan_count': 0,
            'inf_count': 0,
            'zero_count': 0
        }
        
        total_norm = 0.0
        all_grads = []
        
        for name, param in self.q_network.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                
                # Count problematic gradients
                grad_stats['nan_count'] += torch.isnan(grad).sum().item()
                grad_stats['inf_count'] += torch.isinf(grad).sum().item()
                grad_stats['zero_count'] += (grad == 0).sum().item()
                
                # Collect finite gradients for statistics
                finite_mask = torch.isfinite(grad)
                if finite_mask.any():
                    finite_grads = grad[finite_mask]
                    all_grads.extend(finite_grads.flatten().tolist())
                    total_norm += torch.norm(finite_grads) ** 2
        
        if all_grads:
            grad_stats['total_norm'] = total_norm ** 0.5
            grad_stats['max_grad'] = max(all_grads)
            grad_stats['min_grad'] = min(all_grads)
        
        # Log warning if gradients are unhealthy
        if grad_stats['nan_count'] > 0 or grad_stats['inf_count'] > 0:
            logger.warning(f"Unhealthy gradients detected: "
                          f"NaN: {grad_stats['nan_count']}, "
                          f"Inf: {grad_stats['inf_count']}")
        
        return grad_stats


# Backward compatibility alias
DQNAgent = EnhancedDQNAgent 