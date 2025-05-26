import random
import numpy as np
from collections import namedtuple, deque
from typing import List, Tuple, Optional
import logging

from .base_buffer import BaseBuffer

logger = logging.getLogger(__name__)

# Experience tuple for N-step learning
NStepExperience = namedtuple("NStepExperience", 
                           field_names=["state", "action", "n_step_reward", 
                                        "next_n_state", "done", "gamma_n"])

class NStepReplayBuffer(BaseBuffer):
    """Replay buffer that stores N-step experiences."""
    
    def __init__(self, capacity: int, n_step: int, gamma: float):
        """
        Initialize NStepReplayBuffer.
        
        Args:
            capacity: Maximum size of the main buffer for N-step experiences.
            n_step: The number of steps for N-step returns.
            gamma: Discount factor.
        """
        super().__init__(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=self.n_step) # Temporary buffer for current N-step sequence
        
    def add(self, state, action, reward, next_state, done):
        """Add a single step experience and form N-step experiences."""
        # Store current transition in the temporary n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If n_step_buffer is not full yet, we can't form an N-step transition
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # Calculate N-step reward and find the Nth next state
        n_step_reward = 0.0
        current_gamma = 1.0
        final_gamma_n = self.gamma ** self.n_step # Gamma to the power of N
        
        # Iterate backwards from the (N-1)th element up to the 0th element in n_step_buffer
        # This corresponds to t+N-1 down to t for the first state in the sequence
        for i in range(self.n_step):
            s_i, a_i, r_i, s_prime_i, d_i = self.n_step_buffer[i]
            n_step_reward += current_gamma * r_i
            current_gamma *= self.gamma
            
            if d_i: # If any intermediate step is done, N-step sequence ends early
                final_gamma_n = self.gamma ** (i + 1)
                break
        
        # The first experience in the n_step_buffer is the (state, action) for this N-step return
        s_t, a_t, _, _, _ = self.n_step_buffer[0]
        
        # The Nth next state is the next_state of the last experience in the current window
        # If the sequence was cut short by a 'done', then s_n_prime is the state after the terminating step
        _, _, _, s_n_prime, d_n = self.n_step_buffer[-1] # Last element in the deque
        if self.n_step_buffer[-1][4]: # if done is true for the last element
            # if the last element is done, this becomes the s_n_prime and done_n_step is True
            done_n_step = True
        else:
            # If not done, then s_n_prime is the next state from n_step_buffer[N-1]
            # and done_n_step is False because the N-step sequence didn't end due to termination
            done_n_step = False 
            
        # Create N-step experience tuple
        n_step_experience = NStepExperience(
            state=s_t, 
            action=a_t, 
            n_step_reward=n_step_reward, 
            next_n_state=s_n_prime, 
            done=done_n_step, # This 'done' refers to the state of s_n_prime
            gamma_n=final_gamma_n
        )
        
        # Add to the main replay buffer
        self.buffer.append(n_step_experience)
        
    def sample(self, batch_size: int) -> List[NStepExperience]:
        """Randomly sample a batch of N-step experiences."""
        if batch_size > len(self.buffer):
            logger.warning(f"Requested batch_size {batch_size} > buffer size {len(self.buffer)}. Sampling all.")
            return list(self.buffer)
        return random.sample(self.buffer, k=batch_size)

    def __len__(self) -> int:
        """Return current size of the main N-step experience buffer."""
        return len(self.buffer) 