import random
import numpy as np
from collections import namedtuple, deque
from typing import List, Tuple, Optional
import logging

from .base_buffer import BaseBuffer

logger = logging.getLogger(__name__)

# Define a named tuple for experiences with priorities
Experience = namedtuple("Experience", 
                        field_names=["state", "action", "reward", "next_state", "done"])

class SumTree:
    """Binary tree data structure for efficient priority sampling."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Update sum tree upwards."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float):
        """Find sample on the tree with given cumulative sum."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return total priority sum."""
        return self.tree[0]

    def add(self, priority: float, data):
        """Add experience with given priority."""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        """Update priority of experience at given index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        """Get experience based on cumulative sum."""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]


class PrioritizedReplayBuffer(BaseBuffer):
    """Prioritized Experience Replay Buffer for better learning from important experiences."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4, 
                 beta_frames: int = 100000, epsilon: float = 1e-6):
        """
        Initialize Prioritized Replay Buffer.
        
        Args:
            capacity: Maximum size of the buffer
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames over which beta is annealed
            epsilon: Small constant to prevent zero priorities
        """
        super().__init__(capacity)
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 1
        
        # Track statistics
        self.max_priority = 1.0
        
    def beta(self) -> float:
        """Calculate current beta value for importance sampling."""
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience with maximum priority."""
        experience = Experience(state, action, reward, next_state, done)
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
        
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch with priorities.
        
        Returns:
            experiences: List of sampled experiences
            indices: Indices of sampled experiences (for updating priorities)
            weights: Importance sampling weights
        """
        batch = []
        indices = np.empty((batch_size,), dtype=np.int32)
        weights = np.empty((batch_size, 1), dtype=np.float32)
        priorities = np.empty((batch_size,), dtype=np.float32)
        
        # Calculate priority segment size
        priority_segment = self.tree.total() / batch_size
        current_beta = self.beta()
        
        # Sample from each segment
        for i in range(batch_size):
            a = priority_segment * i
            b = priority_segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            
            # Calculate importance sampling weight
            sampling_prob = priority / self.tree.total()
            weight = (self.tree.n_entries * sampling_prob) ** (-current_beta)
            
            batch.append(experience)
            indices[i] = idx
            weights[i] = weight
            priorities[i] = priority
            
        # Normalize weights
        weights /= weights.max()
        
        self.frame += 1
        
        return batch, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for given experiences."""
        for idx, priority in zip(indices, priorities):
            # Clip priority to avoid numerical issues
            priority = max(priority, self.epsilon)
            priority = priority ** self.alpha
            
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return self.tree.n_entries
    
    def get_statistics(self) -> dict:
        """Return buffer statistics for monitoring."""
        return {
            'size': len(self),
            'capacity': self.capacity,
            'alpha': self.alpha,
            'beta': self.beta(),
            'max_priority': self.max_priority,
            'total_priority': self.tree.total()
        } 