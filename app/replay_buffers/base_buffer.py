from abc import ABC, abstractmethod
from collections import deque

class BaseBuffer(ABC):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        super().__init__()

    @abstractmethod
    def add(self, state, action, reward, next_state, done):
        """Adds an experience to the buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size):
        """Samples a batch of experiences from the buffer."""
        pass

    def __len__(self):
        return len(self.buffer) 