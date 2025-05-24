import random
from collections import namedtuple, deque
import numpy as np

from .base_buffer import BaseBuffer

# Define a named tuple for experiences for clarity
Experience = namedtuple("Experience", 
                        field_names=["state", "action", "reward", "next_state", "done"])

class UniformReplayBuffer(BaseBuffer):
    def __init__(self, capacity):
        """
        Initialize a UniformReplayBuffer.
        Args:
            capacity (int): Maximum size of the buffer.
        """
        super().__init__(capacity) # Initializes self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        Args:
            batch_size (int): Number of experiences to sample.
        Returns:
            A list of experience tuples.
        """
        if batch_size > len(self.buffer):
            # Not enough samples yet, perhaps return what we have or raise error
            # For now, let's return all if batch_size is too large, 
            # but DQNAgent usually waits until len(buffer) > batch_size
            # However, random.sample will raise ValueError if population is smaller than k.
            experiences = list(self.buffer) # Return all if not enough, or handle in agent
        else:
            experiences = random.sample(self.buffer, k=batch_size)
        
        # The DQNAgent currently expects to convert these to numpy arrays and then tensors.
        # So returning a list of namedtuples is fine.
        # Alternatively, one could pre-process them into stacked numpy arrays here.
        # For example:
        # states = np.vstack([e.state for e in experiences if e is not None])
        # actions = np.vstack([e.action for e in experiences if e is not None])
        # ... and so on, then convert to tensors.
        # For now, let DQNAgent handle this conversion for flexibility.

        return experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

# Example Usage (for testing)
# if __name__ == '__main__':
#     buffer = UniformReplayBuffer(capacity=100)
#     print(f"Initial buffer length: {len(buffer)}")

#     # Add some dummy experiences
#     for i in range(10):
#         state = np.random.rand(4) # Dummy state
#         action = random.randint(0, 1)
#         reward = random.random()
#         next_state = np.random.rand(4)
#         done = random.choice([True, False])
#         buffer.add(state, action, reward, next_state, done)
    
#     print(f"Buffer length after adding 10 experiences: {len(buffer)}")

#     # Sample experiences
#     if len(buffer) >= 5:
#         sampled_experiences = buffer.sample(batch_size=5)
#         print(f"Sampled {len(sampled_experiences)} experiences.")
#         for exp in sampled_experiences:
#             print(exp)
    
#     # Test sampling when batch_size > len(buffer)
#     small_buffer = UniformReplayBuffer(capacity=10)
#     for i in range(3):
#         small_buffer.add(f"s{i}", i, i, f"ns{i}", False)
#     print(f"Small buffer length: {len(small_buffer)}")
#     # samples = small_buffer.sample(5) # This would cause random.sample to error if not handled
#     # print(f"Samples from small buffer (requested 5, got {len(samples)}): {samples}")

#     # Test max capacity
#     for i in range(120):
#         buffer.add(f"s{i}", i, i, f"ns{i}", False)
#     print(f"Buffer length after adding 120 (capacity 100): {len(buffer)}")
#     # First element should be s20, not s0
#     if len(buffer) > 0:
#         print(f"First element in buffer: {buffer.buffer[0].state}") 
#         assert buffer.buffer[0].state == "s20" 