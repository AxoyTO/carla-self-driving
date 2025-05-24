from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        super().__init__()

    @abstractmethod
    def select_action(self, observation, epsilon=0.0):
        """Selects an action based on the current observation.

        Args:
            observation: The current state of the environment.
            epsilon: Exploration rate, if applicable (e.g., for epsilon-greedy policies).

        Returns:
            action: The action selected by the agent.
        """
        pass

    @abstractmethod
    def learn(self, experiences):
        """Updates the agent's knowledge based on a batch of experiences.

        Args:
            experiences: A batch of (state, action, reward, next_state, done) tuples.
        """
        pass

    # Optional: methods for saving/loading agent models
    # @abstractmethod
    # def save(self, path):
    #     pass

    # @abstractmethod
    # def load(self, path):
    #     pass 