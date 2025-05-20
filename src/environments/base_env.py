from abc import ABC, abstractmethod

class BaseEnv(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset(self):
        """Resets the environment to an initial state and returns an initial observation.
        
        Returns:
            observation: The initial observation of the space.
        """
        pass

    @abstractmethod
    def step(self, action):
        """Run one timestep of the environment's dynamics.
        
        Args:
            action: An action provided by the agent.
            
        Returns:
            observation: Agent's observation of the current environment.
            reward: Amount of reward returned after previous action.
            terminated: Whether the episode has ended (e.g., due to reaching a terminal state).
            truncated: Whether the episode has been truncated (e.g., due to a time limit).
            info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        pass

    @abstractmethod
    def render(self, mode='human'):
        """Renders the environment.
        
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        
        Args:
            mode (str): The mode to render with.
        """
        pass

    @abstractmethod
    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    @property
    @abstractmethod
    def action_space(self):
        """Return the action space of the environment."""
        pass

    @property
    @abstractmethod
    def observation_space(self):
        """Return the observation space of the environment."""
        pass 