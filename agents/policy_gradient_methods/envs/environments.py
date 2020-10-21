from pathlib import Path
from typing import List

import gym
import numpy as np


class Episode(object):
    """A single episode of an environment.

    Attributes:
        states: A list with all the states that occurred in the episode
        rewards: A list with all the rewards obtained in the episode
        actions: A list with all the actions taken in the episode
        total_reward: The total reward obtained in the episode
    """
    def __init__(self, states: list, actions: list, rewards: list, action_space: str):
        """Creates a new episode.

        Args:
            states: A list with all the states that occurred in the episode
            rewards: A list with all the rewards obtained in the episode
            actions: A list with all the actions taken in the episode
            action_space: The type of action space. One of ["continuous", "discrete"]
        """
        assert len(states) == len(rewards) == len(actions)
        self.states = np.array(states, dtype=np.float32)
        self.rewards = np.array(rewards, dtype=np.float32)

        if action_space == "discrete":
            self.actions = np.array(actions, dtype=np.int32)
        elif action_space == "continuous":
            self.actions = np.array(actions, dtype=np.float32)
        else:
            raise ValueError(f"Found unsupported action space type = {action_space}")

        self.total_reward = np.sum(self.rewards)

    def __len__(self) -> int:
        """
        Returns:
            The length of the episode.
        """
        return len(self.states)


class EpisodesBatch(object):
    """
    A collection of episodes.
    """
    def __init__(self, max_size: int):
        """
        Creates an empty episodes batch.
        :param max_size: The max number of stored steps.
        """
        self.episodes = []
        self.max_size = max_size
        self.current_size = 0

    def __len__(self) -> int:
        """
        :return the total number of stored steps
        """
        return self.current_size

    def __iter__(self):
        """
        Iterate over the stored episodes and yield one at the time
        :return: An Episode object
        """
        for episode in self.episodes:
            yield episode

    def add_episode(self, episode: Episode):
        """
        Add and episode to the batch. Update number of stored steps.
        :param episode: An Episode object
        :raises ValueError if the episodes batch is full (max number of steps stored)
        """
        if not self.is_full():
            self.episodes.append(episode)
            self.current_size += len(episode)
        else:
            raise ValueError(f"The batch is full! max_size: {self.max_size} -"
                             f" current_size: {self.current_size}")

    def is_full(self) -> bool:
        """
        :return: True if the number of stored steps is more than or equal to the max batch size.
        """
        return self.current_size >= self.max_size


class Environment(object):
    """Base class to create environments that can be used to train a
    policy gradient algorithm. All methods need to be implemented.
    """
    def __init__(self, env, action_space_n: int, state_space_n: int,
                 actions: List[str], state_names: List[str]=None,
                 action_space: str="discrete"):
        """Create a new Environment object.

        Args:
            env: An object with the environment implementation
            action_space_n: The number of possible actions
            state_space_n: The length of the state vector representation
            actions: A list with the actions names
            state_names: A list with the state attributes names
            action_space: The type of action space. One of ["continuous", "discrete"]
        """
        self.env = env
        self.action_space_n = action_space_n
        self.state_space_n = state_space_n
        self.actions = actions
        self.state_names = state_names
        self.action_space = action_space

    def reset_environment(self):
        """Reset the environment to start a new episode."""
        raise NotImplementedError

    def get_environment_state(self) -> np.array:
        """Get the current state of the environment. Must be ready to feed to
        the neural network.

        Returns:
            The current state (np.array)
        """
        raise NotImplementedError

    def environment_step(self, action: int) -> (np.array, float, bool):
        """Make a move in the environment with given action.

        Args:
            action: The action index or action value

        Returns:
            next_environment_state (np.array), reward (float), terminated_environment (bool)
        """
        raise NotImplementedError

    def get_possible_states(self) -> np.array:
        """Returns a list of every possible environment state, or a sample of them.

        Returns:
            List of states ready to be feed into de neural network (np.array)
        """
        raise NotImplementedError

    def policy_values_plot(self, save_fig: Path=None, show_plot: bool=False):
        """
        TODO: Change name and usage (policy_values_info)
        """
        raise NotImplementedError

    def render_environment(self):
        """Render the environment."""
        raise NotImplementedError

    @staticmethod
    def win_condition(episode: Episode):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class CartPoleEnvironment(Environment):

    def __init__(self):
        env = gym.make("CartPole-v0")
        action_space = env.action_space.n
        state_space = env.observation_space.shape[0]
        actions = ["left", "right"]
        state_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]

        Environment.__init__(self, env, action_space, state_space, actions, state_names)

    def reset_environment(self):
        self.env.reset()

    def get_environment_state(self) -> np.array:
        return self.env.state

    def environment_step(self, action: int) -> (np.array, float, bool):
        next_state, reward, done, _ = self.env.step(int(action))
        return next_state, reward, done

    def get_possible_states(self) -> np.array:
        return None

    def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
        return None, None

    def render_environment(self):
        self.env.render()

    @staticmethod
    def win_condition(episode: Episode):
        return episode.total_reward >= 200

    def close(self):
        self.env.close()


class AcrobotEnvironment(Environment):

    def __init__(self):
        env = gym.make("Acrobot-v1")
        action_space = env.action_space.n
        state_space = env.observation_space.shape[0]
        actions = ["left", "null", "right"]
        state_names = ["cos(theta1)", "sin(theta1)", "cos(theta2)",
                       "sin(theta2)", "thetaDot1", "thetaDot2"]

        Environment.__init__(self, env, action_space, state_space, actions, state_names)

    def reset_environment(self):
        self.env.reset()

    def get_environment_state(self) -> np.array:
        s = self.env.state
        return np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]])

    def environment_step(self, action: int) -> (np.array, float, bool):
        next_state, reward, done, _ = self.env.step(int(action))
        return next_state, reward, done

    def get_possible_states(self) -> np.array:
        return None

    def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
        return None, None

    def render_environment(self):
        self.env.render()

    @staticmethod
    def win_condition(episode: Episode):
        return episode.total_reward >= 200

    def close(self):
        self.env.close()


class HeuristicMountainCarEnvironment(Environment):

    def __init__(self):
        env = gym.make("MountainCar-v0")
        action_space = env.action_space.n
        state_space = env.observation_space.shape[0]
        actions = ["left", "null", "right"]
        state_names = ["position", "velocity"]

        Environment.__init__(self, env, action_space, state_space, actions, state_names)

    def reset_environment(self):
        self.env.reset()

    def get_environment_state(self) -> np.array:
        return self.env.state

    def environment_step(self, action: int) -> (np.array, float, bool):
        next_state, reward, done, _ = self.env.step(int(action))
        if done:
            reward += next_state[0] * 100
            if next_state[0] >= 0.5:
                reward += 100
        return next_state, reward, done

    def get_possible_states(self) -> np.array:
        return None

    def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
        return None, None

    def render_environment(self):
        self.env.render()

    @staticmethod
    def win_condition(episode: Episode):
        return episode.states[-1][0] >= 0.5

    def close(self):
        self.env.close()
