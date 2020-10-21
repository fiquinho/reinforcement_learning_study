import os
import sys
from pathlib import Path

import gym
import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods.envs import Environment, Episode


class PendulumEnvironment(Environment):

    def __init__(self):
        env = gym.make("Pendulum-v0")
        action_space = 1
        state_space = env.observation_space.shape[0]
        actions = ["torque"]
        state_names = ["cos(theta)", "sin(theta)", "Theta dot"]

        Environment.__init__(self, env, action_space, state_space, actions, state_names, "continuous")

    def reset_environment(self):
        self.env.reset()

    def get_environment_state(self) -> np.array:
        theta, thetadot = self.env.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def environment_step(self, action: float) -> (np.array, float, bool):
        """Do a move in the environment.

        Args:
            action: In the range [-1, 1]

        Returns:
            The next state, the reward obtained by doing the action, and if the environment is terminated
        """
        action = action * 2.
        next_state, reward, done, _ = self.env.step(action)

        reward = reward / 16.2736044

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
