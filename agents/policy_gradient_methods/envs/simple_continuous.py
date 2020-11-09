import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods.envs import Environment, Episode
from environments.simple_continuous.simple_countinuous import SimpleContinuous


class SimpleContinuousEnvironment(Environment):

    def __init__(self):
        env = SimpleContinuous(target=4., max_reward=1., target_range=0.01)
        action_space = 1
        state_space = 1
        actions = ["action"]
        state_names = ["unique_state"]

        Environment.__init__(self, env, action_space, state_space,
                             actions, state_names, action_space="continuous")

    def reset_environment(self):
        pass

    def get_environment_state(self) -> np.array:
        state = self.env.get_state()
        return np.array([state])

    def environment_step(self, action: int) -> (np.array, float, bool):

        action = (action * 4.) + 4.

        return self.env.step(action)

    def get_possible_states(self) -> np.array:
        return self.get_environment_state()

    def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
        raise NotImplementedError

    def render_environment(self):
        pass

    @staticmethod
    def win_condition(episode: Episode):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
