import os
import sys
from pathlib import Path

import pygame
import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods.envs import Environment, Episode
from ple.games.catcher import Catcher
from ple import PLE


class CatcherEnvironment(Environment):

    def __init__(self):
        env = Catcher(width=32, height=32, init_lives=1)
        self.p = PLE(env, add_noop_action=True)
        self.p.init()
        action_space = len(self.p.getActionSet())
        state_space = len(self.p.getGameState())
        actions = ["left", "right", "nothing"]
        state_names = list(self.p.getGameState().keys())

        Environment.__init__(self, env, action_space, state_space, actions, state_names)

    def reset_environment(self):
        self.p.reset_game()

    def get_environment_state(self) -> np.array:
        state = list(self.p.getGameState().values())
        return np.array(state, dtype=np.float32) / 32.

    def environment_step(self, action: int) -> (np.array, float, bool):

        p_action = self.p.getActionSet()[action]
        reward = self.p.act(p_action)
        done = self.p.game_over()
        if reward == 1:
            done = True
        next_state = self.get_environment_state()
        return next_state, reward, done

    def get_possible_states(self) -> np.array:
        return None

    def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
        return None, None

    def render_environment(self):
        self.p.display_screen = True
        self.p.force_fps = False

    @staticmethod
    def win_condition(episode: Episode):
        return None

    def close(self):
        pygame.quit()
