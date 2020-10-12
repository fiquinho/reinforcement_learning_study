import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods.envs import Environment, Episode
from environments.move_to_goal import MoveToGoalSimple


class MoveToGoalSimpleSmallEnvironment(Environment):

    def __init__(self):
        env = MoveToGoalSimple(board_x=3, board_y=3, goal_reward=1, move_reward=-1,
                               game_end=10, goal_initial_pos=(2, 2), player_initial_pos=(0, 0))
        action_space = env.action_space
        state_space = env.board_x + env.board_y
        actions = env.actions
        state_names = [f"x{i}" for i in range(env.board_x)] + [f"x{j}" for j in range(env.board_y)]

        Environment.__init__(self, env, action_space, state_space, actions, state_names)

    def reset_environment(self):
        self.env.prepare_game()

    def get_environment_state(self) -> np.array:
        converted_sate = self.convert_sate(self.env.get_state())
        return converted_sate

    def environment_step(self, action: int) -> (np.array, float, bool):
        next_state, reward, done = self.env.step(int(action))
        return next_state, reward, done

    def get_possible_states(self) -> np.array:
        possible_states = []
        for x in range(self.env.board_x):
            for y in range(self.env.board_y):
                possible_states.append((x, y))
        return possible_states

    def render_environment(self):
        self.env.display_game()

    @staticmethod
    def win_condition(episode: Episode):
        return None

    def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
        return None, None

    # def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
    #     possible_states = self.get_possible_states()
    #     converted_possible_states = self.get_converted_possible_states(possible_states)
    #     logits = self.policy(np.array(converted_possible_states))
    #     states_predictions = self.policy.get_probabilities(logits)
    #     action_space = self.game.action_space
    #     xs = [state[0] for state in possible_states]
    #     ys = [state[1] for state in possible_states]
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #
    #     for action in range(action_space):
    #         action_predictions = [prediction[action] for prediction in states_predictions]
    #         ax.scatter(xs, ys, action_predictions, marker="o", label=self.game.actions[action])
    #
    #     ax.legend()
    #
    #     if save_fig is not None:
    #         fig.savefig(save_fig)
    #         plt.close(fig)
    #
    #     if show_plot:
    #         plt.show()
    #
    #     return possible_states, states_predictions

    def convert_sate(self, state) -> np.array:
        converted_sate = np.zeros(self.env.board_x + self.env.board_y)
        converted_sate[state[0]] = 1
        converted_sate[state[1] + self.env.board_x] = 1
        return np.array(converted_sate)

    def get_converted_possible_states(self, possible_states: list) -> list:
        converted_possible_states = []
        for state in possible_states:
            converted_sate = self.convert_sate(state)
            converted_possible_states.append(converted_sate)
        return converted_possible_states

    def close(self):
        self.env.close()
