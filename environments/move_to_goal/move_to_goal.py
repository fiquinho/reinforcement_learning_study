import os
from typing import Tuple

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import numpy as np


PYGAME_SCALE = 50
DEFAULT_COLORS = {"player": (0, 0, 255),
                  "goal": (0, 255, 0),
                  "enemy": (255, 0, 0)}


class GameObject(object):
    """
    Object that interacts in a move to goal environment.
    """

    def __init__(self, position: Tuple[int, int], name: str, color: Tuple[int, int, int]):
        """
        Creates a new game object instance.

        :param position: x and y initial positions for the object in that order
        :param name: The name of the object (i.e.: player, goal, etc)
        :param color: The color of this object when it's rendered (RGB)
        """
        self.position = position
        self.name = name
        self.color = color

    def change_position(self, new_position: Tuple[int, int]):
        """
        Change the position of the object

        :param new_position: x and y new positions for the object in that order
        """
        self.position = new_position


class MoveToGoal(object):
    """
    Base class for the move to goal environment.
    All different versions of the game should be constructed using this.
    """

    def __init__(self, board_x: int, board_y: int, goal_reward: int, move_reward: int,
                 game_end: int, state_space: int):
        """
        :param board_x: Board width
        :param board_y: Board height
        :param goal_reward: Reward given when reaching the GOAL (WIN)
        :param move_reward: Reward given when moving and not reaching a terminal state
        :param game_end: How many steps before the game ends
        """
        self.board_x = board_x
        self.board_y = board_y
        self.goal_reward = goal_reward
        self.move_reward = move_reward
        self.game_end = game_end
        self.steps_played = 0
        self.board = None
        self.state_space = state_space
        self.prepare_game()
        self.actions = ["up", "right", "down", "left"]
        self.action_space = len(self.actions)

    def get_board_size(self) -> Tuple[int, int]:
        return self.board_x, self.board_y

    def display_game(self, title: str = None, q_table=None):
        win = pygame.display.set_mode((self.board_x * PYGAME_SCALE, self.board_y * PYGAME_SCALE))
        title = "Move to goal" if title is None else title
        pygame.display.set_caption(title)
        if q_table is not None:
            shape = np.shape(q_table)
            if len(shape) != 3 or shape[2] != 4:
                raise ValueError(
                    f"q_table must have 3 dimentions and four elements in its deepest dimention to be plotted ")

            q_table = np.flip(q_table, 1)
            for i in range(self.board_x):
                for j in range(self.board_y):
                    for q_i in range(4):
                            x_gen = i * PYGAME_SCALE
                            y_gen = j * PYGAME_SCALE
                            if q_i == 0:
                                x_sub = x_gen + PYGAME_SCALE / 2 * 0.5
                                y_sub = y_gen
                            if q_i == 1:
                                x_sub = x_gen + PYGAME_SCALE / 2
                                y_sub = y_gen + PYGAME_SCALE / 2 * 0.5
                            if q_i == 2:
                                x_sub = x_gen + PYGAME_SCALE / 2 * 0.5
                                y_sub = y_gen + PYGAME_SCALE / 2
                            if q_i == 3:
                                x_sub = x_gen
                                y_sub = y_gen + PYGAME_SCALE / 2 * 0.55
                            q_cell_value = q_table[i][j][q_i]
                            cell_max = np.amax(q_table[i][j])
                            cell_min = np.amin(q_table[i][j])
                            red = np.interp(
                                q_cell_value, [cell_min, cell_max], [0, 255])
                            color = [red, 10, 10]
                            pygame.draw.rect(win, color, (x_sub, y_sub,
                                                          PYGAME_SCALE / 4, PYGAME_SCALE / 4))

        board_image = np.flip(self.board, 1)
        draw_objects = []
        for i in range(self.board_x):
            for j in range(self.board_y):
                if not (board_image[i, j] == (0, 0, 0)).all():
                    draw_objects.append((i, j, board_image[i, j]))

        for item in draw_objects:
            pygame.draw.rect(win, item[2], (item[0] * PYGAME_SCALE, item[1] * PYGAME_SCALE,
                                            PYGAME_SCALE, PYGAME_SCALE))

        pygame.display.update()

    def update_board(self):
        raise NotImplementedError()

    def prepare_game(self, **kwargs):
        raise NotImplementedError()

    def execute_object_action(self, **kwargs):
        raise NotImplementedError()

    def get_state(self):
        raise NotImplementedError()

    def step(self, **kwargs):
        raise NotImplementedError()

    def specific_step_results(self, **kwargs):
        raise NotImplementedError()
