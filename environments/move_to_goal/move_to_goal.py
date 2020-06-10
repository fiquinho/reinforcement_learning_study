from typing import Tuple

import cv2
import numpy as np
from PIL import Image


DEFAULT_COLORS = {"player": (255, 0, 0),
                  "goal": (0, 255, 0),
                  "enemy": (0, 0, 255)}


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

    def __init__(self, board_x: int, board_y: int, goal_reward: int, move_reward: int, game_end: int):
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
        self.prepare_game()
        self.actions = ["up", "right", "down", "left"]

    def get_board_size(self) -> Tuple[int, int]:
        return self.board_x, self.board_y

    def display_game(self):
        board_image = np.flip(np.transpose(self.board, (1, 0, 2)), 0)

        img = Image.fromarray(board_image, 'RGB')
        img = img.resize((self.board_x * 20, self.board_y * 20), cv2.INTER_AREA)
        cv2.imshow("image", np.array(img))

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
