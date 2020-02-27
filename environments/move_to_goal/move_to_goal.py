import time
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class GameObject(object):

    def __init__(self, name: str):
        self.name = name
        self.actions = None

    def assign_actions(self, actions: dict):
        self.actions = actions


class MoveToGoal(object):

    def __init__(self, board_x: int, board_y: int):

        self.board_x = board_x
        self.board_y = board_y
        self.player = None
        self.goal = None
        self.board = None
        self.positions = {"player": None, "goal": None}
        self.colors = {"player": (255, 150, 0),
                       "goal": (0, 255, 0)}
        self.actions = ["up", "right", "down", "left"]

    def add_player(self, player: GameObject):
        self.player = player

    def add_goal(self, goal: GameObject):
        self.goal = goal

    def assign_player_actions(self):
        self.player.assign_actions(self.actions)

    def generate_board(self):
        self.board = np.zeros((self.board_x, self.board_y, 3), dtype=np.uint8)
        self.board[self.positions["player"]] = self.colors["player"]
        self.board[self.positions["goal"]] = self.colors["goal"]

    def prepare_random_game(self):
        self.positions["player"] = (np.random.randint(0, self.board_y), np.random.randint(0, self.board_x))
        self.positions["goal"] = (np.random.randint(0, self.board_y), np.random.randint(0, self.board_x))
        while self.positions["player"] == self.positions["goal"]:
            self.positions["goal"] = (np.random.randint(0, self.board_y), np.random.randint(0, self.board_x))

        self.generate_board()
        self.assign_player_actions()

    def prepare_game(self, player_pos: Tuple[int, int], goal_pos: Tuple[int, int]):

        if player_pos == goal_pos:
            raise ValueError(f"The goal and player position can't be the same. Both are {player_pos}")

        self.positions["player"] = player_pos
        self.positions["goal"] = goal_pos

        self.generate_board()
        self.assign_player_actions()

    def display_game(self):
        board_image = np.transpose(self.board, (1, 0, 2))

        img = Image.fromarray(board_image, 'RGB')
        img = img.resize((self.board_x * 20, self.board_y * 20), cv2.INTER_AREA)
        cv2.imshow("image", np.array(img))

    def execute_player_action(self, action: int):
        action = self.actions[action]
        player_x = self.positions["player"][0]
        player_y = self.positions["player"][1]

        if action == "up":
            if player_y < self.board_y - 1:
                player_y += 1
        elif action == "right":
            if player_x < self.board_x - 1:
                player_x += 1
        elif action == "down":
            if player_y > 0:
                player_y -= 1
        elif action == "left":
            if player_x > 0:
                player_x -= 1
        else:
            raise ValueError(f"Wrong action: {action}")

        self.positions["player"] = (player_x, player_y)

        self.generate_board()

    def play_game(self, agent, human_view: bool=False):

        while True:
            self.display_game()
            action = agent.produce_action()
            self.execute_player_action(action)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if human_view:
                time.sleep(.5)
