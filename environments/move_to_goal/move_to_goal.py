import time

import cv2
import numpy as np
from PIL import Image


class GameObject(object):

    def __init__(self, name: str):
        self.name = name
        self.actions = None

    def assign_actions(self, actions: dict):
        self.actions = actions

    def produce_action(self):

        if self.actions is None:
            raise ValueError("Please set the object actions attribute first")

        return np.random.randint(1, len(self.actions) + 1)


class MoveToGoal(object):

    def __init__(self, board_height: int, board_width: int):

        self.board_height = board_height
        self.board_width = board_width
        self.player = None
        self.goal = None
        self.board = None
        self.positions = {"player": None, "goal": None}
        self.colors = {"player": (255, 150, 0),
                       "goal": (0, 255, 0)}
        self.actions = {1: "up", 2: "right", 3: "down", 4: "left"}

    def add_player(self, player: GameObject):
        self.player = player

    def add_goal(self, goal: GameObject):
        self.goal = goal

    def assign_player_actions(self):
        self.player.assign_actions(self.actions)

    def generate_board(self):
        self.board = np.zeros((self.board_height, self.board_width, 3), dtype=np.uint8)
        self.board[self.positions["player"]] = self.colors["player"]
        self.board[self.positions["goal"]] = self.colors["goal"]

    def prepare_game(self):
        self.positions["player"] = (np.random.randint(0, self.board_height), np.random.randint(0, self.board_width))
        self.positions["goal"] = (np.random.randint(0, self.board_height), np.random.randint(0, self.board_width))
        while self.positions["player"] == self.positions["goal"]:
            self.positions["goal"] = (np.random.randint(0, self.board_height), np.random.randint(0, self.board_width))

        self.generate_board()
        self.assign_player_actions()

    def display_game(self, log_frame_time: bool=False):
        frame_start = None
        if log_frame_time:
            frame_start = time.time()
        img = Image.fromarray(self.board, 'RGB')
        img = img.resize((200, 200), cv2.INTER_AREA)
        cv2.imshow("image", np.array(img))

        if log_frame_time:
            frame_time = time.time() - frame_start
            print(f"Frame time: {frame_time * 1_000} ms")

    def execute_player_action(self):
        action = self.player.produce_action()
        player_x = self.positions["player"][0]
        player_y = self.positions["player"][1]

        if action == 1:
            if player_x < self.board_height - 1:
                player_x += 1
        elif action == 2:
            if player_y < self.board_width - 1:
                player_y += 1
        elif action == 3:
            if player_x > 0:
                player_x -= 1
        elif action == 4:
            if player_y > 0:
                player_y -= 1

        self.positions["player"] = (player_x, player_y)

        self.generate_board()

    def play_game(self, logs: bool=False, human_view: bool=False):

        while True:
            self.display_game(logs)
            self.execute_player_action()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if human_view:
                time.sleep(.5)


test_game = MoveToGoal(10, 10)
test_game.add_player(GameObject("player1"))
test_game.add_goal(GameObject("goal"))
test_game.prepare_game()
test_game.play_game(logs=True, human_view=True)
