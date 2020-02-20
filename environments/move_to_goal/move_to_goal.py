import cv2
import numpy as np
from PIL import Image


class GameObject(object):

    def __init__(self, name: str, idx: int):
        self.name = name
        self.idx = idx


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

    def add_player(self, player: GameObject):
        self.player = player

    def add_goal(self, goal: GameObject):
        self.goal = goal

    def prepare_game(self):
        self.board = np.zeros((self.board_height, self.board_width, 3), dtype=np.uint8)
        self.positions["player"] = (np.random.randint(0, self.board_height), np.random.randint(0, self.board_width))
        self.positions["goal"] = (np.random.randint(0, self.board_height), np.random.randint(0, self.board_width))
        while self.positions["player"] == self.positions["goal"]:
            self.positions["goal"] = (np.random.randint(0, self.board_height), np.random.randint(0, self.board_width))

        self.board[self.positions["player"]] = self.colors["player"]
        self.board[self.positions["goal"]] = self.colors["goal"]

    def display_game(self):
        print(self.board)
        img = Image.fromarray(self.board, 'RGB')
        img = img.resize((200, 200))
        while True:
            cv2.imshow("image", np.array(img))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


test_game = MoveToGoal(10, 10)
test_game.add_player(GameObject("player1", 1))
test_game.add_goal(GameObject("goal", 2))
test_game.prepare_game()
test_game.display_game()
