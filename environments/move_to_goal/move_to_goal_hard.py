from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class GameObject(object):

    def __init__(self, position: Tuple[int, int], name: str):

        self.position = position
        self.name = name

    def change_position(self, new_position: Tuple[int, int]):
        self.position = new_position


class MoveToGoal(object):

    def __init__(self, board_x: int, board_y: int, goal_reward: int, move_reward: int, enemy_reward: int,
                 enemy_movement: str="random"):

        self.board_x = board_x
        self.board_y = board_y
        self.goal_reward = goal_reward
        self.move_reward = move_reward
        self.enemy_reward = enemy_reward
        self.player = None
        self.goal = None
        self.enemy = None
        self.board = None
        self.enemy_movement = enemy_movement
        self.colors = {"player": (255, 150, 0),
                       "goal": (0, 255, 0),
                       "enemy": (0, 0, 255)}
        self.actions = ["up", "right", "down", "left"]

    def get_board_size(self):
        return self.board_x, self.board_y

    def generate_board(self):
        self.board = np.zeros((self.board_x, self.board_y, 3), dtype=np.uint8)
        self.board[self.player.position] = self.colors["player"]
        self.board[self.goal.position] = self.colors["goal"]
        self.board[self.enemy.position] = self.colors["enemy"]

    def prepare_game(self, player_pos: Tuple[int, int]=None, goal_pos: Tuple[int, int]=None,
                     enemy_pos: Tuple[int, int]=None):

        if player_pos is None:
            player_pos = (np.random.randint(0, self.board_x), np.random.randint(0, self.board_y))

        if goal_pos is None:
            goal_pos = (np.random.randint(0, self.board_x), np.random.randint(0, self.board_y))
            while goal_pos == player_pos:
                goal_pos = (np.random.randint(0, self.board_x), np.random.randint(0, self.board_y))

        if enemy_pos is None:
            enemy_pos = (np.random.randint(0, self.board_x), np.random.randint(0, self.board_y))
            while enemy_pos == player_pos or enemy_pos == goal_pos:
                enemy_pos = (np.random.randint(0, self.board_x), np.random.randint(0, self.board_y))

        self.player = GameObject(player_pos, "player")
        self.goal = GameObject(goal_pos, "goal")
        self.enemy = GameObject(enemy_pos, "enemy")

        self.generate_board()

    def display_game(self):
        board_image = np.flip(np.transpose(self.board, (1, 0, 2)), 0)

        img = Image.fromarray(board_image, 'RGB')
        img = img.resize((self.board_x * 20, self.board_y * 20), cv2.INTER_AREA)
        cv2.imshow("image", np.array(img))

    def execute_object_action(self, game_object: GameObject, action: int):
        action = self.actions[action]
        game_object_x, game_object_y = game_object.position

        if action == "up":
            if game_object_y < self.board_y - 1:
                game_object_y += 1
        elif action == "right":
            if game_object_x < self.board_x - 1:
                game_object_x += 1
        elif action == "down":
            if game_object_y > 0:
                game_object_y -= 1
        elif action == "left":
            if game_object_x > 0:
                game_object_x -= 1
        else:
            raise ValueError(f"Wrong action: {action}")

        game_object.change_position((game_object_x, game_object_y))

        self.generate_board()

    def get_state(self):
        return self.player.position, self.goal.get_position(), self.enemy.get_position()

    def step(self, player_action: int) -> (Tuple[Tuple[int, int], Tuple[int, int],
                                                 Tuple[int, int]], float, bool):

        self.execute_object_action(self.player, player_action)
        if self.enemy_movement == "random":
            self.execute_object_action(self.enemy, np.random.randint(0, len(self.actions)))

        if self.player.position == self.goal.position:
            reward = self.goal_reward
            done = True
        elif self.player.position == self.enemy.position:
            reward = self.enemy_reward
            done = True
        else:
            reward = self.move_reward
            done = False

        state = self.get_state()

        return state, reward, done
