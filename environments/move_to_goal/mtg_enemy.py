from typing import Tuple

import numpy as np

from .move_to_goal import GameObject, MoveToGoal, DEFAULT_COLORS


class MoveToGoalEnemy(MoveToGoal):

    def __init__(self, board_x: int, board_y: int, goal_reward: int, move_reward: int,
                 enemy_reward: int, game_end: int, enemy_movement: str="random"):

        self.enemy_reward = enemy_reward
        self.player = None
        self.goal = None
        self.enemy = None
        self.enemy_movement = enemy_movement
        self.state_space = 6

        MoveToGoal.__init__(self, board_x, board_y, goal_reward, move_reward, game_end)

    def update_board(self):
        board = np.zeros((self.board_x, self.board_y, 3), dtype=np.uint8)
        board[self.player.position] = self.player.color
        board[self.goal.position] = self.goal.color
        board[self.enemy.position] = self.enemy.color
        return board

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

        self.player = GameObject(player_pos, "player", DEFAULT_COLORS["player"])
        self.goal = GameObject(goal_pos, "goal", DEFAULT_COLORS["goal"])
        self.enemy = GameObject(enemy_pos, "enemy", DEFAULT_COLORS["enemy"])
        self.steps_played = 0

        self.board = self.update_board()

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

        self.board = self.update_board()

    def get_state(self) -> Tuple[int, int, int, int, int, int]:
        return self.player.position + self.goal.position + self.enemy.position

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

        self.steps_played += 1
        if self.steps_played >= self.game_end:
            done = True

        state = self.get_state()

        return state, reward, done
