from typing import Tuple

import numpy as np

from .move_to_goal import GameObject, MoveToGoal, DEFAULT_COLORS


class MoveToGoalSimple(MoveToGoal):

    def __init__(self, board_x: int, board_y: int, goal_reward: int, move_reward: int, game_end: int,
                 goal_initial_pos: Tuple[int, int], player_initial_pos: Tuple[int, int]=None):

        self.board_x = board_x
        self.board_y = board_y
        self.state_space = 2
        self.player_initial_pos = player_initial_pos
        self.goal_initial_pos = goal_initial_pos
        self.player = None
        self.goal = None
        game_name = f"move_to_goal_simple"
        game_configurations = f"{board_x}x{board_y}_GR{goal_reward}_MR{move_reward}_ge{game_end}"

        MoveToGoal.__init__(self, board_x, board_y, goal_reward, move_reward,
                            game_end, self.state_space, game_name, game_configurations)

    def update_board(self):
        board = np.zeros((self.board_x, self.board_y, 3), dtype=np.uint8)
        board[self.player.position] = self.player.color
        board[self.goal.position] = self.goal.color
        return board

    def prepare_game(self):

        if self.player_initial_pos is None:
            player_pos = (np.random.randint(0, self.board_x), np.random.randint(0, self.board_y))
        else:
            player_pos = self.player_initial_pos

        self.player = GameObject(player_pos, "player", DEFAULT_COLORS["player"])
        self.goal = GameObject(self.goal_initial_pos, "goal", DEFAULT_COLORS["goal"])
        self.steps_played = 0

        self.board = self.update_board()

    def execute_object_action(self, game_object: GameObject, action: int):
        action = self.actions[action]
        move_results = self.get_move_results(game_object.position, action)
        game_object.change_position(move_results)

        self.board = self.update_board()

    def get_move_results(self, original_position: tuple, action: str) -> tuple:
        x, y = original_position
        new_x, new_y = original_position
        if action == "up":
            if y < self.board_y - 1:
                new_y += 1
        elif action == "right":
            if x < self.board_x - 1:
                new_x += 1
        elif action == "down":
            if y > 0:
                new_y -= 1
        elif action == "left":
            if x > 0:
                new_x -= 1
        else:
            raise ValueError(f"Wrong action: {action}")

        return new_x, new_y

    def specific_step_results(self, state: tuple, action: int):
        action = self.actions[action]
        new_position = self.get_move_results(state, action)

        if new_position == self.goal.position:
            reward = self.goal_reward
            done = True
        else:
            reward = self.move_reward
            done = False

        return new_position, reward, done

    def get_state(self) -> tuple:
        return self.player.position

    def step(self, player_action: int) -> (tuple, float, bool):

        self.execute_object_action(self.player, player_action)

        if self.player.position == self.goal.position:
            reward = self.goal_reward
            done = True
        else:
            reward = self.move_reward
            done = False

        self.steps_played += 1
        if self.steps_played >= self.game_end:
            done = True

        state = self.get_state()

        return state, reward, done
