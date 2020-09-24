import csv
from copy import deepcopy
from pathlib import Path

import numpy as np

from .move_to_goal import GameObject, MoveToGoal, DEFAULT_COLORS


class MoveToGoalHoles(MoveToGoal):

    def __init__(self, board_file: Path, goal_reward: int, move_reward: int,
                 hole_reward: int, game_end: int):

        self.hole_reward = hole_reward
        self.board_name = board_file.stem
        self.player_initial_pos = None
        self.goal_initial_pos = None
        self.initial_board = None

        self.player = None
        self.goal = None
        self.holes = []
        state_space = 2
        game_configurations = f"GR{goal_reward}_HR{hole_reward}_MR{move_reward}_ge{game_end}"

        self.generate_new_game(board_file)

        MoveToGoal.__init__(self, len(self.initial_board), len(self.initial_board[0]), goal_reward,
                            move_reward, game_end, state_space, game_name=self.board_name,
                            game_configs=game_configurations)

    def generate_new_game(self, board_file: Path):
        board_blueprint = []
        with open(board_file, "r", newline='', encoding="utf8") as csvfile:
            reader = csv.reader(csvfile, delimiter="\t")
            for row in reader:
                board_blueprint.append(row)
        board_blueprint.reverse()

        board = []
        for y, row in enumerate(board_blueprint):
            board_row = []
            for x, element in enumerate(row):
                if element == "P":
                    self.player_initial_pos = (x, y)
                    board_row.append(DEFAULT_COLORS["player"])
                elif element == "G":
                    self.goal_initial_pos = (x, y)
                    board_row.append(DEFAULT_COLORS["goal"])
                elif element == "H":
                    self.holes.append((x, y))
                    board_row.append(DEFAULT_COLORS["hole"])
                else:
                    board_row.append(DEFAULT_COLORS["floor"])

            board.append(board_row)

        self.initial_board = np.array(board, dtype=np.uint8).transpose(1, 0, 2)

    def update_board(self):
        self.board[self.player.last_position] = DEFAULT_COLORS["floor"]
        self.board[self.player.position] = self.player.color

    def prepare_game(self):

        self.player = GameObject(self.player_initial_pos, "player", DEFAULT_COLORS["player"])
        self.goal = GameObject(self.goal_initial_pos, "goal", DEFAULT_COLORS["goal"])
        self.steps_played = 0
        self.board = deepcopy(self.initial_board)

    def execute_object_action(self, game_object: GameObject, action: int):
        action = self.actions[action]
        move_results = self.get_move_results(game_object.position, action)
        game_object.change_position(move_results)

        self.update_board()

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

    def get_state(self) -> tuple:
        return self.player.position

    def step(self, player_action: int) -> (tuple, float, bool):

        self.execute_object_action(self.player, player_action)

        if self.player.position == self.goal.position:
            reward = self.goal_reward
            done = True
        elif self.player.position in self.holes:
            reward = self.hole_reward
            done = True
        else:
            reward = self.move_reward
            done = False

        self.steps_played += 1
        if self.steps_played >= self.game_end:
            done = True

        state = self.get_state()

        return state, reward, done

    def specific_step_results(self, state: tuple, action: int) -> (tuple, float, bool):
        action = self.actions[action]
        new_position = self.get_move_results(state, action)

        if new_position == self.goal.position:
            reward = self.goal_reward
            done = True
        elif self.player.position in self.holes:
            reward = self.hole_reward
            done = True
        else:
            reward = self.move_reward
            done = False

        return new_position, reward, done
