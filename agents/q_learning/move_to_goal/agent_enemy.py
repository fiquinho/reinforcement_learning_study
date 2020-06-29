import argparse
import logging
import sys
import os
from pathlib import Path

from matplotlib import style

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from environments.move_to_goal.mtg_enemy import MoveToGoalEnemy
from agents.q_learning.move_to_goal.agent import MoveToGoalQAgent
from code_utils.logger_utils import prepare_stream_logger


logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)


# Default values
BOARD_SIZE = (7, 10)
GOAL_REWARD = 200
MOVE_REWARD = -1
ENEMY_REWARD = -200
EPISODES = 20000
CYCLES = 4
GAME_END = 200
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1

style.use("ggplot")


def main():
    parser = argparse.ArgumentParser(description="Q Learning agent that plays the MoveToGoal hard environment.")
    parser.add_argument("--board_size", type=int, nargs="*", default=BOARD_SIZE,
                        help="Board size. X and Y vales. Example: --board_size 6 6")
    parser.add_argument("--player_pos", type=int, nargs="*", default=None,
                        help="Player starting position. Defaults to random starting position. "
                             "Example: --player_pos 0 0")
    parser.add_argument("--goal_pos", type=int, nargs="*", default=None,
                        help="Goal starting position. Defaults to random starting position."
                             "Example: --goal_pos 4 5")
    parser.add_argument("--enemy_pos", type=int, nargs="*", default=None,
                        help="Goal starting position. Defaults to random starting position."
                             "Example: --enemy_pos 4 5")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--cycles", type=int, default=CYCLES)
    parser.add_argument("--show_every", type=int, default=None, help="Defaults to 10% of the total episodes.")
    parser.add_argument("--goal_reward", type=int, default=GOAL_REWARD)
    parser.add_argument("--move_reward", type=int, default=MOVE_REWARD)
    parser.add_argument("--enemy_reward", type=int, default=ENEMY_REWARD)
    parser.add_argument("--epsilon", type=int, default=EPSILON)
    parser.add_argument("--discount", type=int, default=DISCOUNT)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--game_end", type=int, default=GAME_END)
    parser.add_argument("--plot_game", action="store_true", default=False)
    args = parser.parse_args()

    board_size = args.board_size
    if len(board_size) != 2:
        raise ValueError(f"The board size must be 2 values. "
                         f"Found ( {len(board_size)} ) values = {board_size}")

    player_pos = tuple(args.player_pos) if args.player_pos is not None else None
    goal_pos = tuple(args.goal_pos) if args.goal_pos is not None else None
    enemy_pos = tuple(args.enemy_pos) if args.enemy_pos is not None else None

    if args.show_every is None:
        args.show_every = int(args.episodes * args.cycles * 0.1)

    test_game = MoveToGoalEnemy(board_x=board_size[0],
                                board_y=board_size[1],
                                goal_reward=args.goal_reward,
                                move_reward=args.move_reward,
                                game_end=args.game_end,
                                enemy_reward=args.enemy_reward,
                                enemy_movement="random",
                                player_initial_pos=player_pos,
                                goal_initial_pos=goal_pos,
                                enemy_initial_pos=enemy_pos)
    test_agent = MoveToGoalQAgent(game=test_game)
    test_agent.train_agent(episodes=args.episodes, epsilon=args.epsilon, plot_game=args.plot_game,
                           show_every=args.show_every, learning_rate=args.learning_rate,
                           discount=args.discount, cycles=args.cycles)


if __name__ == '__main__':
    main()
