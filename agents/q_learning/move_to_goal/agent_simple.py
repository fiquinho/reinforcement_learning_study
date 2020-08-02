import argparse
import logging
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from environments.move_to_goal.mtg_simple import MoveToGoalSimple
from agents.q_learning.move_to_goal.agent import MoveToGoalQAgent
from code_utils.logger_utils import prepare_stream_logger


logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)

# Default values
BOARD_SIZE = (7, 10)
GOAL_REWARD = 1
MOVE_REWARD = -1
EPISODES = 5000
CYCLES = 4
GAME_END = 200
EPSILON = 1
LEARNING_RATE = 0.1
DISCOUNT = 0.95


def main():
    parser = argparse.ArgumentParser(description="Q Learning agent that plays the MoveToGoal hard environment.")
    parser.add_argument("--board_size", type=int, nargs="*", default=BOARD_SIZE,
                        help="Board size. X and Y vales. Example: --board_size 6 6")
    parser.add_argument("--player_pos", type=int, nargs="*", default=None,
                        help="Player starting position. Defaults to random starting position. "
                             "Example: --player_pos 0 0")
    parser.add_argument("--goal_pos", type=int, nargs="*", default=None,
                        help="Goal starting position. Defaults to top right corner (Max X and Y possible)."
                             "Example: --goal_pos 4 5")
    parser.add_argument("--episodes", type=int, default=EPISODES,
                        help="The number of episodes to play in a single cycle.")
    parser.add_argument("--cycles", type=int, default=CYCLES,
                        help="The number of episodes to play in a single cycle.")
    parser.add_argument("--show_every", type=int, default=None,
                        help="Defaults to 0.1 of the episodes.")
    parser.add_argument("--goal_reward", type=int, default=GOAL_REWARD,
                        help="The reward when the player reaches the goal.")
    parser.add_argument("--move_reward", type=int, default=MOVE_REWARD,
                        help="The reward when the player moves and doesn't reach the goal.")
    parser.add_argument("--epsilon", type=int, default=EPSILON,
                        help="The starting exploration rate. In range [0, 1).")
    parser.add_argument("--discount", type=int, default=DISCOUNT,
                        help="The cumulative reward discount.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="The learning rate to use during training.")
    parser.add_argument("--game_end", type=int, default=GAME_END,
                        help="How many steps before the game ends.")
    parser.add_argument("--plot_game", action="store_true", default=False,
                        help="Activate to plot the agent playing the game during training.")
    parser.add_argument("--experiments_dir", type=str, default=None,
                        help="Pass a directory to save the model.")
    parser.add_argument("--replace", action="store_true", default=False,
                        help="Activate overwrite an experiment in memory.")
    args = parser.parse_args()

    board_size = args.board_size
    if len(board_size) != 2:
        raise ValueError(f"The board size must be 2 values. "
                         f"Found ( {len(board_size)} ) values = {board_size}")

    player_pos = tuple(args.player_pos) if args.player_pos is not None else None
    goal_pos = tuple(args.goal_pos) if args.goal_pos is not None else (board_size[0] - 1, board_size[1] - 1)

    if args.show_every is None:
        args.show_every = int(args.episodes * args.cycles * 0.1)

    experiments_dir = Path(args.experiments_dir) if args.experiments_dir is not None else None

    test_game = MoveToGoalSimple(board_x=board_size[0],
                                 board_y=board_size[1],
                                 goal_reward=args.goal_reward,
                                 move_reward=args.move_reward,
                                 game_end=args.game_end,
                                 goal_initial_pos=goal_pos,
                                 player_initial_pos=player_pos)
    test_agent = MoveToGoalQAgent(game=test_game)
    test_agent.train_agent(episodes=args.episodes, epsilon=args.epsilon, plot_game=args.plot_game,
                           show_every=args.show_every, learning_rate=args.learning_rate,
                           discount=args.discount, cycles=args.cycles, save_model=experiments_dir,
                           replace=args.replace)


if __name__ == '__main__':
    main()
