import argparse
import logging
import sys
import os
import json
import shutil
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

import tensorflow as tf

from code_utils.logger_utils import prepare_stream_logger, prepare_file_logger
from agents.deep_q_learning.mountain_car.agent import MountainCarAgent


logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


EPISODES = 600
CYCLES = 2
LEARNING_RATE = 0.001
DISCOUNT = 0.95
EPSILON = 1
REPLAY_MEMORY_SIZE = 20_000
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in memory to start training
BATCH_SIZE = 32  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)
LAYER_SIZE = 30
SAVE_Q_VALUES_EVERY = 10_000  # Number of training steps (aprox: 200 per episode)
EXPERIMENTS_DIR = Path(Path.home(), "rl_experiments", "deep_q_learning")


def main():
    parser = argparse.ArgumentParser(description="Q Learning agent that plays the MoveToGoal hard environment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of current experiment.")
    parser.add_argument("--episodes", type=int, default=EPISODES, help=" ")
    parser.add_argument("--cycles", type=int, default=CYCLES, help=" ")
    parser.add_argument("--show_every", type=int, default=None, help="Defaults to 0.1 * episodes")
    parser.add_argument("--epsilon", type=int, default=EPSILON, help=" ")
    parser.add_argument("--discount", type=int, default=DISCOUNT, help=" ")
    parser.add_argument("--replay_memory_size", type=int, default=REPLAY_MEMORY_SIZE, help=" ")
    parser.add_argument("--min_replay_memory_size", type=int, default=MIN_REPLAY_MEMORY_SIZE, help=" ")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=" ")
    parser.add_argument("--layer_size", type=int, default=LAYER_SIZE, help=" ")
    parser.add_argument("--update_target_every", type=int, default=UPDATE_TARGET_EVERY, help=" ")
    parser.add_argument("--output_dir", type=str, default=EXPERIMENTS_DIR,
                        help="Where to save the experiments files")
    parser.add_argument("--plot_game", action="store_true", default=False, help=" ")
    parser.add_argument("--debug", action="store_true", default=False, help=" ")
    parser.add_argument("--replace", action="store_true", default=False, help=" ")
    parser.add_argument("--learning_rate", default=LEARNING_RATE, type=float, help=" ")
    parser.add_argument("--save_q_values_every", default=SAVE_Q_VALUES_EVERY, type=int, help=" ")
    args = parser.parse_args()

    # On debug mode all functions are executed normally (eager mode)
    if args.debug:
        tf.config.run_functions_eagerly(True)

    # Create experiment folder and handle old results
    output_dir = Path(args.output_dir)
    game_experiments_dir = Path(output_dir, "mountain_car")
    game_experiments_dir.mkdir(exist_ok=True, parents=True)
    agent_folder = Path(game_experiments_dir, args.experiment_name)
    if agent_folder.exists():
        if args.replace:
            shutil.rmtree(agent_folder)
        else:
            raise FileExistsError(f"The experiment {agent_folder} already exists."
                                  f"Change output folder, experiment name or use -replace "
                                  f"to overwrite.")
    agent_folder.mkdir()

    # Save experiments configurations and start experiment log
    args.__dict__["output_dir"] = str(output_dir)
    with open(Path(agent_folder, "configurations.json"), "w", encoding="utf8") as cfile:
        json.dump(args.__dict__, cfile)
    prepare_file_logger(logger, logging.INFO, Path(agent_folder, "experiment.log"))

    show_every = int(args.episodes * 0.1) if args.show_every is None else args.show_every

    # Create and train the agent
    test_agent = MountainCarAgent(min_replay_memory_size=args.min_replay_memory_size,
                                  update_target_every=args.update_target_every,
                                  replay_memory_size=args.replay_memory_size,
                                  batch_size=args.batch_size,
                                  layer_size=args.layer_size,
                                  learning_rate=args.learning_rate)
    test_agent.train_agent(episodes=args.episodes, epsilon=args.epsilon, plot_game=args.plot_game,
                           show_every=show_every, save_model=agent_folder,
                           discount=args.discount, cycles=args.cycles,
                           save_q_values_every=args.save_q_values_every)


if __name__ == '__main__':
    main()
