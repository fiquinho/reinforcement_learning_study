import argparse
import logging
import sys
import os
import shutil
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

import tensorflow as tf

from code_utils.logger_utils import prepare_stream_logger, prepare_file_logger
from code_utils.config_utils import BaseConfig
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


class AgentConfig(BaseConfig):

    def __init__(self, config_file: Path):
        BaseConfig.__init__(self, config_file)

        self.experiment_name = self.config_file.stem
        self.episodes = self.config_dict["episodes"]
        self.cycles = self.config_dict["cycles"]
        self.show_every = self.config_dict["show_every"]
        self.epsilon = self.config_dict["epsilon"]
        self.learning_rate = self.config_dict["learning_rate"]
        self.discount = self.config_dict["discount"]
        self.replay_memory_size = self.config_dict["replay_memory_size"]
        self.min_replay_memory_size = self.config_dict["min_replay_memory_size"]
        self.batch_size = self.config_dict["batch_size"]
        self.update_target_every = self.config_dict["update_target_every"]
        self.hidden_layer_size = self.config_dict["hidden_layer_size"]
        self.plot_game = self.config_dict["plot_game"]
        self.save_q_values_every = self.config_dict["save_q_values_every"]


def main():
    parser = argparse.ArgumentParser(description="Q Learning agent that plays the MoveToGoal hard environment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file", type=str, required=True,
                        help="Configuration file for the experiment")
    parser.add_argument("--output_dir", type=str, default=EXPERIMENTS_DIR,
                        help="Where to save the experiments files")
    parser.add_argument("--debug", action="store_true", default=False, help=" ")
    parser.add_argument("--replace", action="store_true", default=False, help=" ")
    args = parser.parse_args()

    # On debug mode all functions are executed normally (eager mode)
    if args.debug:
        tf.config.run_functions_eagerly(True)

    config_file = Path(args.config_file)
    config = AgentConfig(config_file)

    # Create experiment folder and handle old results
    output_dir = Path(args.output_dir)
    game_experiments_dir = Path(output_dir, "mountain_car")
    game_experiments_dir.mkdir(exist_ok=True, parents=True)
    agent_folder = Path(game_experiments_dir, config.experiment_name)
    if agent_folder.exists():
        if args.replace:
            shutil.rmtree(agent_folder)
        else:
            raise FileExistsError(f"The experiment {agent_folder} already exists."
                                  f"Change output folder, experiment name or use -replace "
                                  f"to overwrite.")
    agent_folder.mkdir()

    # Save experiments configurations and start experiment log
    prepare_file_logger(logger, logging.INFO, Path(agent_folder, "experiment.log"))
    config.log_configurations(logger)
    config.copy_config(agent_folder)

    show_every = int(config.episodes * 0.1) if config.show_every is None else config.show_every

    # Create and train the agent
    test_agent = MountainCarAgent(min_replay_memory_size=config.min_replay_memory_size,
                                  update_target_every=config.update_target_every,
                                  replay_memory_size=config.replay_memory_size,
                                  batch_size=config.batch_size,
                                  layer_size=config.hidden_layer_size,
                                  learning_rate=config.learning_rate)
    test_agent.train_agent(episodes=config.episodes,
                           epsilon=config.epsilon,
                           plot_game=config.plot_game,
                           show_every=show_every,
                           save_model=agent_folder,
                           discount=config.discount,
                           cycles=config.cycles,
                           save_q_values_every=config.save_q_values_every)


if __name__ == '__main__':
    main()
