import argparse
import logging
import sys
import os
import shutil
from pathlib import Path

import tensorflow as tf

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from code_utils.logger_utils import prepare_stream_logger, prepare_file_logger
from agents.deep_q_learning.mountain_car.agent import MountainCarAgent, AgentConfig


logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


EXPERIMENTS_DIR = Path(Path.home(), "rl_experiments", "deep_q_learning")
CONFIG_FILE = Path(SCRIPT_DIR.parent, "configurations", "default.json")


def main():
    parser = argparse.ArgumentParser(description="Q Learning agent that plays the MoveToGoal hard environment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('REQUIRED named arguments')
    required_named.add_argument("--name", type=str, required=True,
                                help="The name of this experiment. The experiments files "
                                     "get saved under this name.")
    parser.add_argument("--config_file", type=str, default=CONFIG_FILE,
                        help="Configuration file for the experiment.")
    parser.add_argument("--output_dir", type=str, default=EXPERIMENTS_DIR,
                        help="Where to save the experiments files")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Activate to run Tensorflow in eager mode.")
    parser.add_argument("--replace", action="store_true", default=False,
                        help="Activate to replace old experiment in the output folder.")
    args = parser.parse_args()

    # On debug mode all functions are executed normally (eager mode)
    if args.debug:
        tf.config.run_functions_eagerly(True)

    config_file = Path(args.config_file)
    config = AgentConfig(args.name, config_file)

    # Create experiment folder and handle old results
    output_dir = Path(args.output_dir)
    game_experiments_dir = Path(output_dir, "mountain_car")
    game_experiments_dir.mkdir(exist_ok=True, parents=True)
    agent_folder = Path(game_experiments_dir, config.name)
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
    agent = MountainCarAgent(min_replay_memory_size=config.min_replay_memory_size,
                             update_target_every=config.update_target_every,
                             replay_memory_size=config.replay_memory_size,
                             batch_size=config.batch_size,
                             layer_size=config.hidden_layer_size,
                             learning_rate=config.learning_rate,
                             double_q_learning=config.double_q_learning,
                             activation=config.activation,
                             hidden_layers_count=config.hidden_layers_count)
    agent.train_agent(episodes=config.episodes,
                      epsilon=config.epsilon,
                      plot_game=config.plot_game,
                      show_every=show_every,
                      save_model=agent_folder,
                      discount=config.discount,
                      cycles=config.cycles,
                      save_q_values_every=config.save_q_values_every)

    results = agent.test_agent(episodes=1000)

    logger.info(f"Agent performance = {sum(results) * 100 / len(results)} % of Wins")


if __name__ == '__main__':
    main()
