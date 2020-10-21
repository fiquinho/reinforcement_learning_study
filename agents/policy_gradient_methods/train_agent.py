import argparse
import logging
import os
import sys
import shutil
import json
import time
from pathlib import Path

import git
import tensorflow as tf

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent))

from agents.policy_gradient_methods import ENVIRONMENTS, PG_METHODS
from code_utils import prepare_file_logger, prepare_stream_logger

logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


EXPERIMENTS_DIR = Path(Path.home(), "rl_experiments", "policy_gradient")
SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
CONFIGS_DIR = Path(SCRIPT_DIR.parent, "configurations")


def main():
    parser = argparse.ArgumentParser(description="Naive Policy Gradient agent that plays the "
                                                 "MountainC simple environment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('REQUIRED named arguments')
    required_named.add_argument("--name", type=str, required=True,
                                help="The name of this experiment. The experiments files "
                                     "get saved under this name.")
    required_named.add_argument("--env", type=str, choices=ENVIRONMENTS.keys(), required=True,
                                help="The environment to solve.")
    required_named.add_argument("--agent", type=str, choices=PG_METHODS.keys(), required=True,
                                help="The policy gradient method to use.")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Configuration file for the experiment. If no file is "
                             "provided use the default configuration for the "
                             "selected environment and agent.")
    parser.add_argument("--desc", type=str, default=None,
                        help="Description of the experiment for future reference.")
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

    # Get git version
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    # Use provided configurations file or the default for the selected environment and agent
    if args.config_file is None:
        config_file = Path(CONFIGS_DIR, f"{args.env}_{args.agent}_default.json")
    else:
        config_file = Path(args.config_file)
    config = PG_METHODS[args.agent]["config"](args.name, config_file, args.desc)

    # Create experiment folder and handle old results
    output_dir = Path(args.output_dir)
    agent_folder = Path(output_dir, args.env, args.agent, config.name)
    deleted_old = False
    if agent_folder.exists():
        if args.replace:
            shutil.rmtree(agent_folder)
            deleted_old = True
        else:
            raise FileExistsError(f"The experiment {agent_folder} already exists."
                                  f"Change output folder, experiment name or use -replace "
                                  f"to overwrite.")
    agent_folder.mkdir(parents=True)

    # Save experiments configurations and start experiment log
    prepare_file_logger(logger, logging.INFO, Path(agent_folder, "experiment.log"))
    logger.info(f"Running {args.agent} policy gradient on {args.env}")
    if deleted_old:
        logger.info(f"Deleted old experiment in {agent_folder}")
    config.log_configurations(logger)
    config.copy_config(agent_folder)

    # Handle default show_every value
    show_every = int(config.training_steps * 0.1) if config.show_every is None else config.show_every

    # Create and train the agent
    agent = PG_METHODS[args.agent]["agent"](env=ENVIRONMENTS[args.env](),
                                            agent_path=agent_folder,
                                            agent_config=config)
    start_time = time.time()
    test_reward = agent.train_policy(train_steps=config.training_steps,
                                     experience_size=config.experience_size,
                                     show_every=show_every,
                                     save_policy_every=config.save_policy_every,
                                     minibatch_size=config.minibatch_size)
    train_time = time.time() - start_time

    experiment_info = {"mean_test_reward": float(test_reward),
                       "description": args.desc,
                       "git_hash": sha,
                       "train_time": train_time}
    with open(Path(agent_folder, "experiment_information.json"), "w") as outfile:
        json.dump(experiment_info, outfile, indent=4)


if __name__ == '__main__':
    main()
