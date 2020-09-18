import argparse
import logging
import os
import sys
import shutil
from pathlib import Path

import tensorflow as tf

from environments.move_to_goal import MoveToGoalSimple
from agents.policy_gradient_methods import BaseNaivePolicyGradientAgent
from code_utils import prepare_file_logger, prepare_stream_logger, BaseConfig

logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


EXPERIMENTS_DIR = Path(Path.home(), "rl_experiments", "policy_gradient")
SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
CONFIG_FILE = Path(SCRIPT_DIR.parent, "configurations", "simple_default.json")


class AgentConfig(BaseConfig):

    def __init__(self, name: str, config_file: Path):
        BaseConfig.__init__(self, config_file)

        self.name = name

        self.board_size = self.config_dict["board_size"]
        self.player_pos = tuple(self.config_dict["player_pos"])
        self.goal_pos = tuple(self.config_dict["goal_pos"])
        self.goal_reward = self.config_dict["goal_reward"]
        self.move_reward = self.config_dict["move_reward"]
        self.game_end = self.config_dict["game_end"]
        self.training_steps = self.config_dict["training_steps"]
        self.show_every = self.config_dict["show_every"]
        self.learning_rate = self.config_dict["learning_rate"]
        self.batch_size = self.config_dict["batch_size"]
        self.hidden_layer_size = self.config_dict["hidden_layer_size"]
        self.hidden_layers_count = self.config_dict["hidden_layers_count"]
        self.activation = self.config_dict["activation"]


class NaivePolicyGradientAgent(BaseNaivePolicyGradientAgent):

    def __init__(self, game: MoveToGoalSimple, layer_size: int, learning_rate: float,
                 hidden_layers_count: int, activation: str):
        self.game = game

        BaseNaivePolicyGradientAgent.__init__(self,
                                              layer_size=layer_size,
                                              output_size=self.game.action_space,
                                              learning_rate=learning_rate,
                                              hidden_layers_count=hidden_layers_count,
                                              activation=activation)

    def reset_environment(self):
        self.game.prepare_game()

    def get_environment_state(self):
        return self.game.get_state()

    def environment_step(self, action: int):
        next_state, reward, done = self.game.step(action)
        return next_state, reward, done


def main():
    parser = argparse.ArgumentParser(description="Naive Policy Gradient agent that plays the "
                                                 "MoveToGoal simple environment.",
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

    env = MoveToGoalSimple(board_x=config.board_size[0],
                           board_y=config.board_size[1],
                           goal_reward=config.goal_reward,
                           move_reward=config.move_reward,
                           player_initial_pos=config.player_pos,
                           goal_initial_pos=config.goal_pos,
                           game_end=config.game_end)

    # Create experiment folder and handle old results
    output_dir = Path(args.output_dir)
    game_experiments_dir = Path(output_dir, "mtg_simple", env.game_configs)
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

    show_every = int(config.training_steps * 0.1) if config.show_every is None else config.show_every

    # Create and train the agent
    agent = NaivePolicyGradientAgent(game=env,
                                     layer_size=config.hidden_layer_size,
                                     learning_rate=config.learning_rate,
                                     hidden_layers_count=config.hidden_layers_count,
                                     activation=config.activation)
    agent.train_policy(train_steps=config.training_steps, batch_size=config.batch_size, show_every=show_every,
                       save_model=agent_folder)


if __name__ == '__main__':
    main()
