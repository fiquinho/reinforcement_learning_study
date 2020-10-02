import argparse
import logging
import os
import sys
import shutil
import time
from pathlib import Path

import tensorflow as tf
import numpy as np
import gym

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods import NaivePolicyGradientAgent
from code_utils import prepare_file_logger, prepare_stream_logger, BaseConfig

logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)
logging.getLogger("tensorflow").setLevel(logging.ERROR)


EXPERIMENTS_DIR = Path(Path.home(), "rl_experiments", "policy_gradient")
SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
CONFIG_FILE = Path(SCRIPT_DIR.parent, "configurations", "default.json")


class AgentConfig(BaseConfig):

    def __init__(self, name: str, config_file: Path):
        BaseConfig.__init__(self, config_file)

        self.name = name
        self.training_steps = self.config_dict["training_steps"]
        self.show_every = self.config_dict["show_every"]
        self.learning_rate = self.config_dict["learning_rate"]
        self.batch_size = self.config_dict["batch_size"]
        self.hidden_layer_size = self.config_dict["hidden_layer_size"]
        self.hidden_layers_count = self.config_dict["hidden_layers_count"]
        self.activation = self.config_dict["activation"]
        self.save_policy_every = self.config_dict["save_policy_every"]


class CartPoleNaivePolicyGradient(NaivePolicyGradientAgent):

    def __init__(self, layer_size: int, learning_rate: float,
                 hidden_layers_count: int, activation: str):
        self.env = gym.make("CartPole-v0")
        self.action_space = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]
        self.actions = ["left", "right"]

        NaivePolicyGradientAgent.__init__(self,
                                          input_size=self.state_space,
                                          layer_size=layer_size,
                                          output_size=self.action_space,
                                          learning_rate=learning_rate,
                                          hidden_layers_count=hidden_layers_count,
                                          activation=activation)

    def reset_environment(self):
        self.env.reset()

    def get_environment_state(self) -> np.array:
        return self.env.state

    def environment_step(self, action: int):
        next_state, reward, done, _ = self.env.step(int(action))
        return next_state, reward, done

    def get_possible_states(self) -> list:
        return None

    def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
        return None, None

    def render_environment(self):
        self.env.render()

    @staticmethod
    def win_condition(episode_reward: int):
        return episode_reward >= 200

    def play_game(self, plot_game: bool=False, delay: float=None):
        self.reset_environment()
        done = False
        episode_reward = 0
        while not done:
            if plot_game:
                self.render_environment()
                if delay is not None:
                    time.sleep(delay)

            state = self.get_environment_state()
            tf_current_state = tf.constant(np.array([state]), dtype=tf.float32)
            action = self.policy.produce_actions(tf_current_state)[0][0]

            new_state, reward, done = self.environment_step(action)
            episode_reward += 1

        win = self.win_condition(episode_reward)

        self.reset_environment()

        return episode_reward, win


def main():
    parser = argparse.ArgumentParser(description="Naive Policy Gradient agent that plays the "
                                                 "MountainC simple environment.",
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
    agent_folder = Path(output_dir, "cart_pole", config.name)
    agent_folder.mkdir(exist_ok=True, parents=True)
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
    agent = CartPoleNaivePolicyGradient(layer_size=config.hidden_layer_size,
                                        learning_rate=config.learning_rate,
                                        hidden_layers_count=config.hidden_layers_count,
                                        activation=config.activation)
    agent.train_policy(train_steps=config.training_steps, batch_size=config.batch_size, show_every=show_every,
                       save_model=agent_folder, save_policy_every=config.save_policy_every)


if __name__ == '__main__':
    main()
