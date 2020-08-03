import logging
import sys
import os
import argparse
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from environments.move_to_goal.mtg_simple import MoveToGoalSimple
from agents.deep_q_learning.move_to_goal.agent import MoveToGoalDQNAgent
from code_utils.logger_utils import prepare_stream_logger, prepare_file_logger
from code_utils.config_utils import BaseConfig


EXPERIMENTS_DIR = Path(Path.home(), "rl_experiments", "deep_q_learning")
DEFAULT_CONFIG = Path(SCRIPT_DIR.parent, "configurations", "simple_test.json")


logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)


class AgentConfig(BaseConfig):

    def __init__(self, config_file: Path):
        BaseConfig.__init__(self, config_file)

        self.board_size = tuple(self.config_dict["board_size"])
        self.goal_reward = self.config_dict["goal_reward"]
        self.move_reward = self.config_dict["move_reward"]
        self.episodes = self.config_dict["episodes"]
        self.cycles = self.config_dict["cycles"]
        self.game_end = self.config_dict["game_end"]
        self.epsilon = self.config_dict["epsilon"]
        self.learning_rate = self.config_dict["learning_rate"]
        self.discount = self.config_dict["discount"]
        self.player_initial_pos = tuple(self.config_dict["player_initial_pos"])
        self.goal_initial_pos = tuple(self.config_dict["goal_initial_pos"])
        self.show_every = self.config_dict["show_every"]
        self.replay_memory_size = self.config_dict["replay_memory_size"]
        self.min_replay_memory_size = self.config_dict["min_replay_memory_size"]
        self.batch_size = self.config_dict["batch_size"]
        self.update_target_every = self.config_dict["update_target_every"]
        self.hidden_layer_size = self.config_dict["hidden_layer_size"]


def main():
    parser = argparse.ArgumentParser(description="Deep Q Learning agent that plays the "
                                                 "MoveToGoal simple environment.")
    parser.add_argument("--config_file", type=str, default=DEFAULT_CONFIG,
                        help=f"Config file for the experiment. Defaults to {DEFAULT_CONFIG}")
    parser.add_argument("--output_dir", type=str, default=EXPERIMENTS_DIR,
                        help=f"Where to save the experiments files. Defaults to {EXPERIMENTS_DIR}")
    parser.add_argument("--replace", action="store_true", default=False,
                        help="Activate to overwrite old experiment with same configurations.")
    args = parser.parse_args()

    config_file = Path(args.config_file)
    output_dir = Path(args.output_dir)

    experiment_config = AgentConfig(config_file)

    test_game = MoveToGoalSimple(board_x=experiment_config.board_size[0],
                                 board_y=experiment_config.board_size[1],
                                 goal_reward=experiment_config.goal_reward,
                                 move_reward=experiment_config.move_reward,
                                 game_end=experiment_config.game_end,
                                 goal_initial_pos=experiment_config.goal_initial_pos,
                                 player_initial_pos=experiment_config.player_initial_pos)

    # Experiment folder
    game_experiments_dir = Path(output_dir, test_game.game_name)
    game_experiments_dir.mkdir(exist_ok=True, parents=True)
    agent_folder = Path(game_experiments_dir, f"ep{experiment_config.episodes}_e{experiment_config.epsilon}_"
                                              f"lr{experiment_config.learning_rate}_d{experiment_config.discount}_"
                                              f"c{experiment_config.cycles}")
    agent_folder.mkdir(exist_ok=args.replace)

    logs_file = Path(agent_folder, "experiment.log")
    if logs_file.exists() and args.replace:
        logs_file.unlink()
    prepare_file_logger(logger, logging.INFO, Path(agent_folder, "experiment.log"))
    experiment_config.log_configurations(logger)

    test_agent = MoveToGoalDQNAgent(game=test_game,
                                    learning_rate=experiment_config.learning_rate,
                                    replay_memory_size=experiment_config.replay_memory_size,
                                    min_replay_memory_size=experiment_config.min_replay_memory_size,
                                    batch_size=experiment_config.batch_size,
                                    update_target_every=experiment_config.update_target_every,
                                    hidden_layer_size=experiment_config.hidden_layer_size)
    test_agent.train_agent(episodes=experiment_config.episodes,
                           epsilon=experiment_config.epsilon,
                           plot_game=False,
                           show_every=experiment_config.show_every,
                           discount=experiment_config.discount,
                           cycles=experiment_config.cycles,
                           save_model=agent_folder)


if __name__ == '__main__':
    main()
