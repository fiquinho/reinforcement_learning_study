import argparse
import sys
import os
import json
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.deep_q_learning.mountain_car.agent import MountainCarAgent


def main():
    parser = argparse.ArgumentParser(description="Analyse Deep Q Learning agent that plays the "
                                                 "MountainCar environment.")
    parser.add_argument("--experiment_dir", type=str, required=True)
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    config_file = Path(experiment_dir, "configurations.json")
    with open(config_file, "r", encoding="utf8") as cfile:
        config = json.load(cfile)

    agent = MountainCarAgent(min_replay_memory_size=config["min_replay_memory_size"],
                             update_target_every=config["update_target_every"],
                             replay_memory_size=config["replay_memory_size"],
                             batch_size=config["batch_size"],
                             layer_size=config["hidden_layer_size"],
                             learning_rate=config["learning_rate"])

    agent.load_model(Path(experiment_dir, "model"))

    agent.q_values_plot(show_plot=True)


if __name__ == '__main__':
    main()
