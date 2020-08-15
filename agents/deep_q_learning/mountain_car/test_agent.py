import argparse
import sys
import os
import json
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.deep_q_learning.mountain_car.agent import MountainCarAgent


def main():
    parser = argparse.ArgumentParser(description="Test Deep Q Learning agent that plays the "
                                                 "MountainCar environment.")
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--render_games", action="store_true", default=False)
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

    results = []
    for i in range(args.episodes):
        starting_state, win = agent.play_game(plot_game=args.render_games)
        results.append(win)

        print(f"Episode = {i} - Starting State = {starting_state} - Victory = {win}")

    print(f"Agent performance = {sum(results) * 100 / len(results)} % of Wins")


if __name__ == '__main__':
    main()
