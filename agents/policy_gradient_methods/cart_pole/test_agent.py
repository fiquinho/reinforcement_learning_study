import argparse
import sys
import os
import json
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods.cart_pole.naive_pg import CartPoleNaivePolicyGradient


def main():
    parser = argparse.ArgumentParser(description="Test Deep Q Learning agent that plays the "
                                                 "MountainCar environment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('REQUIRED named arguments')
    required_named.add_argument("--experiment_dir", type=str, required=True,
                                help="The path to a trained agent directory.")
    parser.add_argument("--episodes", type=int, default=200,
                        help="The number of episodes to play during testing.")
    parser.add_argument("--render_games", action="store_true", default=False,
                        help="Activate to render the agent playing each episode.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    config_file = Path(experiment_dir, "configurations.json")
    with open(config_file, "r", encoding="utf8") as cfile:
        config = json.load(cfile)

    agent = CartPoleNaivePolicyGradient(layer_size=config["hidden_layer_size"],
                                        learning_rate=config["learning_rate"],
                                        hidden_layers_count=config["hidden_layers_count"],
                                        activation=config["activation"])

    agent.load_model(Path(experiment_dir, "model"))

    results = []
    for i in range(args.episodes):
        total_reward, win = agent.play_game(plot_game=args.render_games, delay=None)
        results.append(win)

        print(f"Episode = {i} - Total Reward = {total_reward} - Victory = {win}")

    print(f"Agent performance = {sum(results) * 100 / len(results)} % of Wins")


if __name__ == '__main__':
    main()
