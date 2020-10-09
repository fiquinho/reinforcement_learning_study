import argparse
import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent))

from agents.policy_gradient_methods import ENVIRONMENTS, PG_METHODS


def main():
    parser = argparse.ArgumentParser(description="Test a trained agent on it's environment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('REQUIRED named arguments')
    required_named.add_argument("--experiment_dir", type=str, required=True,
                                help="The path to a trained agent directory.")
    required_named.add_argument("--env", type=str, choices=ENVIRONMENTS.keys(), required=True,
                                help="The environment to solve.")
    required_named.add_argument("--agent", type=str, choices=PG_METHODS.keys(), required=True,
                                help="The policy gradient method to use.")
    parser.add_argument("--episodes", type=int, default=200,
                        help="The number of episodes to play during testing.")
    parser.add_argument("--render_games", action="store_true", default=False,
                        help="Activate to render the agent playing each episode.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    config_file = Path(experiment_dir, "configurations.json")
    config = PG_METHODS[args.agent]["config"](experiment_dir.stem, config_file)

    agent = PG_METHODS[args.agent]["agent"](env=ENVIRONMENTS[args.env](), agent_config=config)

    agent.load_model(Path(experiment_dir, "model"))

    results = []
    for i in range(args.episodes):
        total_reward, win = agent.play_game(plot_game=args.render_games, delay=None)
        results.append(win)

        print(f"Episode = {i} - Total Reward = {total_reward} - Victory = {win}")

    print(f"Agent performance = {sum(results) * 100 / len(results)} % of Wins")

    agent.env.env.close()


if __name__ == '__main__':
    main()
