import argparse
import sys
import os
import json
from pathlib import Path

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.deep_q_learning.mountain_car.agent import MountainCarAgent


def moves_analysis(agent: MountainCarAgent):
    x_moves = []
    y_moves = []
    x_min = 10
    x_max = 0
    y_min = 10
    y_max = 0

    wins = []
    new_state = ()
    for i in range(100):
        agent.env.reset()
        done = False
        while not done:

            state = agent.env.state
            action = agent.produce_action(state)

            new_state, reward, done, _ = agent.env.step(action)
            x_move = abs(new_state[0] - state[0])
            y_move = abs(new_state[1] - state[1])
            x_moves.append(x_move)
            y_moves.append(y_move)
            x_min = min(x_min, x_move)
            x_max = max(x_max, x_move)
            y_min = min(y_min, y_move)
            y_max = max(y_max, y_move)

        wins.append(bool(new_state[0] >= agent.env.goal_position and new_state[1] >= agent.env.goal_velocity))

    print(f"x_min {x_min}")
    print(f"x_max {x_max}")
    print(f"y_min {y_min}")
    print(f"y_max {y_max}")
    print(f"Wins = {sum(wins)}")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(x_moves, 100)
    ax2.hist(y_moves, 100)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyse Deep Q Learning agent that plays the "
                                                 "MountainCar environment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_named = parser.add_argument_group('REQUIRED named arguments')
    required_named.add_argument("--experiment_dir", type=str, required=True)
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
                             learning_rate=config["learning_rate"],
                             double_q_learning=config["double_q_learning"])

    agent.load_model(Path(experiment_dir, "model"))

    agent.q_values_plot(show_plot=True)


if __name__ == '__main__':
    main()
