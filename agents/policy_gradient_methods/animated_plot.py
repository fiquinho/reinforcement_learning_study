import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import numpy as np


SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods import ENVIRONMENTS, PG_METHODS

tf.config.run_functions_eagerly(True)

env_name = "HeuristicMountainCar-v0"
agent_name = "naive"
experiment_name = "00.02"

experiment_dir = Path("C://", "Users", "Fico", "rl_experiments",
                      "policy_gradient", env_name, agent_name, experiment_name)
config_file = Path(experiment_dir, "configurations.json")
config = PG_METHODS[agent_name]["config"](experiment_name, config_file)
agent = PG_METHODS[agent_name]["agent"](env=ENVIRONMENTS[env_name](),
                                        agent_config=config)

agent.load_model(Path(experiment_dir, "model"))


agent.env.reset_environment()
start_state = agent.env.get_environment_state()
tf_start_state = tf.constant(np.array([start_state]), dtype=tf.float32)
start_logits = agent.policy(tf_start_state)
start_probabilities = agent.policy.get_probabilities(start_logits)[0]

fig = plt.figure(figsize=[10, 10])
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_ylim(0, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_xlim(0, 210)
ax3 = fig.add_subplot(2, 2, 4)
ax3.set_xlim(0, 210)
ax3.set_ylim(-1, 1)


actions = agent.env.actions
actions_idx = np.arange(len(actions))
bars = ax1.bar(actions_idx, start_probabilities, tick_label=actions)

actions_lines = [ax2.plot([], [], "o", markersize=3, label=action) for action in actions]
ax2.legend()
state_lines = [ax3.plot([], [], label=state_name) for state_name in agent.env.state_names]
ax3.legend()

MAX_FRAMES = 201

left_probabilities = []
right_probabilities = []

states_history = []
probabilities_history = []


def animate(frame):

    if frame == MAX_FRAMES - 1:
        print(f"Reached {frame + 1} frames, closing fig!")
        plt.close(fig)
    else:
        agent.env.render_environment()

        state = agent.env.get_environment_state()
        tf_current_state = tf.constant(np.array([state]), dtype=tf.float32)
        action = agent.policy.produce_actions(tf_current_state)[0][0]
        next_state, reward, done = agent.env.environment_step(action)
        logits = agent.policy(tf_current_state)
        probabilities = agent.policy.get_probabilities(logits)[0]

        states_history.append(state)
        probabilities_history.append(probabilities)
        # left_probabilities.append(probabilities[0])
        # right_probabilities.append(probabilities[1])

        [bar.set_height(probabilities[i]) for i, bar in enumerate(bars)]

        for i, line in enumerate(actions_lines):
            line[0].set_data(np.array(range(len(probabilities_history))), [p[i] for p in probabilities_history])

        for i, line in enumerate(state_lines):
            line[0].set_data(np.array(range(len(states_history))), [s[i] for s in states_history])

        if done:
            print("Finished episode!")
            plt.close(fig)

    return bars, actions_lines, state_lines


# Set up formatting for the movie files
# writer = animation.HTMLWriter()
# writer.setup(fig, "animation.html", fig.dpi)

ani = animation.FuncAnimation(
    fig, animate, frames=MAX_FRAMES, interval=100, repeat=False)


# ani.save('game.gif', writer='imagemagick')


plt.show()
plt.close('all')

fig.savefig("game.png")
agent.env.env.close()
