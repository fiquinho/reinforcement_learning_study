import os
import sys
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))
CONFIGS_DIR = Path(SCRIPT_DIR.parent.parent, "configurations")

from environments.simple_continuous.simple_continuous import SimpleContinuous
from agents.policy_gradient_methods.envs import Environment, Episode, EpisodesBatch


class SimpleContinuousEnvironment(Environment):

    def __init__(self):
        env = SimpleContinuous(target=4., max_reward=1., target_range=0.001)
        action_space = 1
        state_space = 1
        actions = ["action"]
        state_names = ["unique_state"]

        Environment.__init__(self, env, action_space, state_space,
                             actions, state_names, action_space="continuous")

    def reset_environment(self):
        pass

    def get_environment_state(self) -> np.array:
        state = self.env.get_state()
        return np.array([state])

    def environment_step(self, action: int) -> (np.array, float, bool):

        action = (action * 4.) + 4.

        return self.env.step(action)

    def get_possible_states(self) -> np.array:
        return self.get_environment_state()

    def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
        raise NotImplementedError

    def render_environment(self):
        pass

    @staticmethod
    def win_condition(episode: Episode):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


def main():
    """Train an agent with custom plots"""
    from agents.policy_gradient_methods.pg_methods import NaivePolicyGradientAgent, BaseAgentConfig

    agent_type = "naive"
    experiment_name = "01.01"
    desc = "test"
    env = SimpleContinuousEnvironment()
    agent_folder = Path("C://", "Users", "Fico", "rl_experiments",
                        "policy_gradient", "SimpleContinuous", agent_type, experiment_name)
    if agent_folder.exists():
        shutil.rmtree(agent_folder)
    agent_folder.mkdir(parents=True)

    config_file = Path(CONFIGS_DIR, "SimpleContinuous_naive_default.json")
    config = BaseAgentConfig(experiment_name, config_file, desc)
    agent = NaivePolicyGradientAgent(env, agent_folder, config)

    metrics_history = {"mean_mu": [],
                       "mean_sigma": [],
                       "mean_reward": [],
                       "mean_action": []}
    current_xlim = [10]
    fig, ax = plt.subplots()
    ax.set_xlim(0, current_xlim[0])
    ax.set_ylim(-1.1, 1.1)

    metrics_lines = [ax.plot([], [], label=metric_name) for metric_name in metrics_history.keys()]
    ax.legend()

    def animate(frame):
        if frame == config.training_steps - 1:
            print(f"Reached {frame + 1} frames, closing fig!")
            plt.close(fig)
        else:

            # Collect experience
            episodes_batch = EpisodesBatch(max_size=config.experience_size)
            state_list = []
            reward_list = []
            action_list = []
            while not episodes_batch.is_full():
                agent.env.reset_environment()
                current_state = agent.env.get_environment_state()
                tf_current_state = tf.constant(np.array([current_state]), dtype=tf.float32)
                action = agent.policy.produce_actions(tf_current_state)[0][0]
                next_state, reward, done = agent.env.environment_step(action.numpy())

                state_list.append(current_state)
                action_list.append(action.numpy())
                reward_list.append(reward)

                episode = Episode([current_state], [action], [reward], agent.env.action_space)
                episodes_batch.add_episode(episode=episode)

            tf_states = tf.constant(np.array(state_list), dtype=tf.float32)
            mu_list, sigma_list = agent.policy(tf_states)

            metrics_history["mean_mu"].append(np.mean(mu_list))
            metrics_history["mean_sigma"].append(np.mean(sigma_list))
            asd = np.mean(reward_list)
            metrics_history["mean_reward"].append(asd)
            metrics_history["mean_action"].append(np.mean(action_list))

            training_experience = agent.get_training_experience(episodes=episodes_batch)

            states_batch = tf.constant(training_experience.states, dtype=np.float32)
            weights_batch = tf.constant(training_experience.weights, dtype=np.float32)
            actions_batch = tf.constant(training_experience.actions, dtype=np.float32)

            data = tf.data.Dataset.from_tensor_slices((states_batch, actions_batch, weights_batch)).batch(config.experience_size)

            for data_batch in data:
                policy_outputs, loss, log_probabilities = agent.policy.train_step(
                    data_batch[0], data_batch[1], data_batch[2])

            if not len(metrics_history["mean_mu"]) % 10:
                for i, line in enumerate(metrics_lines):
                    line[0].set_data(list(range(len(metrics_history["mean_mu"]))),
                                     list(metrics_history.values())[i])

            if len(metrics_history["mean_mu"]) >= current_xlim[0]:
                current_xlim[0] = int(len(metrics_history["mean_mu"]) * 1.25)
                ax.set_xlim(0, current_xlim[0])

        return metrics_lines

    ani = animation.FuncAnimation(
        fig, animate, frames=config.training_steps, interval=100, repeat=False)
    plt.show()

    # fig.savefig("game.png")


if __name__ == '__main__':
    main()
