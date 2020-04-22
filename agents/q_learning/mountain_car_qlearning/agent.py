import argparse
from typing import Tuple

import gym
import numpy as np
import matplotlib.pyplot as plt


EPISODES = 5000
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1


class MountainCarAgent(object):

    def __init__(self, discrete_positions: Tuple[int, int]):

        self.env = gym.make("MountainCar-v0")
        self.env.reset()

        self.discrete_window_size = None
        self.q_table = self.build_qtable(discrete_positions[0], discrete_positions[1])

    def build_qtable(self, positions_discrete_size, velocities_discrete_size):

        discrete_total_size = (self.env.observation_space.high - self.env.observation_space.low)
        self.discrete_window_size = discrete_total_size / [positions_discrete_size, velocities_discrete_size]

        return np.random.uniform(low=-2, high=0, size=([positions_discrete_size, velocities_discrete_size] +
                                                       [self.env.action_space.n]))

    def get_discrete_state(self, state: np.ndarray):
        discrete_state = (state - self.env.observation_space.low) / self.discrete_window_size
        return tuple(discrete_state.astype(np.int))

    def produce_action(self, state: np.ndarray):
        action = np.argmax(self.q_table[self.get_discrete_state(state)])
        return action

    def flat_q_table(self):
        return np.reshape(self.q_table, -1)

    def train_agent(self, episodes: int=25000, epsilon: float=1, plot_game: bool=False,
                    show_every: int=None, learning_rate: float=0.1, discount: float=0.95,
                    cycles: int=1):

        episodes_wins = []
        episodes_rewards = []
        end_epsilon_decay = episodes // 2
        epsilon_decay_value = epsilon / (end_epsilon_decay - 1)
        if show_every is None:
            show_every = episodes

        print("Starting training...")
        for cycle in range(cycles):
            current_epsilon = epsilon
            for episode in range(episodes):
                self.env.reset()
                done = False
                show = False
                episode_reward = 0

                if episode > 0:
                    if not episode % show_every:
                        print(f"Showing episode N° {episode} of cycle {cycle}")
                        print(f"Epsilon is {current_epsilon}")
                        print(f"Last {show_every} episodes reward mean: {np.mean(episodes_rewards[-show_every:])}")
                        batch_wins = np.sum(episodes_wins[-show_every:])
                        print(f"Wins in last {show_every} episodes = {batch_wins}")
                        show = True
                    else:
                        show = False

                while not done:

                    if show and plot_game:
                        self.env.render()

                    state = self.env.state
                    if np.random.random() > current_epsilon:
                        action = self.produce_action(state)
                    else:
                        action = np.random.randint(0, self.env.action_space.n)
                    new_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward

                    if done:
                        if new_state[0] >= self.env.goal_position:
                            self.q_table[self.get_discrete_state(state) + (action,)] = 0
                            episodes_wins.append(True)
                        else:
                            episodes_wins.append(False)
                    else:
                        max_future_q = np.max(self.q_table[self.get_discrete_state(new_state)])
                        current_q = self.q_table[self.get_discrete_state(state) + (action,)]

                        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)

                        self.q_table[self.get_discrete_state(state) + (action,)] = new_q

                if end_epsilon_decay >= episode >= 0:
                    current_epsilon -= epsilon_decay_value

                episodes_rewards.append(episode_reward)

        moving_avg = np.convolve(episodes_rewards, np.ones((show_every,)) / show_every, mode='valid')

        plt.figure(figsize=(10, 5))
        # Moving average plot
        plt.subplot(121)

        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel(f"Reward {show_every}ma")
        plt.xlabel("episode #")
        plt.title("Reward moving average")
        text_x = int(len(moving_avg) * 0.6)
        text_y = (max(moving_avg) + min(moving_avg)) // 2
        plt.text(text_x, text_y,
                 f"Learning rate: {learning_rate}\n",
                 fontweight='bold',
                 bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

        plt.subplot(122)
        flat_qtable = self.flat_q_table()
        plt.hist(flat_qtable, bins=20)
        plt.title("Q table values - Histogram")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Q Learning agent that plays the MoveToGoal hard environment.")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--show_every", type=int, default=None, help="Defaults to 10% of the episodes.")
    parser.add_argument("--epsilon", type=int, default=EPSILON)
    parser.add_argument("--discount", type=int, default=DISCOUNT)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--plot_game", action="store_true", default=False)
    args = parser.parse_args()

    test_agent = MountainCarAgent((20, 20))
    test_agent.train_agent(episodes=args.episodes, epsilon=args.epsilon, plot_game=args.plot_game,
                           show_every=args.show_every, learning_rate=args.learning_rate,
                           discount=args.discount)


if __name__ == '__main__':
    main()
