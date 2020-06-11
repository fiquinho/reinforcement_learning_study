import time
import argparse
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from environments.move_to_goal.mtg_enemy import MoveToGoalEnemy


BOARD_SIZE = (7, 10)
GOAL_REWARD = 20
MOVE_REWARD = -1
ENEMY_REWARD = -20
EPISODES = 40000
GAME_END = 200
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 1

style.use("ggplot")


class MoveToGoalEnemyAgent(object):

    def __init__(self, game: MoveToGoalEnemy):

        self.game = game
        self.board_size = self.game.get_board_size()
        self.q_table = self.generate_q_table()

    def generate_q_table(self):

        table_size = []
        for i in range(0, self.game.state_space, 2):
            table_size.append(self.board_size[0])
            table_size.append(self.board_size[1])
        table_size.append(len(self.game.actions))

        return np.random.uniform(low=-2, high=0, size=table_size)

    def produce_action(self, state: Tuple[int, int, int, int, int, int]):
        action = np.argmax(self.q_table[state])
        return action

    def flat_q_table(self):
        return np.reshape(self.q_table, -1)

    def train_agent(self, episodes: int=25000, epsilon: float=1, plot_game: bool=False,
                    show_every: int=None, learning_rate: float=0.1, discount: float=0.95):

        end_epsilon_decay = episodes // 2
        epsilon_decay = 0.9998

        episodes_wins = []
        episode_rewards = []

        print("Starting training...")
        for episode in range(episodes):

            self.game.prepare_game()
            if not episode % show_every and episode > 0:
                print(f"Showing episode N° {episode}")
                print(f"on #{episode}, epsilon is {epsilon}")
                print(f"{show_every} ep mean: {np.mean(episode_rewards[-show_every:])}")
                batch_wins = np.sum(episodes_wins[-show_every:])
                print(f"Wins in last {show_every} episodes = {batch_wins}")
                show = True
            else:
                show = False

            episode_reward = 0
            done = False
            while not done:

                if show and plot_game:
                    self.game.display_game()
                    time.sleep(.01)

                board_state = self.game.get_state()

                if np.random.random() > epsilon:
                    action = self.produce_action(board_state)
                else:
                    action = np.random.randint(0, len(self.game.actions))
                new_board_state, reward, done = self.game.step(action)

                if done:
                    self.q_table[board_state + (action,)] = reward
                    episodes_wins.append(reward == self.game.goal_reward)
                else:
                    max_future_q = np.max(self.q_table[new_board_state])
                    current_q = self.q_table[board_state + (action,)]

                    new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)

                    self.q_table[board_state + (action,)] = new_q

                episode_reward += reward

            if end_epsilon_decay >= episode >= 0:
                epsilon *= epsilon_decay

            episode_rewards.append(episode_reward)

        moving_avg = np.convolve(episode_rewards, np.ones((show_every,)) / show_every, mode='valid')

        plt.figure(figsize=(10, 5))
        # Moving average plot
        plt.subplot(121)

        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel(f"Reward {show_every}ma")
        plt.xlabel("episode #")
        plt.title("Reward moving average")
        text_x = int(len(moving_avg) * 0.4)
        text_y = (max(moving_avg) + min(moving_avg)) // 2
        plt.text(text_x, text_y,
                 f"Goal reward: {self.game.goal_reward}\n"
                 f"Move reward: {self.game.move_reward}\n"
                 f"Enemy reward: {self.game.enemy_reward}",
                 fontweight='bold',
                 bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

        plt.subplot(122)
        flat_qtable = self.flat_q_table()
        plt.hist(flat_qtable, bins=200)
        plt.title("Q table values - Histogram")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Q Learning agent that plays the MoveToGoal hard environment.")
    parser.add_argument("--player_pos", type=int, nargs="*", default=None)
    parser.add_argument("--goal_pos", type=int, nargs="*", default=None)
    parser.add_argument("--enemy_pos", type=int, nargs="*", default=None)
    parser.add_argument("--board_size", type=int, nargs="*", default=BOARD_SIZE)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--show_every", type=int, default=None, help="Defaults to 10% of the episodes.")
    parser.add_argument("--goal_reward", type=int, default=GOAL_REWARD)
    parser.add_argument("--move_reward", type=int, default=MOVE_REWARD)
    parser.add_argument("--enemy_reward", type=int, default=ENEMY_REWARD)
    parser.add_argument("--epsilon", type=int, default=EPSILON)
    parser.add_argument("--discount", type=int, default=DISCOUNT)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--game_end", type=int, default=GAME_END)
    parser.add_argument("--plot_game", action="store_true", default=False)
    args = parser.parse_args()

    board_size = args.board_size
    if len(board_size) != 2:
        raise ValueError(f"The board size must be 2 values. "
                         f"Found ( {len(board_size)} ) values = {board_size}")

    player_pos = tuple(args.player_pos) if args.player_pos is not None else None
    goal_pos = tuple(args.goal_pos) if args.goal_pos is not None else None
    enemy_pos = tuple(args.enemy_pos) if args.enemy_pos is not None else None

    if args.show_every is None:
        args.show_every = int(args.episodes * 0.1)

    test_game = MoveToGoalEnemy(board_x=board_size[0],
                                board_y=board_size[1],
                                goal_reward=args.goal_reward,
                                move_reward=args.move_reward,
                                game_end=args.game_end,
                                enemy_reward=args.enemy_reward,
                                enemy_movement="random",
                                player_initial_pos=player_pos,
                                goal_initial_pos=goal_pos,
                                enemy_initial_pos=enemy_pos)
    test_agent = MoveToGoalEnemyAgent(game=test_game)
    test_agent.train_agent(episodes=args.episodes, epsilon=args.epsilon, plot_game=args.plot_game,
                           show_every=args.show_every, learning_rate=args.learning_rate,
                           discount=args.discount)


if __name__ == '__main__':
    main()
