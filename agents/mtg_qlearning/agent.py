import time
import argparse
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

from environments.move_to_goal.move_to_goal import MoveToGoal


DEFAULT_BOARD_SIZE = (10, 15)
GOAL_REWARD = 250
MOVE_REWARD = -1
EPISODES = 5000
GAME_END = 200
SHOW_EVERY = int(EPISODES * 0.1)
LEARNING_RATE = 0.1
DISCOUNT = 0.95

style.use("ggplot")


class Agent(object):

    def __init__(self, game: MoveToGoal):

        self.game = game
        self.board_size = self.game.get_board_size()
        self.q_table = np.random.uniform(low=-2, high=0, size=(self.board_size[0],
                                                               self.board_size[1],
                                                               len(self.game.actions)))

    def produce_action(self, state: Tuple[Tuple[int, int], Tuple[int, int]]):

        action = np.argmax(self.q_table[state[0]])

        return action

    def reshape_q_table(self):
        return self.q_table.reshape([self.board_size[0] * self.board_size[1], len(self.game.actions)])


def main():
    parser = argparse.ArgumentParser(description="Q Learning agent that plays the MoveToGoal environment.")
    parser.add_argument("--board_size", type=int, nargs="*", default=DEFAULT_BOARD_SIZE)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--game_end", type=int, default=GAME_END)
    parser.add_argument("--show_every", type=int, default=SHOW_EVERY)
    args = parser.parse_args()

    board_size = args.board_size
    if len(board_size) != 2:
        raise ValueError(f"The board size must be 2 values. "
                         f"Found ( {len(board_size)} ) values = {board_size}")

    test_game = MoveToGoal(board_size[0], board_size[1], GOAL_REWARD, MOVE_REWARD)
    test_agent = Agent(game=test_game)

    epsilon = 1
    end_epsilon_decay = args.episodes // 2
    epsilon_decay = 0.9998

    print("Training the agent")

    episodes_wins = []
    episode_rewards = []

    for episode in range(args.episodes):

        test_game.prepare_game(player_pos=(0, 0), goal_pos=(board_size[0] - 1, board_size[1] - 1))

        steps_played = 0
        done = False

        if not episode % args.show_every:
            print(f"Showing episode N° {episode}")
            print(f"on #{episode}, epsilon is {epsilon}")
            print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
            batch_wins = np.sum(episodes_wins[-SHOW_EVERY:])
            print(f"Wins in last {SHOW_EVERY} episodes = {batch_wins}")
            show = True
        else:
            show = False

        episode_reward = 0
        while not done:

            if show:
                test_game.display_game()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(.01)

            board_state = test_game.get_state()

            if np.random.random() > epsilon:
                action = test_agent.produce_action(board_state)
            else:
                action = np.random.randint(0, len(test_game.actions))
            new_board_state, reward, done = test_game.step(action)
            steps_played += 1

            if done:
                test_agent.q_table[board_state[0] + (action,)] = reward
                episodes_wins.append(True)
            else:
                max_future_q = np.max(test_agent.q_table[new_board_state[0]])
                current_q = test_agent.q_table[board_state[0] + (action,)]

                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                test_agent.q_table[board_state[0] + (action,)] = new_q

            episode_reward += reward

            if steps_played >= args.game_end:
                episodes_wins.append(False)
                done = True

        if end_epsilon_decay >= episode >= 0:
            epsilon *= epsilon_decay

        episode_rewards.append(episode_reward)

    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"Reward {SHOW_EVERY}ma")
    plt.xlabel("episode #")
    text_x = int(len(moving_avg) * 0.6)
    text_y = 0
    plt.text(text_x, text_y,
             f"Goal reward: {GOAL_REWARD}\n"
             f"Move reward: {MOVE_REWARD}\n",
             fontweight='bold',
             bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    plt.show()


if __name__ == '__main__':
    main()
