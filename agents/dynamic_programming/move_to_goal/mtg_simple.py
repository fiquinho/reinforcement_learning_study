import logging
from copy import deepcopy

import numpy as np
from matplotlib import style

from environments.move_to_goal import MoveToGoal, MoveToGoalSimple
from code_utils.logger_utils import prepare_stream_logger


logger = logging.getLogger()
prepare_stream_logger(logger, logging.INFO)


style.use("ggplot")


class MoveToGoalSimplePolicyAgent(object):

    def __init__(self, game: MoveToGoal, policy: str="deterministic"):

        self.game = game
        self.board_size = self.game.get_board_size()

        if policy == "deterministic":
            self.policy = self.random_deterministic_policy()
        elif policy == "stochastic":
            self.policy = self.random_stochastic_policy()

    def random_stochastic_policy(self):
        raise NotImplementedError()

    def random_deterministic_policy(self):
        policy = []
        for i in range(self.board_size[0]):
            state_policy = []
            for j in range(self.board_size[1]):
                random_action = np.random.randint(self.game.action_space)
                state_policy.append(random_action)
            policy.append(state_policy)

        return np.array(policy)

    def produce_action(self, state: tuple) -> int:
        action = self.policy[state]
        return action

    def policy_evaluation(self, discount_factor=1.0, max_iterations=1000, theta=1e-6) -> np.array:

        # Number of evaluation iterations
        evaluation_iterations = 0

        # Initialize a value function for each state as zero
        values = np.zeros(self.board_size)

        # Repeat until change in value is below the threshold
        for iteration in range(int(max_iterations)):
            # Initialize a change of value function as zero
            delta = 0

            # Initial new value of current state
            initial_values = deepcopy(values)

            # Iterate though each state
            for i in range(self.board_size[0]):
                for j in range(self.board_size[1]):

                    if self.game.goal.position == (i, j):
                        continue

                    # Look the policy to find the action we would take in this state
                    action = self.policy[(i, j)]
                    # Check how good next state will be
                    new_state, reward, done = self.game.specific_step_results(state=(i, j), action=action)
                    v = reward + discount_factor * initial_values[new_state]
                    delta = max(delta, np.abs(initial_values[(i, j)] - v))
                    # Update value function
                    values[(i, j)] = v
            evaluation_iterations += 1

            if delta < theta:
                break

        logger.info(f'Policy evaluated in {evaluation_iterations} iterations.')

        return values

    def update_policy(self, policy_values: np.array, discount_factor=1.0):
        updated_policy = False
        for i in range(self.board_size[0]):
            for j in range(self.board_size[1]):
                # Choose the best action in a current state under current policy
                current_action = self.policy[(i, j)]

                action_values = np.zeros(self.game.action_space)
                for action in range(self.game.action_space):
                    new_state, reward, done = self.game.specific_step_results(state=(i, j), action=action)
                    action_values[action] += reward + discount_factor * policy_values[new_state]
                # Select a better action
                best_action = np.random.choice(np.flatnonzero(action_values == max(action_values)))
                if best_action != current_action:
                    updated_policy = True

                self.policy[(i, j)] = best_action
                
        return updated_policy

    def play_game(self):
        self.game.prepare_game()

        episode_reward = 0
        done = False
        win = False
        while not done:
            board_state = self.game.get_state()
            action = self.produce_action(board_state)
            new_board_state, reward, done = self.game.step(player_action=action)

            if done and reward == self.game.goal_reward:
                win = True

            episode_reward += reward

        return win, episode_reward

    def agent_evaluation(self, episodes: int=1):
        wins = 0
        rewards = []
        for i in range(episodes):
            win, episode_reward = self.play_game()
            rewards.append(episode_reward)
            if win:
                wins += 1
        avg_reward = sum(rewards) / len(rewards)
        logger.info(f"Agent won {wins}/{episodes} episodes.")
        logger.info(f"Average episode reward was {avg_reward}")

        return wins, avg_reward

    def train_agent(self, episodes: int=10_000, epsilon: float=1, plot_game: bool=False,
                    show_every: int=None, learning_rate: float=0.1, discount: float=0.95,
                    cycles: int=4):
        raise NotImplementedError()

    def training_step(self, epsilon, learning_rate, discount):
        raise NotImplementedError()


def main():
    game = MoveToGoalSimple(board_x=4, board_y=6, goal_reward=2, move_reward=-1,
                            game_end=100, goal_initial_pos=(3, 5), player_initial_pos=(0, 0))
    agent = MoveToGoalSimplePolicyAgent(game=game)
    test_episodes = 100

    for i in range(20):
        policy_values = agent.policy_evaluation(discount_factor=0.9, max_iterations=1000)
        updated_policy = agent.update_policy(policy_values, discount_factor=0.9)
        wins, avg_reward = agent.agent_evaluation(test_episodes)
        if wins == test_episodes:
            print(updated_policy)
            break


if __name__ == '__main__':
    main()
