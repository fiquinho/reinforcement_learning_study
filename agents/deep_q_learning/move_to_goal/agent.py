import time
import logging
import random
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from environments.move_to_goal.move_to_goal import MoveToGoal


logger = logging.getLogger()

style.use("ggplot")


class MoveToGoalDQNAgent(object):

    def __init__(self, game: MoveToGoal, learning_rate: float=0.1, replay_memory_size: int=50_000,
                 min_replay_memory_size: int=1000, batch_size: int=64, update_target_every: int=5):

        self.game = game
        self.learning_rate = learning_rate
        self.replay_memory_size = replay_memory_size
        self.min_replay_memory_size = min_replay_memory_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        self.board_size = self.game.get_board_size()
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model(learning_rate)
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        self.target_update_counter = 0

    def create_model(self, learning_rate: float=0.1):
        model = Sequential()
        input_shape = (self.game.state_space,)
        model.add(Dense(64, activation="relu", input_shape=input_shape))
        model.add(Dense(64, activation="relu"))

        model.add(Dense(self.game.action_space, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
        return model

    def produce_action(self, state: tuple):
        q_values = self.get_q_values(state)
        action = np.argmax(q_values)
        return action

    def get_q_values(self, state: tuple):
        return self.model.predict(np.array([state]))

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train_agent(self, episodes: int=10_000, epsilon: float=1, plot_game: bool=False,
                    show_every: int=None, discount: float=0.95, cycles: int=4, save_model: Path=None):

        episodes_counter = 0
        end_epsilon_decay = episodes // 2
        epsilon_decay_value = epsilon / (end_epsilon_decay - 1)
        episodes_wins = []
        episode_rewards = []

        logger.info("Starting training...")
        start_time = time.time()
        start_epsilon = epsilon
        for cycle in range(cycles):
            epsilon = start_epsilon

            for episode in tqdm(range(1, episodes + 1), ascii=True, unit='episodes'):

                self.game.prepare_game()
                if show_every is not None and not episodes_counter % show_every and episodes_counter > 0:
                    logger.info("#########################")
                    logger.info(f"Showing episode N° {episode - 1}/{episodes} of cycle {cycle}/{cycles}")
                    logger.info(f"Epsilon is {epsilon}")
                    logger.info(f"Last {show_every} episodes reward mean: {np.mean(episode_rewards[-show_every:])}")
                    batch_wins = np.sum(episodes_wins[-show_every:])
                    logger.info(f"Wins in last {show_every} episodes = {batch_wins}")
                    logger.info(f"Batch time = {time.time() - start_time} sec")
                    show = True
                    start_time = time.time()
                else:
                    show = False

                episode_reward = 0
                done = False
                while not done:

                    if show and plot_game:
                        self.game.display_game(f"Episode {episode}")
                        time.sleep(.01)

                    board_state = self.game.get_state()

                    # Select action with exploration/exploitation
                    if np.random.random() > epsilon:
                        action = self.produce_action(board_state)
                    else:
                        action = np.random.randint(0, len(self.game.actions))

                    new_state, reward, done = self.game.step(player_action=action)
                    episode_reward += reward

                    # Every step we update replay memory and train main network
                    self.update_replay_memory((board_state, action, reward, new_state, done))
                    self.training_step(discount)

                    if done:
                        episodes_wins.append(reward == self.game.goal_reward)
                        self.target_update_counter += 1

                    # If counter reaches set value, update target network with weights of main network
                    if self.target_update_counter > self.update_target_every:
                        self.target_model.set_weights(self.model.get_weights())
                        self.target_update_counter = 0

                    episode_reward += reward

                if end_epsilon_decay >= episode >= 0:
                    epsilon -= epsilon_decay_value

                episode_rewards.append(episode_reward)
                episodes_counter += 1

        moving_avg = np.convolve(episode_rewards, np.ones((show_every,)) / show_every, mode='valid')

        if save_model is not None:
            logger.info(f"Saving trained model to {save_model}")
            self.plot_training_info(moving_avg, save_model)
            self.model.save(Path(save_model, "model"))

    def training_step(self, discount):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.min_replay_memory_size:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, self.batch_size)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        # Fill the NN inputs
        states_input = []
        target_q_values = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + discount * max_future_q
            else:
                new_q = reward

            # Get target Q value for given state
            # We only change the Q value of the taken action
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            states_input.append(current_state)
            target_q_values.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(states_input), np.array(target_q_values),
                       batch_size=self.batch_size, verbose=0, shuffle=False)

    @staticmethod
    def plot_training_info(moving_avg: np.array, agent_folder: Path=None):
        plt.figure(figsize=(5, 5))

        # Moving average plot
        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel(f"Reward")
        plt.xlabel("Episode #")
        plt.title("Reward moving average")

        if agent_folder is not None:
            plt.savefig(Path(agent_folder, "reward_moving_average.png"))

    def save_agent(self, output_dir: Path):
        raise NotImplementedError()
