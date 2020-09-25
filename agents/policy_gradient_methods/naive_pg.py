import logging
import time
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


logger = logging.getLogger()


class NaivePolicyGradientModel(Model):

    def __init__(self, layer_size: int, output_size: int, learning_rate: float,
                 hidden_layers_count: int, activation: str="relu"):
        super(NaivePolicyGradientModel, self).__init__()
        self.layer_size = layer_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_layers_count = hidden_layers_count
        self.activation = activation

        self.input_layer = Dense(layer_size, activation=activation)
        self.hidden_layers = []
        for i in range(self.hidden_layers_count - 1):  # The input layer is counted in hidden_layers_count
            self.hidden_layers.append(Dense(layer_size, activation=activation))

        self.output_logits = Dense(output_size, activation=None)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def get_config(self):
        return {"layer_size": self.layer_size,
                "output_size": self.output_size,
                "learning_rate": self.learning_rate,
                "hidden_layers_count": self.hidden_layers_count,
                "activation": self.activation}

    @tf.function
    def call(self, inputs: np.array, training=None, mask=None):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        logits = self.output_logits(x)
        return logits

    @tf.function
    def train_step(self, sates: np.array, actions: np.array, rewards: np.array):
        with tf.GradientTape() as tape:
            logits = self(sates)
            action_masks = tf.one_hot(actions, self.output_size)
            log_probabilities = tf.reduce_sum(action_masks * self.get_log_probabilities(logits), axis=-1)
            loss = -tf.reduce_mean(rewards * log_probabilities)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def get_probabilities(self, logits):
        probabilities = tf.nn.softmax(logits)
        return probabilities

    @tf.function
    def get_log_probabilities(self, logits):
        log_probabilities = tf.nn.log_softmax(logits)
        return log_probabilities

    @tf.function
    def produce_actions(self, states: np.array):
        logits = self(states)
        actions = tf.random.categorical(logits, 1)
        return actions

    @tf.function
    def get_policy_values(self, states: np.array):
        logits = self(states)
        predictions = self.get_probabilities(logits)
        return predictions


class BaseNaivePolicyGradientAgent(object):

    def __init__(self, output_size: int, layer_size: int, learning_rate: float,
                 hidden_layers_count: int, activation: str):

        self.policy = NaivePolicyGradientModel(layer_size=layer_size,
                                               output_size=output_size,
                                               learning_rate=learning_rate,
                                               hidden_layers_count=hidden_layers_count,
                                               activation=activation)

    def reset_environment(self):
        raise NotImplementedError

    def get_environment_state(self) -> np.array:
        raise NotImplementedError

    def environment_step(self, action: int):
        """
        Make a move in the environment with given action.
        :param action: The action index
        :return: next_environment_state, reward (float), terminated_environment (bool)
        """
        raise NotImplementedError

    def get_possible_states(self) -> list:
        raise NotImplementedError

    def policy_values_plot(self, save_fig: Path=None, show_plot: bool=False):
        raise NotImplementedError

    def get_states_index(self) -> dict:
        state_space = self.get_possible_states()
        states_index = {}
        for i, state in enumerate(state_space):
            states_index[state] = i
        return states_index

    def collect_experience(self, size: int):

        states_batch = []
        rewards_batch = []
        actions_batch = []
        total_rewards = []
        episode_lengths = []

        while len(states_batch) < size:
            self.reset_environment()
            done = False
            episode_reward = 0
            episode_len = 0
            while not done:
                current_state = self.get_environment_state()
                action = self.policy.produce_actions(np.array([current_state]))[0][0]
                next_state, reward, done = self.environment_step(action)

                states_batch.append(current_state)
                actions_batch.append(action)
                episode_reward += reward
                episode_len += 1

            rewards_batch += [episode_reward] * episode_len
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_len)

        states_batch = np.array(states_batch)
        rewards_batch = np.array(rewards_batch, dtype=np.float32)
        actions_batch = np.array(actions_batch)

        return states_batch, rewards_batch, actions_batch, total_rewards, episode_lengths

    def train_policy(self, train_steps: int, batch_size: int, show_every: int=None,
                     save_model: Path=None, save_policy_every: int=None):

        if save_policy_every is not None and save_model is None:
            raise ValueError("If you want to save the policy values during training you must "
                             "specify an output folder.")
        else:
            policy_values_dir = Path(save_model, "policy_values")
            policy_values_dir.mkdir()

        train_steps_avg_rewards = []
        start_time = time.time()

        for i in range(train_steps):
            states_batch, rewards_batch, actions_batch, batch_total_rewards, batch_episode_lengths = \
                self.collect_experience(batch_size)
            mean_reward = np.mean(batch_total_rewards)

            if show_every is not None:
                if i > 0 and not i % show_every:
                    logger.info("====================================================")
                    logger.info(f"Training step N° {i}")
                    logger.info(f"Batch time = {time.time() - start_time} sec")
                    logger.info(f"Last {len(batch_total_rewards)} episodes reward mean: {mean_reward}")
                    start_time = time.time()

            self.policy.train_step(states_batch, actions_batch, rewards_batch)
            train_steps_avg_rewards.append(mean_reward)

            if save_policy_every is not None:
                if not i % save_policy_every:
                    possible_states, states_predictions = self.policy_values_plot(
                        Path(policy_values_dir, f"policy_values_{i}.png"))
                    with open(Path(policy_values_dir, f"policy_values_{i}.pickle"), "wb") as pfile:
                        pickle.dump(states_predictions, pfile, protocol=pickle.HIGHEST_PROTOCOL)

        moving_avg = np.convolve(train_steps_avg_rewards, np.ones((show_every,)) / show_every, mode='valid')

        if save_model is not None:
            self.save_agent(save_model)
            self.plot_training_info(moving_avg, save_model)

    def save_agent(self, output_dir: Path):
        logger.info(f"Saving trained policy to {output_dir}")
        start = time.time()
        self.policy.save(Path(output_dir, "model"))
        logger.info(f"Saving time {time.time() - start}")

    def load_model(self, model_dir: Path):
        self.policy = tf.keras.models.load_model(model_dir)

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


def main():
    tf.config.run_functions_eagerly(True)
    model = NaivePolicyGradientModel(10, 2, 0.001, 2)
    state = [0.5, 1.0]
    action = model.produce_action(np.array([state]))
    print(action)
    logits = model(np.array([state]))
    print(logits)
    probabilities = model.get_probabilities(logits)
    print(probabilities)
    log_probabilities = model.get_log_probabilities(logits)
    print(log_probabilities)


if __name__ == '__main__':
    main()
