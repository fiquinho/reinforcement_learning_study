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


class BasicPolicyGradientModel(Model):
    """
    Feed Forward Neural Network that represents a stochastic policy.
    The input shapes are built on first usage so any environment can be used.
    The output shape should be the number of possible actions.
    """

    def __init__(self, layer_size: int, output_size: int, learning_rate: float,
                 hidden_layers_count: int, activation: str="relu"):
        """
        Create a new FFNN model to represent a policy.

        :param layer_size: The number of neurons on each hidden layer.
        :param output_size: The number of neurons on output layer
                            (number of possible actions).
        :param learning_rate: The training step size.
        :param hidden_layers_count: The number of FF layers before the output layer.
        :param activation: Activation function for hidden layer neurons.
        """

        super(BasicPolicyGradientModel, self).__init__()
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
    def call(self, inputs: tf.Tensor, training=None, mask=None):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        logits = self.output_logits(x)
        return logits

    @tf.function
    def train_step(self, sates: tf.Tensor, actions: tf.Tensor, weights: tf.Tensor):
        with tf.GradientTape() as tape:
            logits = self(sates)
            action_masks = tf.one_hot(actions, self.output_size)
            log_probabilities = tf.reduce_sum(action_masks * self.get_log_probabilities(logits), axis=-1)
            loss = -tf.reduce_mean(weights * log_probabilities)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def get_probabilities(self, logits: tf.Tensor):
        probabilities = tf.nn.softmax(logits)
        return probabilities

    @tf.function
    def get_log_probabilities(self, logits: tf.Tensor):
        log_probabilities = tf.nn.log_softmax(logits)
        return log_probabilities

    @tf.function
    def produce_actions(self, states: tf.Tensor):
        logits = self(states)
        actions = tf.random.categorical(logits, 1)
        return actions

    @tf.function
    def get_policy_values(self, states: tf.Tensor):
        logits = self(states)
        predictions = self.get_probabilities(logits)
        return predictions


class TrainingExperience(object):
    """
    A batch of collected experience to do a training step in the network
    """
    def __init__(self, states: list, weights: list, actions: list,
                 total_rewards: list, episode_lengths: list):
        """
        Create instance of collected experiences to be feed to the network
        :param states: The list of states
        :param weights: The weight used by the current algorithm to approximate
                        the network gradients (ie. total reward, reward to go, etc.)
        :param actions: The list of actions
        :param total_rewards: The total reward obtained in each episode of the collected experience.
        :param episode_lengths: The length of each episode of the collected experience.
        """
        assert len(states) == len(weights) == len(actions)
        assert len(total_rewards) == len(episode_lengths)

        self.states = np.array(states, dtype=np.float32)
        self.weights = np.array(weights, dtype=np.float32)
        self.actions = np.array(actions, dtype=np.int32)
        self.total_rewards = np.array(total_rewards, dtype=np.float32)
        self.episode_lengths = np.array(episode_lengths, dtype=np.int32)


class Episode(object):
    """
    A single episode of an environment
    """
    def __init__(self, states: list, actions: list, rewards: list):
        assert len(states) == len(rewards) == len(actions)
        self.states = np.array(states, dtype=np.float32)
        self.rewards = np.array(rewards, dtype=np.float32)
        self.actions = np.array(actions, dtype=np.int32)
        self.total_reward = np.sum(self.rewards)

    def __len__(self):
        return len(self.states)


class EpisodesBatch(object):

    def __init__(self, max_size: int):
        self.episodes = []
        self.max_size = max_size
        self.current_size = 0

    def __len__(self) -> int:
        return self.current_size

    def __iter__(self):
        for episode in self.episodes:
            yield episode

    def add_episode(self, episode: Episode):
        if self.current_size < self.max_size:
            self.episodes.append(episode)
            self.current_size += len(episode)
        else:
            raise ValueError(f"The batch is full! max_size: {self.max_size} -"
                             f" current_size: {self.current_size}")

    def is_full(self) -> bool:
        return self.current_size >= self.max_size


class BasePolicyGradientAgent(object):
    """
    Base class for basic policy gradient algorithms.
     - Creates a FFNN model to use as policy.
     - Has training logic
    """

    def __init__(self, output_size: int, layer_size: int, learning_rate: float,
                 hidden_layers_count: int, activation: str):
        """
        Create an agent that uses a FFNN model to represent its policy.

        :param layer_size: The number of neurons on each hidden layer.
        :param output_size: The number of neurons on output layer
                            (number of possible actions).
        :param learning_rate: The training step size.
        :param hidden_layers_count: The number of FF layers before the output layer.
        :param activation: Activation function for hidden layer neurons.
        """

        self.policy = BasicPolicyGradientModel(layer_size=layer_size,
                                               output_size=output_size,
                                               learning_rate=learning_rate,
                                               hidden_layers_count=hidden_layers_count,
                                               activation=activation)

    def reset_environment(self):
        """
        Reset the environment to start a new episode.
        """
        raise NotImplementedError

    def get_environment_state(self) -> np.array:
        """
        Get the current state of the environment. Must be ready to feed to
        the neural network.
        :return The current state (np.array)
        """
        raise NotImplementedError

    def environment_step(self, action: int):
        """
        Make a move in the environment with given action.
        :param action: The action index.
        :return: next_environment_state (np.array), reward (float), terminated_environment (bool)
        """
        raise NotImplementedError

    def get_possible_states(self) -> np.array:
        """
        Returns a list of every possible environment state, or a sample of them.
        :return: List of states ready to be feed into de neural network (np.array)
        """
        raise NotImplementedError

    def policy_values_plot(self, save_fig: Path=None, show_plot: bool=False):
        """
        TODO: Change name and usage (policy_values_info)
        """
        raise NotImplementedError

    def render_environment(self):
        """
        Render the environment to see the agent playing it.
        """
        raise NotImplementedError

    def get_training_experience(self, episodes: EpisodesBatch) -> TrainingExperience:
        """
        Transforms an EpisodesBatch into a TrainingExperience.
        Each algorithm should implement this with the appropriate conversion.
        :param episodes: An EpisodesBatch object
        :return: A TrainingExperience object
        """
        raise NotImplementedError

    def collect_experience(self, size: int) -> TrainingExperience:
        """
        Collects a batch of steps in the environment using the current policy
        to be feed to the neural network.

        :param size: Batch size
        :return: An ExperienceBatch object with the collected steps information.
        """

        episodes_batch = EpisodesBatch(max_size=size)

        while not episodes_batch.is_full():
            self.reset_environment()
            done = False
            states = []
            rewards = []
            actions = []

            while not done:
                current_state = self.get_environment_state()
                tf_current_state = tf.constant(np.array([current_state]))
                action = self.policy.produce_actions(tf_current_state)[0][0]
                next_state, reward, done = self.environment_step(action)

                states.append(current_state)
                actions.append(action)
                rewards.append(reward)

            episode = Episode(states, actions, rewards)
            episodes_batch.add_episode(episode=episode)

        training_experience = self.get_training_experience(episodes=episodes_batch)

        return training_experience

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
            training_experience = self.collect_experience(batch_size)
            mean_reward = np.mean(training_experience.total_rewards)

            if show_every is not None:
                if i > 0 and not i % show_every:
                    logger.info("====================================================")
                    logger.info(f"Training step N° {i}")
                    logger.info(f"Batch time = {time.time() - start_time} sec")
                    logger.info(f"Last {len(training_experience.total_rewards)} episodes reward mean: {mean_reward}")
                    start_time = time.time()

            states_batch = tf.constant(training_experience.states)
            actions_batch = tf.constant(training_experience.actions)
            weights_batch = tf.constant(training_experience.weights)

            self.policy.train_step(states_batch, actions_batch, weights_batch)
            train_steps_avg_rewards.append(mean_reward)

            if save_policy_every is not None:
                if not i % save_policy_every:
                    # TODO: Make this better changing policy_values_plot to something more generic
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
        plt.xlabel("Training step #")
        plt.title("Reward moving average")

        if agent_folder is not None:
            plt.savefig(Path(agent_folder, "reward_moving_average.png"))
