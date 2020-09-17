import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

import numpy as np

from environments.gridworld_corridor import GridworldCorridor


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
    def get_probabilities(self, logits):
        probabilities = tf.nn.softmax(logits)
        return probabilities

    @tf.function
    def get_log_probabilities(self, logits):
        log_probabilities = tf.nn.log_softmax(logits)
        return log_probabilities

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(inputs)
        logits = self.output_logits(x)
        return logits

    @tf.function
    def train_step(self, sates, actions, rewards):
        with tf.GradientTape() as tape:
            logits = self(sates)
            action_masks = tf.one_hot(actions, self.output_size)
            log_probabilities = tf.reduce_sum(action_masks * self.get_log_probabilities(logits))
            loss = -tf.reduce_mean(rewards * log_probabilities)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    @tf.function
    def produce_action(self, state):
        logits = self(np.array([state]))
        action = tf.random.categorical(logits, 1)
        return tf.squeeze(action)


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

    def get_environment_state(self):
        raise NotImplementedError

    def environment_step(self, action: int):
        """
        Make a move in the environment with given action.
        :param action: The action index
        :return: next_environment_state, reward (float), terminated_environment (bool)
        """
        raise NotImplementedError

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
                action = self.policy.produce_action(current_state)
                next_state, reward, done = self.environment_step(action)

                states_batch.append(current_state)
                actions_batch.append(action)
                episode_reward += reward
                episode_len += 1

            rewards_batch += [episode_reward] * episode_len
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_len)

        return states_batch, rewards_batch, actions_batch, total_rewards, episode_lengths

    def train_policy(self, train_steps: int, batch_size: int):

        for i in range(train_steps):
            states_batch, rewards_batch, actions_batch, total_rewards, episode_lengths = \
                self.collect_experience(batch_size)

            self.policy.train_step(states_batch, actions_batch, rewards_batch)
            print(f"train_step: {i} - mean_rewards: {np.mean(total_rewards)} - "
                  f"mean_lengths: {np.mean(episode_lengths)}")


def main():
    env = GridworldCorridor(goal_reward=0, move_reward=-1)

    tf.config.run_functions_eagerly(True)
    model = NaivePolicyGradientModel(10, env.action_space, 0.001, 2)
    state = env.get_state()
    action = model.produce_action(env.get_state())
    print(action)
    logits = model(np.array([state]))
    print(logits)
    probabilities = model.get_probabilities(logits)
    print(probabilities)
    log_probabilities = model.get_log_probabilities(logits)
    print(log_probabilities)


if __name__ == '__main__':
    main()
