"""
Neural Network models used to represent stochastic policies.
The models are implemented as sub-classes of tf.keras.Models.
The training loop is custom as well as the training step.
Using @tf.function decorators for performance optimization.
"""

import logging
import math
from pathlib import Path
from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Layer


logger = logging.getLogger()


def feed_forward_discrete_model_constructor(input_dim, output_dim):
    """Creates a tf.keras.Model subclass for a Feed Forward Neural Network
    that represents a categorical stochastic policy.
    Args:
        input_dim: The length of the state vector
        output_dim: The number of possible actions

    Returns:
        A class to instantiate the model object.
    """
    class FeedForwardPolicyGradientModel(Model):
        """Feed Forward Neural Network that represents a categorical stochastic policy.
        The input and output sizes are already defined.
        """

        def __init__(self, model_path: Path, layer_size: int, learning_rate: float,
                     hidden_layers_count: int, activation: str="relu"):
            """
            Creates a new FFNN model to represent a policy. Implements all needed
            methods from tf.keras.Model.
            Args:
                layer_size: The number of neurons on each hidden layer.
                learning_rate: The training step size.
                hidden_layers_count: The number of FF layers before the output layer.
                activation: Activation function for hidden layer neurons.
            """

            super(FeedForwardPolicyGradientModel, self).__init__()
            self.model_path = model_path
            self.layer_size = layer_size
            self.output_size = output_dim
            self.input_size = input_dim
            self.learning_rate = learning_rate
            self.hidden_layers_count = hidden_layers_count
            self.activation = activation

            self.input_layer = Dense(layer_size, activation=activation)
            self.hidden_layers = []
            for i in range(self.hidden_layers_count - 1):  # The input layer is counted in hidden_layers_count
                self.hidden_layers.append(Dense(layer_size, activation=activation))

            self.output_logits = Dense(output_dim, activation=None, name="output_logits")

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            self.train_log_dir = Path(model_path, "train_log")
            self.summary_writer = tf.summary.create_file_writer(str(self.train_log_dir))

        def get_config(self):
            """Used by tf.keras to load a saved model."""
            return {"layer_size": self.layer_size,
                    "learning_rate": self.learning_rate,
                    "hidden_layers_count": self.hidden_layers_count,
                    "activation": self.activation}

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32), ))
        def call(self, inputs: tf.Tensor):
            """See base Class."""

            logger.info("[Retrace] call")
            x = self.input_layer(inputs)
            for layer in self.hidden_layers:
                x = layer(x)
            logits = self.output_logits(x)
            return logits

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None], dtype=tf.int32),
                                      tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def train_step(self, sates: tf.Tensor, actions: tf.Tensor,
                       weights: tf.Tensor) -> (Tuple[tf.Tensor], tf.Tensor, tf.Tensor):
            """See base Class."""

            logger.info("[Retrace] train_step")
            with tf.GradientTape() as tape:
                logits = self(sates)
                action_masks = tf.one_hot(actions, self.output_size)
                log_probabilities = tf.reduce_sum(action_masks * self._get_log_probabilities(logits), axis=-1)
                loss = -tf.reduce_mean(weights * log_probabilities)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            return (logits, ), loss, log_probabilities

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, output_dim], dtype=tf.float32)])
        def get_probabilities(self, logits: tf.Tensor) -> tf.Tensor:
            """Gets the actual probabilities of each action for each set of logits.

            Args:
                logits: The output of this model: self(states)

            Returns:
                The probabilities (for each set of logits they add up to 1)
            """

            logger.info("[Retrace] get_probabilities")
            probabilities = tf.nn.softmax(logits)
            return probabilities

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, output_dim], dtype=tf.float32)])
        def _get_log_probabilities(self, logits: tf.Tensor) -> tf.Tensor:
            """Gets the logarithmic probabilities of each action for each set of logits.

            Args:
                logits: The output of this model: self(states)

            Returns:
                The logarithmic probabilities
            """

            logger.info("[Retrace] get_log_probabilities")
            log_probabilities = tf.nn.log_softmax(logits)
            return log_probabilities

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32)])
        def produce_actions(self, states: tf.Tensor) -> tf.Tensor:
            """Get a sample from the action probability distribution produced
            by the model, for each passed state.

            Args:
                states: The list of states representations

            Returns:
                The sampled action for each state
            """

            logger.info("[Retrace] produce_actions")
            logits = self(states)
            actions = tf.random.categorical(logits, 1)
            return actions

    return FeedForwardPolicyGradientModel


def feed_forward_continuous_model_constructor(input_dim, output_dim):
    """Creates a tf.keras.Model subclass for a Feed Forward Neural Network
    that represents a stochastic policy for continuous action spaces.
    Args:
        input_dim: The length of the state vector
        output_dim: The number of possible actions

    Returns:
        A class to instantiate the model object.
    """

    class MuBlock(Layer):
        def __init__(self, layer_size: int, hidden_layers_count: int,
                     activation: str="tanh"):
            super(MuBlock, self).__init__(name="Mu")

            self.layer_size = layer_size
            self.output_size = output_dim
            self.input_size = input_dim
            self.hidden_layers_count = hidden_layers_count
            self.activation = activation

            self.input_layer = Dense(layer_size, activation=activation)
            self.hidden_layers = []
            for i in range(self.hidden_layers_count - 1):  # The input layer is counted in hidden_layers_count
                self.hidden_layers.append(Dense(layer_size, activation=activation))

            self.output_mu = Dense(output_dim, activation="tanh")

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32)])
        def call(self, inputs: tf.Tensor):
            """See base Class."""

            logger.info("[Retrace] call Mu")
            x = self.input_layer(inputs)
            for layer in self.hidden_layers:
                x = layer(x)
            mu = self.output_mu(x)
            return mu

    class SigmaBlock(Layer):
        def __init__(self, layer_size: int, hidden_layers_count: int,
                     activation: str = "tanh"):
            super(SigmaBlock, self).__init__(name="Sigma")

            self.layer_size = layer_size
            self.output_size = output_dim
            self.input_size = input_dim
            self.hidden_layers_count = hidden_layers_count
            self.activation = activation

            self.input_layer = Dense(layer_size, activation=activation)
            self.hidden_layers = []
            for i in range(self.hidden_layers_count - 1):  # The input layer is counted in hidden_layers_count
                self.hidden_layers.append(Dense(layer_size, activation=activation))

            self.output_sigma = Dense(output_dim, activation=tf.nn.relu)

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32)])
        def call(self, inputs: tf.Tensor):
            """See base Class."""

            logger.info("[Retrace] call Sigma")
            x = self.input_layer(inputs)
            for layer in self.hidden_layers:
                x = layer(x)
            sigma = self.output_sigma(x)
            return sigma

    class FeedForwardContinuousPolicyGradientModel(Model):
        """Feed Forward Neural Network that represents a categorical stochastic policy
        for continuous action spaces.
        The input and output sizes are already defined.
        """

        def __init__(self, model_path: Path, layer_size: int, learning_rate: float,
                     hidden_layers_count: int, activation: str="tanh"):
            """
            Creates a new FFNN model to represent a policy. Implements all needed
            methods from tf.keras.Model.
            Args:
                layer_size: The number of neurons on each hidden layer.
                learning_rate: The training step size.
                hidden_layers_count: The number of FF layers before the output layer.
                activation: Activation function for hidden layer neurons.
            """

            super(FeedForwardContinuousPolicyGradientModel, self).__init__()
            self.model_path = model_path
            self.layer_size = layer_size
            self.output_size = output_dim
            self.input_size = input_dim
            self.learning_rate = learning_rate
            self.hidden_layers_count = hidden_layers_count
            self.activation = activation

            self.mu = MuBlock(layer_size, hidden_layers_count, activation)
            self.sigma = SigmaBlock(layer_size, hidden_layers_count, activation)

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            self.train_log_dir = Path(model_path, "train_log")
            self.summary_writer = tf.summary.create_file_writer(str(self.train_log_dir))

        def get_config(self):
            """Used by tf.keras to load a saved model."""
            return {"layer_size": self.layer_size,
                    "learning_rate": self.learning_rate,
                    "hidden_layers_count": self.hidden_layers_count,
                    "activation": self.activation}

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32)])
        def call(self, inputs: tf.Tensor):
            """See base Class."""

            logger.info("[Retrace] call")
            mu = self.mu(inputs)
            sigma = self.sigma(inputs)

            return mu, sigma

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None, output_dim], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def train_step(self, states: tf.Tensor, actions: tf.Tensor,
                       weights: tf.Tensor) -> (Tuple[tf.Tensor], tf.Tensor, tf.Tensor):
            """See base Class."""

            logger.info("[Retrace] train_step")
            with tf.GradientTape() as tape:
                mu, sigma = self(states)
                log_probabilities = self._get_log_probabilities(mu, sigma, actions)
                loss = -tf.reduce_mean(weights * log_probabilities)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            return (mu, sigma), loss, log_probabilities

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, output_dim], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None, output_dim], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None, output_dim], dtype=tf.float32)])
        def _get_log_probabilities(self, mu: tf.Tensor, sigma: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
            """Gets the logarithmic probabilities of each action for each set of logits.

            Args:
                mu: The mean value for each action for each step
                sigma: The variance value for each action for each step
                actions: The actual actions used in each step

            Returns:
                The logarithmic probabilities for the actions
            """

            logger.info("[Retrace] get_log_probabilities")

            x1 = actions - mu
            x2 = x1 ** 2
            sigma2 = sigma ** 2
            x3 = x2 / sigma2
            logsigma = tf.math.log(sigma)
            x4 = x3 + (2 * logsigma)
            actions_sum = tf.reduce_sum(x4, axis=-1)
            x5 = actions_sum + self.output_size * tf.math.log(2 * math.pi)
            x6 = - x5 * 0.5
            log_probabilities = x6
            return log_probabilities

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32)])
        def produce_actions(self, states: tf.Tensor) -> tf.Tensor:
            """Get a sample from the action probability distribution produced
            by the model, for each passed state.

            Args:
                states: The list of states representations

            Returns:
                The sampled action for each state
            """

            logger.info("[Retrace] produce_actions")
            mu, sigma = self(states)
            actions = tfp.distributions.Normal(mu, sigma).sample([1])
            return actions

    return FeedForwardContinuousPolicyGradientModel


def test():
    import numpy as np
    tf.config.run_functions_eagerly(True)
    constructor = feed_forward_continuous_model_constructor(3, 2)
    model = constructor(Path("test_model"), 4, learning_rate=0.1, hidden_layers_count=2, activation="tanh")

    state = np.array([[1., 2., 3.]])
    action = np.array([[0.2, -1.8]])
    reward = np.array([0.5])
    mu, sigma = model(state)
    mu, sigma, loss, log_probabilities = model.train_step(state, action, reward)
    log_probabilities = model.get_log_probabilities(mu, sigma, action)
    actions = model.produce_actions(state)
    pass


if __name__ == '__main__':

    test()
    pass
