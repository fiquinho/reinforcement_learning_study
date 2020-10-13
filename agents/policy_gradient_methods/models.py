"""
Neural Network models used to represent stochastic policies.
The models are implemented as sub-classes of tf.keras.Models.
The training loop is custom as well as the training step.
Using @tf.function decorators for performance optimization.
"""

import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


logger = logging.getLogger()


def feed_forward_model_constructor(input_dim, output_dim):
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

            self.output_logits = Dense(output_dim, activation=None)

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
                       weights: tf.Tensor) -> (tf.Tensor, tf.Tensor, tf.Tensor):
            """See base Class."""

            logger.info("[Retrace] train_step")
            with tf.GradientTape() as tape:
                logits = self(sates)
                action_masks = tf.one_hot(actions, self.output_size)
                log_probabilities = tf.reduce_sum(action_masks * self.get_log_probabilities(logits), axis=-1)
                loss = -tf.reduce_mean(weights * log_probabilities)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            return logits, loss, log_probabilities

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
        def get_log_probabilities(self, logits: tf.Tensor) -> tf.Tensor:
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
