import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


def feed_forward_model_constructor(input_dim, output_dim):

    class FeedForwardPolicyGradientModel(Model):
        """
        Feed Forward Neural Network that represents a stochastic policy.
        The input shapes are built on first usage so any environment can be used.
        The output shape should be the number of possible actions.
        """

        def __init__(self, layer_size: int, learning_rate: float,
                     hidden_layers_count: int, activation: str="relu"):
            """
            Create a new FFNN model to represent a policy.

            :param layer_size: The number of neurons on each hidden layer.
            :param learning_rate: The training step size.
            :param hidden_layers_count: The number of FF layers before the output layer.
            :param activation: Activation function for hidden layer neurons.
            """

            super(FeedForwardPolicyGradientModel, self).__init__()
            self.layer_size = layer_size
            self.output_size = output_dim
            self.learning_rate = learning_rate
            self.hidden_layers_count = hidden_layers_count
            self.activation = activation

            self.input_layer = Dense(layer_size, activation=activation)
            self.hidden_layers = []
            for i in range(self.hidden_layers_count - 1):  # The input layer is counted in hidden_layers_count
                self.hidden_layers.append(Dense(layer_size, activation=activation))

            self.output_logits = Dense(output_dim, activation=None)

            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        def get_config(self):
            return {"layer_size": self.layer_size,
                    "output_size": self.output_size,
                    "learning_rate": self.learning_rate,
                    "hidden_layers_count": self.hidden_layers_count,
                    "activation": self.activation}

        @tf.function(input_signature=(tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32), ))
        def call(self, inputs: tf.Tensor):
            print("Check retrace! call")
            x = self.input_layer(inputs)
            for layer in self.hidden_layers:
                x = layer(x)
            logits = self.output_logits(x)
            return logits

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32),
                                      tf.TensorSpec(shape=[None], dtype=tf.int32),
                                      tf.TensorSpec(shape=[None], dtype=tf.float32)])
        def train_step(self, sates: tf.Tensor, actions: tf.Tensor, weights: tf.Tensor):
            print("Check retrace! train_step")
            with tf.GradientTape() as tape:
                logits = self(sates)
                action_masks = tf.one_hot(actions, self.output_size)
                log_probabilities = tf.reduce_sum(action_masks * self.get_log_probabilities(logits), axis=-1)
                loss = -tf.reduce_mean(weights * log_probabilities)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, output_dim], dtype=tf.float32)])
        def get_probabilities(self, logits: tf.Tensor):
            print("Check retrace! get_probabilities")
            probabilities = tf.nn.softmax(logits)
            return probabilities

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, output_dim], dtype=tf.float32)])
        def get_log_probabilities(self, logits: tf.Tensor):
            print("Check retrace! get_log_probabilities")
            log_probabilities = tf.nn.log_softmax(logits)
            return log_probabilities

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32)])
        def produce_actions(self, states: tf.Tensor):
            print("Check retrace! produce_actions")
            logits = self(states)
            actions = tf.random.categorical(logits, 1)
            return actions

        @tf.function(input_signature=[tf.TensorSpec(shape=[None, input_dim], dtype=tf.float32)])
        def get_policy_values(self, states: tf.Tensor):
            print("Check retrace! get_policy_values")
            logits = self(states)
            predictions = self.get_probabilities(logits)
            return predictions

    return FeedForwardPolicyGradientModel
