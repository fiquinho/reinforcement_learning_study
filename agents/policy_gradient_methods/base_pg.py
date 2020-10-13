import logging
import time
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods import *


logger = logging.getLogger()


class TrainingExperience(object):
    """A batch of collected experience to do a training step in the network"""

    def __init__(self, states: list, weights: list, actions: list,
                 total_rewards: list, episode_lengths: list):
        """Create instance of collected experiences to be feed to the network.

        Args:
            states: The list of states
            weights: The weight used by the current algorithm to approximate
                the network gradients (ie. total reward, reward to go, etc.)
            actions: The list of actions
            total_rewards: The total reward obtained in each episode of the collected experience.
            episode_lengths: The length of each episode of the collected experience.
        """
        assert len(states) == len(weights) == len(actions)
        assert len(total_rewards) == len(episode_lengths)

        self.states = np.array(states, dtype=np.float32)
        self.weights = np.array(weights, dtype=np.float32)
        self.actions = np.array(actions, dtype=np.int32)
        self.total_rewards = np.array(total_rewards, dtype=np.float32)
        self.episode_lengths = np.array(episode_lengths, dtype=np.int32)

    def __len__(self):
        """
        Returns:
            The total number of stored environment steps
        """
        return len(self.states)


class BasePolicyGradientAgent(object):
    """
    Base class for basic policy gradient algorithms.
     - Creates a FFNN model to use as policy.
     - Has training logic
    """

    def __init__(self, env: Environment, agent_path: Path, layer_size: int,
                 learning_rate: float, hidden_layers_count: int, activation: str):
        """Create an agent that uses a FFNN model to represent its policy.

        Args:
            env: The environment the agent is trying to solve
            layer_size: The number of neurons on each hidden layer
            learning_rate: The training step size
            hidden_layers_count: The number of FF layers before the output layer
            activation: Activation function for hidden layer neurons
        """
        self.env = env
        self.agent_path = agent_path
        model_path = Path(agent_path, "model")
        policy_constructor = feed_forward_model_constructor(env.state_space_n, env.action_space_n)
        self.policy = policy_constructor(model_path=model_path,
                                         layer_size=layer_size,
                                         learning_rate=learning_rate,
                                         hidden_layers_count=hidden_layers_count,
                                         activation=activation)
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=self.policy.optimizer,
                                        net=self.policy)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                       str(self.policy.model_path) + "\\checkpoints", max_to_keep=3)

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
            self.env.reset_environment()
            done = False
            states = []
            rewards = []
            actions = []

            while not done:
                current_state = self.env.get_environment_state()
                tf_current_state = tf.constant(np.array([current_state]), dtype=tf.float32)
                action = self.policy.produce_actions(tf_current_state)[0][0]
                next_state, reward, done = self.env.environment_step(action)

                states.append(current_state)
                actions.append(action)
                rewards.append(reward)

            episode = Episode(states, actions, rewards)
            episodes_batch.add_episode(episode=episode)

        training_experience = self.get_training_experience(episodes=episodes_batch)

        return training_experience

    def train_policy(self, train_steps: int, experience_size: int,
                     save_policy_every: int=None, show_every: int=None,
                     minibatch_size: int=None):
        """Train the agent to solve the current environment.

        Args:
            train_steps: The number of training steps
            experience_size: The number of environment steps used on each training step
            show_every: How often to show training info (in training steps)
            save_policy_every: How often to save policy information during training
                               (in training steps)
            minibatch_size: How many environment steps are pass to the NN at once.
                            If None, the total number of steps collected for each
                            training step is used (experience_size)
        """

        train_steps_avg_rewards = []
        start_time = time.time()
        training_steps = 0
        for i in range(train_steps):
            training_experience = self.collect_experience(experience_size)
            mean_reward = np.mean(training_experience.total_rewards)

            if show_every is not None:
                if i > 0 and not i % show_every:
                    logger.info("====================================================")
                    logger.info(f"Training step N° {i}")
                    logger.info(f"Batch time = {time.time() - start_time} sec")
                    logger.info(f"Last {len(training_experience.total_rewards)} episodes reward mean: {mean_reward}")
                    start_time = time.time()

            states_batch = tf.constant(training_experience.states, dtype=np.float32)
            actions_batch = tf.constant(training_experience.actions, dtype=np.int32)
            weights_batch = tf.constant(training_experience.weights, dtype=np.float32)

            batch_size = minibatch_size if minibatch_size is not None else len(states_batch)
            data = tf.data.Dataset.from_tensor_slices((states_batch, actions_batch, weights_batch))
            # TODO: Check if randomizing has some effect
            data = data.shuffle(buffer_size=len(states_batch)).batch(batch_size)

            for minibatch_step, data_batch in enumerate(data):

                if i == 0 and minibatch_step == 0:
                    tf.summary.trace_on(graph=True, profiler=True)
                    # Call only one tf.function when tracing.
                    self.policy(data_batch[0])
                    with self.policy.summary_writer.as_default():
                        tf.summary.trace_export(name="policy_call", step=0,
                                                profiler_outdir=str(self.policy.train_log_dir))
                    self.policy.summary_writer.flush()

                logits, loss, log_probabilities = self.policy.train_step(
                    data_batch[0], data_batch[1], data_batch[2])

                if minibatch_step == len(data) - 1:
                    with self.policy.summary_writer.as_default():
                        probabilities = self.policy.get_probabilities(logits)
                        for action_idx, action in enumerate(self.env.actions):
                            action_probs = probabilities[:, action_idx]
                            tf.summary.histogram(f"{action}_prob", data=action_probs, step=training_steps)
                        if self.env.state_names is not None:
                            for state_idx, state in enumerate(self.env.state_names):
                                state_attribute_hist = data_batch[0][:, state_idx]
                                tf.summary.histogram(f"{state}", data=state_attribute_hist, step=training_steps)

                        tf.summary.scalar("mean_reward", data=mean_reward, step=training_steps)
                        tf.summary.histogram("logits", data=logits, step=training_steps)
                        tf.summary.scalar("loss", data=loss, step=training_steps)
                        tf.summary.histogram("log_probabilities", data=log_probabilities, step=training_steps)
                        tf.summary.histogram("weights", data=data_batch[2], step=training_steps)
                    self.policy.summary_writer.flush()

            training_steps += 1
            train_steps_avg_rewards.append(mean_reward)

            self.ckpt.step.assign_add(1)
            if save_policy_every is not None:
                if not i % save_policy_every:
                    save_path = self.ckpt_manager.save()
                    logger.info(f"Checkpoint saved for step {int(self.ckpt.step)}: {save_path}")

        moving_avg = np.convolve(train_steps_avg_rewards, np.ones((show_every,)) / show_every, mode='valid')

        self.save_agent()
        self.plot_training_info(moving_avg, self.agent_path)

    def save_agent(self):
        """Save the policy neural network to files in the model path."""

        logger.info(f"Saving trained policy to {self.policy.model_path}")
        start = time.time()
        self.policy.save(self.policy.model_path)
        logger.info(f"Saving time {time.time() - start}")

    def load_model(self, model_dir: Path):
        """
        Load a trained policy from files.
        :param model_dir: Where the trained model is stored.
        """
        if Path(model_dir, "saved_model.pb").exists():
            self.policy = tf.keras.models.load_model(model_dir)
        else:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            logger.info(f"Restored model from checkpoint {self.ckpt_manager.latest_checkpoint}")

    @staticmethod
    def plot_training_info(moving_avg: np.array, agent_folder: Path=None):
        """
        Plot the reward moving average during training.
        :param moving_avg: The moving average data
        :param agent_folder: Where to save the generated plot
        """
        plt.figure(figsize=(5, 5))

        # Moving average plot
        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel(f"Reward")
        plt.xlabel("Training step #")
        plt.title("Reward moving average")

        if agent_folder is not None:
            plt.savefig(Path(agent_folder, "reward_moving_average.png"))

    def play_game(self, plot_game: bool=False, delay: float=None):
        self.env.reset_environment()
        done = False
        states = []
        rewards = []
        actions = []
        while not done:
            if plot_game:
                self.env.render_environment()
                if delay is not None:
                    time.sleep(delay)

            state = self.env.get_environment_state()
            tf_current_state = tf.constant(np.array([state]), dtype=tf.float32)
            action = self.policy.produce_actions(tf_current_state)[0][0]

            new_state, reward, done = self.env.environment_step(action)

            states.append(state)
            rewards.append(reward)
            actions.append(action)

        episode = Episode(states, actions, rewards)
        win = self.env.win_condition(episode)

        self.env.reset_environment()

        return episode, win
