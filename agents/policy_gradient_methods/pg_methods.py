import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods import *
from code_utils.config_utils import BaseConfig


class BaseAgentConfig(BaseConfig):

    def __init__(self, name: str, config_file: Path, desc: str=None):
        """Agent configurations for Naive and Reward to Go Policy Gradient training.

        Args:
            name: The name of the experiment/agent
            config_file: The configurations file (must be .json)
        """
        BaseConfig.__init__(self, config_file)
        self.name = name
        self.desc = desc
        self.training_steps = self.config_dict["training_steps"]
        self.show_every = self.config_dict["show_every"]
        self.learning_rate = self.config_dict["learning_rate"]
        self.experience_size = self.config_dict["experience_size"]
        self.minibatch_size = self.config_dict["minibatch_size"]
        self.hidden_layer_size = self.config_dict["hidden_layer_size"]
        self.hidden_layers_count = self.config_dict["hidden_layers_count"]
        self.activation = self.config_dict["activation"]
        self.save_policy_every = self.config_dict["save_policy_every"]


class REINFORCEAgentConfig(BaseAgentConfig):

    def __init__(self, name: str, config_file: Path, desc: str):
        """Agent configurations for REINFORCE Policy Gradient training.

        Args:
            name: The name of the experiment/agent
            config_file: The configurations file (must be .json)
        """
        BaseAgentConfig.__init__(self, name, config_file, desc)
        self.discount_factor = self.config_dict["discount_factor"]


class NaivePolicyGradientAgent(BasePolicyGradientAgent):
    """Agent that implements naive policy gradient to train the policy.

    Naive -> The gradient of the log probability of the actions is weighted by
    the total reward of the episode.

    Found here as Basic Policy Gradient:
        https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
    """

    def __init__(self, env: Environment, agent_path: Path, agent_config: BaseAgentConfig):
        """See base class."""
        BasePolicyGradientAgent.__init__(self,
                                         env=env,
                                         agent_path=agent_path,
                                         layer_size=agent_config.hidden_layer_size,
                                         learning_rate=agent_config.learning_rate,
                                         hidden_layers_count=agent_config.hidden_layers_count,
                                         activation=agent_config.activation)

    def get_training_experience(self, episodes: EpisodesBatch) -> TrainingExperience:
        """See base class."""

        states_batch = []
        weights_batch = []
        actions_batch = []
        total_rewards = []
        episode_lengths = []

        for episode in episodes:
            states_batch.append(episode.states)
            actions_batch.append(episode.actions)
            weights_batch += [episode.total_reward] * len(episode)
            total_rewards.append(episode.total_reward)
            episode_lengths.append(len(episode))

        states_batch = np.concatenate(states_batch, axis=0)
        actions_batch = np.concatenate(actions_batch, axis=0)

        return TrainingExperience(states_batch, weights_batch, actions_batch,
                                  total_rewards, episode_lengths)


class RewardToGoPolicyGradientAgent(BasePolicyGradientAgent):

    def __init__(self, env: Environment, agent_path: Path, agent_config: BaseAgentConfig):

        BasePolicyGradientAgent.__init__(self,
                                         env=env,
                                         agent_path=agent_path,
                                         layer_size=agent_config.hidden_layer_size,
                                         learning_rate=agent_config.learning_rate,
                                         hidden_layers_count=agent_config.hidden_layers_count,
                                         activation=agent_config.activation)

    def get_training_experience(self, episodes: EpisodesBatch) -> TrainingExperience:
        """See base class."""

        states_batch = []
        weights_batch = []
        actions_batch = []
        total_rewards = []
        episode_lengths = []

        for episode in episodes:
            states_batch.append(episode.states)
            actions_batch.append(episode.actions)

            episode_rewards_to_go = []
            for i in range(len(episode.rewards)):
                episode_rewards_to_go.append(sum(episode.rewards[i:]))

            weights_batch.append(episode_rewards_to_go)
            total_rewards.append(episode.total_reward)
            episode_lengths.append(len(episode))

        states_batch = np.concatenate(states_batch, axis=0)
        actions_batch = np.concatenate(actions_batch, axis=0)
        weights_batch = np.concatenate(weights_batch, axis=0)

        return TrainingExperience(states_batch, weights_batch, actions_batch,
                                  total_rewards, episode_lengths)


class REINFORCEPolicyGradientAgent(BasePolicyGradientAgent):

    def __init__(self, env: Environment, agent_path: Path, agent_config: REINFORCEAgentConfig):

        self.discount_factor = agent_config.discount_factor
        BasePolicyGradientAgent.__init__(self,
                                         env=env,
                                         agent_path=agent_path,
                                         layer_size=agent_config.hidden_layer_size,
                                         learning_rate=agent_config.learning_rate,
                                         hidden_layers_count=agent_config.hidden_layers_count,
                                         activation=agent_config.activation)

    def get_training_experience(self, episodes: EpisodesBatch) -> TrainingExperience:
        """See base class."""

        states_batch = []
        weights_batch = []
        actions_batch = []
        total_rewards = []
        episode_lengths = []

        for episode in episodes:
            states_batch.append(episode.states)
            actions_batch.append(episode.actions)

            episode_discounted_rewards_to_go = []
            for i in range(len(episode.rewards)):
                rewards_to_go = episode.rewards[i:]
                discounts = (np.ones(len(rewards_to_go)) * self.discount_factor) ** np.arange(len(rewards_to_go))
                discounted_rewards_to_go = discounts * rewards_to_go
                episode_discounted_rewards_to_go.append(sum(discounted_rewards_to_go))

            weights_batch.append(episode_discounted_rewards_to_go)
            total_rewards.append(episode.total_reward)
            episode_lengths.append(len(episode))

        states_batch = np.concatenate(states_batch, axis=0)
        actions_batch = np.concatenate(actions_batch, axis=0)
        weights_batch = np.concatenate(weights_batch, axis=0)

        return TrainingExperience(states_batch, weights_batch, actions_batch,
                                  total_rewards, episode_lengths)
