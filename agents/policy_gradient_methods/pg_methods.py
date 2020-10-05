import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods import *
from code_utils.config_utils import BaseConfig


class NaiveAgentConfig(BaseConfig):

    def __init__(self, name: str, config_file: Path):
        """
        Agent configurations for Naive Policy Gradient training.
        Args:
            name: The name of the experiment/agent
            config_file: The configurations file (must be .json)
        """
        BaseConfig.__init__(self, config_file)
        self.name = name
        self.training_steps = self.config_dict["training_steps"]
        self.show_every = self.config_dict["show_every"]
        self.learning_rate = self.config_dict["learning_rate"]
        self.batch_size = self.config_dict["batch_size"]
        self.hidden_layer_size = self.config_dict["hidden_layer_size"]
        self.hidden_layers_count = self.config_dict["hidden_layers_count"]
        self.activation = self.config_dict["activation"]
        self.save_policy_every = self.config_dict["save_policy_every"]


class NaivePolicyGradientAgent(BasePolicyGradientAgent):

    def __init__(self, env: Environment, agent_config: NaiveAgentConfig):

        BasePolicyGradientAgent.__init__(self,
                                         env=env,
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
