import os
import sys
from pathlib import Path

import numpy as np

from agents.policy_gradient_methods.base_pg import EpisodesBatch, TrainingExperience

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.policy_gradient_methods import BasePolicyGradientAgent


class NaivePolicyGradientAgent(BasePolicyGradientAgent):

    def __init__(self, layer_size: int, learning_rate: float,
                 hidden_layers_count: int, activation: str, output_size: int):

        BasePolicyGradientAgent.__init__(self,
                                         layer_size=layer_size,
                                         output_size=output_size,
                                         learning_rate=learning_rate,
                                         hidden_layers_count=hidden_layers_count,
                                         activation=activation)

    def reset_environment(self):
        raise NotImplementedError

    def get_environment_state(self) -> np.array:
        raise NotImplementedError

    def environment_step(self, action: int):
        raise NotImplementedError

    def get_possible_states(self) -> np.array:
        raise NotImplementedError

    def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
        raise NotImplementedError

    def render_environment(self):
        raise NotImplementedError

    def get_training_experience(self, episodes: EpisodesBatch) -> TrainingExperience:
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
