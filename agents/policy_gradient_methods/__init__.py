from .models import feed_forward_model_constructor
from .environments import EpisodesBatch, Episode, Environment, CartPoleEnvironment, AcrobotEnvironment
from .base_pg import BasePolicyGradientAgent, TrainingExperience
from .pg_methods import NaivePolicyGradientAgent, RewardToGoPolicyGradientAgent, StandardAgentConfig


ENVIRONMENTS = {"CartPole-v0": CartPoleEnvironment,
                "Acrobot-v1": AcrobotEnvironment}

PG_METHODS = {"naive": {"agent": NaivePolicyGradientAgent,
                        "config": StandardAgentConfig},
              "reward_to_go": {"agent": RewardToGoPolicyGradientAgent,
                               "config": StandardAgentConfig}
              }
