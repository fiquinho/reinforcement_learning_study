from .models import feed_forward_model_constructor
from .environments import EpisodesBatch, Episode, Environment, CartPoleEnvironment
from .base_pg import BasePolicyGradientAgent, TrainingExperience
from .pg_methods import NaivePolicyGradientAgent, NaiveAgentConfig


ENVIRONMENTS = {"CartPole-v0": CartPoleEnvironment}

PG_METHODS = {"naive": {"agent": NaivePolicyGradientAgent,
                        "config": NaiveAgentConfig}}
