from .models import feed_forward_model_constructor
from agents.policy_gradient_methods.envs import *
from .base_pg import BasePolicyGradientAgent, TrainingExperience
from .pg_methods import NaivePolicyGradientAgent, RewardToGoPolicyGradientAgent, \
    BaseAgentConfig, REINFORCEPolicyGradientAgent, REINFORCEAgentConfig


ENVIRONMENTS = {"CartPole-v0": CartPoleEnvironment,
                "Acrobot-v1": AcrobotEnvironment,
                "HeuristicMountainCar-v0": HeuristicMountainCarEnvironment,
                "MoveToGoalSimpleSmall": MoveToGoalSimpleSmallEnvironment,
                "Catcher": CatcherEnvironment}

PG_METHODS = {"naive": {"agent": NaivePolicyGradientAgent,
                        "config": BaseAgentConfig},
              "reward_to_go": {"agent": RewardToGoPolicyGradientAgent,
                               "config": BaseAgentConfig},
              "REINFORCE": {"agent": REINFORCEPolicyGradientAgent,
                            "config": REINFORCEAgentConfig}
              }
