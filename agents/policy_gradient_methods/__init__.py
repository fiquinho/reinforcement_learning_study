from .models import feed_forward_discrete_model_constructor, feed_forward_continuous_model_constructor
from agents.policy_gradient_methods.envs import *
from .base_pg import BasePolicyGradientAgent, TrainingExperience
from .pg_methods import NaivePolicyGradientAgent, RewardToGoPolicyGradientAgent, \
    BaseAgentConfig, REINFORCEPolicyGradientAgent, REINFORCEAgentConfig


ENVIRONMENTS = {"CartPole-v0": CartPoleEnvironment,
                "Acrobot-v1": AcrobotEnvironment,
                "HeuristicMountainCar-v0": HeuristicMountainCarEnvironment,
                "MoveToGoalSimpleSmall": MoveToGoalSimpleSmallEnvironment,
                "Catcher": CatcherEnvironment,
                "FlappyBird": FlappyBirdEnvironment,
                "Pendulum-v0": PendulumEnvironment,
                "SimpleContinuous": SimpleContinuousEnvironment}

PG_METHODS = {"naive": {"agent": NaivePolicyGradientAgent,
                        "config": BaseAgentConfig},
              "reward_to_go": {"agent": RewardToGoPolicyGradientAgent,
                               "config": BaseAgentConfig},
              "REINFORCE": {"agent": REINFORCEPolicyGradientAgent,
                            "config": REINFORCEAgentConfig}
              }
