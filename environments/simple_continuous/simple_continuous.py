"""
Simple environment for test purposes.
Inspired by this blogpost:
https://towardsdatascience.com/a-minimal-working-example-for-continuous-policy-gradients-in-tensorflow-2-0-d3413ec38c6b
"""

import matplotlib.pyplot as plt
import numpy as np

# Define properties reward function
# mu_target = 4.0
# target_range = 0.1
# max_reward = 1.0
#
# actions = []
# rewards = []
# for i in np.arange(0, mu_target * 2., 0.1):
#     # Compute reward
#     reward = max_reward / max(target_range, abs(mu_target - i)) * target_range
#     actions.append(i)
#     rewards.append(reward)
#
# plt.plot(actions, rewards)
# plt.show()


# Compute reward
# reward = max_reward / max(target_range, abs(mu_target - action)) * target_range


class SimpleContinuous(object):
    """
    Simple one step environment with a fixed best action solution.
    Used for test purposes.
    """

    def __init__(self, target: int, max_reward: float, target_range: float=0.1):
        self.target = target
        self.max_reward = max_reward
        self.target_range = target_range

    def get_state(self):
        return 1

    def step(self, action=float):
        reward = self.max_reward / max(self.target_range, abs(self.target - action)) * self.target_range
        state = np.array(self.get_state())

        return state, reward, True
