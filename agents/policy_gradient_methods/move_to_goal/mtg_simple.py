import tensorflow as tf

from environments.move_to_goal import MoveToGoalSimple
from agents.policy_gradient_methods import BaseNaivePolicyGradientAgent


class NaivePolicyGradientAgent(BaseNaivePolicyGradientAgent):

    def __init__(self, game: MoveToGoalSimple, layer_size: int, learning_rate: float,
                 hidden_layers_count: int, activation: str):
        self.game = game

        BaseNaivePolicyGradientAgent.__init__(self,
                                              layer_size=layer_size,
                                              output_size=self.game.action_space,
                                              learning_rate=learning_rate,
                                              hidden_layers_count=hidden_layers_count,
                                              activation=activation)

    def reset_environment(self):
        self.game.prepare_game()

    def get_environment_state(self):
        self.game.get_state()

    def environment_step(self, action: int):
        next_state, reward, done = self.game.step(action)
        return next_state, reward, done


def main():
    tf.config.run_functions_eagerly(True)
    game = MoveToGoalSimple(board_x=4, board_y=5, goal_reward=1, move_reward=-1, game_end=50,
                            goal_initial_pos=(3, 4), player_initial_pos=(0, 0))

    agent = NaivePolicyGradientAgent(game=game, layer_size=10, learning_rate=0.001, activation="relu",
                                     hidden_layers_count=2)

    agent.train_policy(train_steps=10, batch_size=20)


if __name__ == '__main__':
    main()
