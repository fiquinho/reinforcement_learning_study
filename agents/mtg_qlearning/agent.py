import numpy as np

from environments.move_to_goal.move_to_goal import MoveToGoal, GameObject


class Agent(object):

    def __init__(self, player: GameObject):

        self.player = player

    def produce_action(self):

        if self.player.actions is None:
            raise ValueError("Please set the object actions attribute first")

        return np.random.randint(0, len(self.player.actions))


test_game = MoveToGoal(10, 20)
test_game.add_player(GameObject("player1"))
test_game.add_goal(GameObject("goal"))
test_game.prepare_game(player_pos=(0, 0), goal_pos=(9, 9))

test_agent = Agent(player=test_game.player)

test_game.play_game(agent=test_agent, human_view=False)
