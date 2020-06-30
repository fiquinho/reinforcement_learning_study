# Move to Goal using Q-Learning
This environment was created for this project. The game is a grid and the objective is to reach the goal.
A couple of versions of the game exists to try different things with scalable difficulties.

![MTG Enemy display](https://i.ibb.co/fN9ZB5L/image.png)

There is an agent training script for each current version of the game.
The agents use Q-Learning to come up with a good strategy to play the game.
The available training scripts at the moment are:

    python agent_simple.py
    
    python agent_enemy.py

To run your own experiments please use the help flag to check the available configurations.
    
    python agent_simple.py -h

    python agent_enemy.py -h