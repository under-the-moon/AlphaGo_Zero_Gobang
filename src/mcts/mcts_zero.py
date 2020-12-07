import numpy as np
from src import config


class MCTSZero(object):
    def __init__(self, mcts):
        self.mcts = mcts
        self.player = 0

    def get_action(self, board):
        availables = board.availables
        width = board.width
        height = board.height
        if len(availables):
            actions, probs = self.mcts.get_actions(board)
            # choose 最大概率的action
            action = actions[np.argmax(probs)]
            return action, np.zeros(width * height)

    def set_player(self, player):
        self.player = player

    def __str__(self):
        return "MCTSZero player is : {}".format(config.players[self.player])
