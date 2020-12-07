import numpy as np


class Model(object):

    def __init__(self):
        self.pure_mcts = True

    def policy_value(self, board):
        probs = np.zeros(board.width * board.height)
        action_probs = np.ones(len(board.availables)) / len(board.availables)
        probs[list(board.availables)] = action_probs
        return probs, 0
