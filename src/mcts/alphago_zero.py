import numpy as np
from src import config


class AlphaGoZero(object):

    def __init__(self, mcts, self_play=False):
        self.mcts = mcts
        self.self_play = self_play
        self.player = 0

    def get_action(self, board, t=1e-3):
        availables = board.availables
        width = board.width
        height = board.height
        if len(availables):
            all_probs = np.zeros(width * height)
            actions, probs = self.mcts.get_actions(board, t=t)
            all_probs[actions] = probs
            if self.self_play:
                action = np.random.choice(actions,
                                          p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))))
                # update mct root node to choosen action node
                self.mcts.do_action(action)
            else:
                # equal choose max visited n
                action = np.random.choice(actions, p=probs)
                self.mcts.reset()
            return action, all_probs
        else:
            print('game is end because of board is full, no position to loc')

    def set_player(self, player):
        self.player = player

    def reset(self):
        self.mcts.reset()

    def __str__(self):
        return "AlphaGoZero player is : {}".format(config.players[self.player])
