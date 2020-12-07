import numpy as np
import pickle
from src import config


class Game(object):
    def __init__(self, board, show=False):
        self.board = board
        self.show = show
        self.width = self.board.width
        self.height = self.board.height
        self.index = 0
        # 每50次对局记录一次
        self.records = []

    def draw(self, player1, player2):
        if not self.show:
            return
        """Draw the board and show game info"""
        width = self.board.width
        height = self.board.height
        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = self.board.actions.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def self_play(self, alphago_zero):
        self.board.reset()
        X = []
        PI = []
        players = []
        while True:
            action, prob = alphago_zero.get_action(self.board, t=1)
            # store the data
            X.append(self.board.state)
            PI.append(prob)
            players.append(self.board.current_player)
            # do move
            self.board.move(action)
            # draw board
            self.draw(self.board.players[0], self.board.players[1])
            is_end, player = self.board.is_end
            if is_end:
                self.index += 1
                if self.index % 50 == 0:
                    self.records.append(self.board.actions)
                    self.save_record()
                z = np.zeros(len(players))
                if player != -1:
                    # set reword for each position state
                    z[np.array(players) == player] = 1.0
                    z[np.array(players) != player] = -1.0
                if self.show:
                    if player != -1:
                        print("Game over. Winner is :", config.players[player])
                    else:
                        print("Game over. Tie")
                return np.array(X), np.array(PI), np.array(z)

    def play(self, alphago_zero, mcts_zero):
        self.board.reset()
        p1, p2 = self.board.players
        alphago_zero.set_player(p1)
        mcts_zero.set_player(p2)
        players = {p1: alphago_zero, p2: mcts_zero}
        self.draw(p1, p2)
        while True:
            current_player = self.board.current_player
            _zero = players[current_player]
            action = _zero.get_action(self.board)
            self.board.move(action)
            self.draw(p1, p2)
            is_end, player = self.board.is_end
            if is_end:
                if self.show:
                    if player != -1:
                        print("Game over. Winner is :", config.players[player])
                    else:
                        print("Game over. Tie")
                return player

    def save_record(self):
        with open(config.record_path, 'wb') as f:
            pickle.dump(self.record, f, pickle.HIGHEST_PROTOCOL)

    def load_record(self):
        with open(config.record_path, 'rb') as f:
            return pickle.load(f)
