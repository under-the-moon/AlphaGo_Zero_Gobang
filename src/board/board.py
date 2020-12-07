import numpy as np
from src import config


class Board(object):

    def __init__(self):
        self.width = config.default_width
        self.height = config.default_height
        self.players = [0, 1]
        # record action
        self.actions = {}
        self.actions_num = self.width * self.height
        self.availables = set(range(self.actions_num))
        self.n_in_row = 5
        self.current_player = self.players[0]

    def reset(self):
        assert self.width > self.n_in_row and self.height > self.n_in_row, 'board cant init please check border size'
        self.current_player = self.players[0]
        self.availables = list(range(self.width * self.height))
        self.actions = {}

    def action_2_loc(self, action):
        h = action // self.width
        w = action % self.width
        return h, w

    def loc_2_action(self, loc):
        h, w = loc
        action = h * self.width + w
        return action

    def move(self, action):
        """
        移动一步
        :param action:
        :return:
        """
        self.actions[action] = self.current_player
        # change the player
        self.current_player = self.players[1] if self.current_player == self.players[0] else self.players[0]
        self.availables.remove(action)
