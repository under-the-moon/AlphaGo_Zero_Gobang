import numpy as np
from src.board.board import Board
from src import config


# 五子棋盘
class GoBang(Board):
    def __init__(self):
        super(GoBang, self).__init__()
        self.channel = config.channel

    @property
    def state(self):
        # shape (height, width, 5)
        state = np.zeros((self.height, self.width, config.channel), dtype=np.float32)
        if self.actions:
            actions, players = np.array(list(zip(*self.actions.items())))
            actions = actions.astype(np.int32)
            action_curr = actions[players == self.current_player]
            action_oppo = actions[players != self.current_player]
            state[action_curr // self.width, action_curr % self.width, 0] = 1
            state[action_oppo // self.width, action_oppo % self.width, 1] = 1
            if len(actions) % 2 == 0:
                state[:, :, 4] = 1.
            # record last action state
            if len(actions) > 2:
                # 得到上一步的状态
                actions = actions[:-1]
                players = players[:-1]
                action_curr = actions[players == self.current_player]
                action_oppo = actions[players != self.current_player]
                state[action_curr // self.width, action_curr % self.width, 2] = 1
                state[action_oppo // self.width, action_oppo % self.width, 3] = 1
        return state

    @property
    def is_end(self):
        width = self.width
        height = self.height
        actions = self.actions
        n = self.n_in_row
        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1
        for m in moved:
            h = m // width
            w = m % width
            player = actions[m]
            # judge 横着是否有五个一样的棋子
            if w in range(width - n + 1) and len(set(actions.get(i, -1) for i in range(m, m + n))) == 1:
                return True, player
            # judge 竖着是否有五个一样的棋子
            if h in range(height - n + 1) and len(set(actions.get(i, -1) for i in range(m, m + n * width, width))) == 1:
                return True, player
            # \ 斜着是否有五个一样的棋子
            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(actions.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player
            # \ 反斜着是否有五个一样的棋子
            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(actions.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player
        # Tie 平局
        if len(self.availables) == 0:
            return True, -1
        return False, -1
