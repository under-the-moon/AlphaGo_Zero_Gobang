import numpy as np
import sys
from src import config

sys.setrecursionlimit(1000000)


class Node(object):

    def __init__(self, parent, p, action=-1):
        # parent for Backpropagation
        self.parent = parent
        #  Each edge (s, a) in the search tree stores a prior probability P(s, a), a visit count N(s, a),
        # and an action-value Q(s, a)
        # Each node s in the search tree contains edges (s, a) for all legal actions a ∈ A(s). Each edge
        # stores a set of statistics, {N(s, a), W(s, a), Q(s, a), P(s, a)},

        # eg: Additional exploration is achieved by adding Dirichlet noise to the
        # prior probabilities in the root node s0, specifically P(s, a) = (1 )pa + ηa, where η ∼ Dir(0.03)
        self.p = p
        # N(s, a) is the visit count,
        self.n = 0
        # W(s, a) is the total action-value
        self.w = 0.0
        # Q(s, a) is the mean action-value,
        self.q = 0.0
        # Each simulation starts from the root state and iteratively selects
        # moves that maximise an upper confidence bound Q(s, a) +U(s, a), where U(s, a) ∝ P(s, a)/(1 + N(s, a))
        self.u = 0.0
        # The MCTS search outputs probabilities π of playing each move.
        # search probabilities π are returned, proportional to N1/τ
        # self.PI = 0.0
        # evaluate : t->0 也就是最大贪婪 we deterministically select the move with maximum visit count, to give the strongest possible play
        self.t = config.t
        # save available actions mapping TreeNode
        self.childrens = {}
        self.c_put = config.c_put
        self.action = action

    def expansion(self, action_probs):
        for action, prob in action_probs:
            if action not in self.childrens.keys():
                self.childrens[action] = Node(self, prob, action)
            else:
                self.childrens[action].p = prob
                self.childrens[action].action = action

    def update(self, value):
        if self.parent:
            self.parent.update(-value)
        self._update(value)

    def _update(self, value):
        # Q(s, a) is the mean action-value,
        #  Xmean = (X1 + X2 + ... + Xm) / m
        #  update_q = (X1 + X2 + ... + Xm+1) / (m + 1) = (m / m + 1) * Xmean + Xm+1 / (m + 1)
        #  = Xmean + (Xmean - Xm+1) / (m + 1)  也就是下面的均值计算
        self.n += 1
        self.q += 1.0 * (value - self.q) / self.n
        # 重新计算节点 π
        # self.PI = self.get_PI()
        self.w += value

    # def get_PI(self):
    #     if self.parent is None:
    #         return 1
    #     return self.n / self.parent.n

    def get_ucb(self):
        """
            Q(s, a) + U(s, a), where U(s, a) ∝ P(s, a)/(1 + N(s, a)) used in AlphaGo Fan and AlphaGo Lee.
            AlphaGo Zero uses a much simpler  c * p * (sqrt(∑N(s, b)) / (1 + N(s, a)))
            ∑N(s, b) equal parent visited
        :param c:
        :return:
        """
        self.u = self.c_put * self.p * (np.sqrt(self.parent.n) / (1 + self.n))
        return self.q + self.u

    def select(self):
        # choose max upper confidence bound --> ucb
        return max(self.childrens.items(), key=lambda node: node[1].get_ucb())

    @property
    def is_leaf(self):
        return len(self.childrens) == 0

    @property
    def is_root(self):
        return self.parent is None
