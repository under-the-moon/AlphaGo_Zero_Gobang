import random
import numpy as np
from collections import defaultdict, deque
from src import config
from src.mcts.mcts import MCTS
from src.mcts.alphago_zero import AlphaGoZero
from src.mcts.mcts_zero import MCTSZero
from src.models.mcts import Model
import time


class PipelineTrain(object):
    def __init__(self, game, alphago_zero, model, epoch):
        self.game = game
        self.model = model
        self.alphago_zero = alphago_zero
        self.width = self.game.board.width
        self.height = self.game.board.height
        self.buffer_data = deque(maxlen=config.buffer_size)
        self.batch_size = config.batch_size
        self.epoch = epoch

    def train(self):
        print('train alphago zero')
        best_loss = np.inf
        print('self play 10 games collect data start')
        start = time.time()
        self.add_selfplay_data(10)
        end = time.time()
        print('self play 10 games collect data end cost: {}'.format(end - start))
        for i in range(self.epoch):
            print('start to sample train data')
            batch_data = random.sample(self.buffer_data,
                                       self.size if self.size < self.batch_size * 100 else self.batch_size * 100)
            X = [data[0] for data in batch_data]
            PI = [data[1] for data in batch_data]
            z = [data[2] for data in batch_data]
            X = np.array(X)
            y = [np.array(PI), np.array(z)]
            start = time.time()
            loss = self.model.train(X, y)
            print('epoch: {} loss: {} cost: {}'.format(i + 1, loss, time.time() - start))
            if (i + 1) % 100 == 0:
                print('save weights...')
                print('loss < best_loss', loss < best_loss)
                if loss < best_loss:
                    self.model.save_weights(i + 1)
                    best_loss = loss
                    # self.evaluate()
            # self play
            self.add_selfplay_data()
        # save self play record
        # print('save record to visualize')
        # self.save_record()

    def save_record(self):
        self.game.save_record()

    def evaluate(self, game=10):
        print('start to evaluate')
        mcts = MCTS(self.model)
        alphago_zero = AlphaGoZero(mcts, self_play=False)
        mcts_ = MCTS(Model())
        mcts_zero = MCTSZero(mcts_)
        win_cnt = defaultdict(int)
        for i in range(game):
            player = self.game.play(alphago_zero=alphago_zero, mcts_zero=mcts_zero)
            win_cnt[player] += 1
        win_ratio = 1.0 * (win_cnt[0] + 0.5 * win_cnt[-1]) / game
        print("game num:{}, win: {}, lose: {}, tie:{} win_rate: {}".
              format(game, win_cnt[0], win_cnt[1], win_cnt[-1], win_ratio))

    def add_selfplay_data(self, games=1):
        for i in range(games):
            print('self play start')
            start = time.time()
            X, PI, z = self.game.self_play(self.alphago_zero)
            end = time.time()
            print('self play cost: {}'.format(end - start))
            X_all, PI_all, z_all = self.augement_data(X, PI, z)
            assert len(X_all) == len(PI_all) and len(X_all) == len(z_all), 'data dims mismatch'
            self.extend(X_all, PI_all, z_all)

    def augement_data(self, X, PI, z):
        X_all = []
        PI_all = []
        z_all = []
        X = np.transpose(X, [0, 3, 2, 1])
        for i in range(len(X)):
            for j in [1, 2, 3, 4]:
                X_tmp = np.array([np.rot90(s, j) for s in X[i]])
                PI_tmp = np.rot90(np.flipud(PI[i].reshape(self.height, self.width)), j)
                X_all.append(np.transpose(X_tmp, [1, 2, 0]))
                PI_all.append(np.flipud(PI_tmp).flatten())
                z_all.append(z[i])

                X_tmp = np.array([np.fliplr(s) for s in X_tmp])
                PI_tmp = np.fliplr(PI_tmp)
                X_all.append(np.transpose(X_tmp, [1, 2, 0]))
                PI_all.append(np.flipud(PI_tmp).flatten())
                z_all.append(z[i])
        return X_all, PI_all, z_all

    def extend(self, X, PI, z):
        for i in range(len(X)):
            self.buffer_data.append((X[i], PI[i], z[i]))

    @property
    def size(self):
        return len(self.buffer_data)
