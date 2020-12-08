import argparse
from importlib import import_module
from src.board.gobang import GoBang
from src.board.game import Game
from src.mcts.mcts import MCTS
from src.mcts.alphago_zero import AlphaGoZero
from src.pipeline import PipelineTrain

parser = argparse.ArgumentParser(description='AlphaGo Zero to study Gobang')
parser.add_argument('--model', default='cnn', type=str,
                    help='choose an models alexnet, restnet, vgg, cnn eg: only cnn can be used')
parser.add_argument('--weights_path', default='../model/alphago_zero_best.h5', type=str,
                    help='train network weights path')
parser.add_argument('--epoch', type=int, default=2000, help='epoch num')
args = parser.parse_args()

if __name__ == '__main__':
    model_name = args.model
    weights_path = args.weights_path
    epoch = args.epoch
    x = import_module('models.' + model_name)
    model = x.Model()
    board = GoBang()
    game = Game(board, show=True)
    mcts = MCTS(model)
    alpahgo_zero = AlphaGoZero(mcts, self_play=True)
    pipelinetrain = PipelineTrain(game, alpahgo_zero, model, epoch)
    pipelinetrain.train()
