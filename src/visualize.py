import pygame
from src import config
import pickle
import os
import time
import sys

"""
    可视化整个训练过程 网络学习情况
"""

pygame.init()
pygame.display.set_caption('五子棋')  # 改标题
screen = pygame.display.set_mode((640, 640))
# 给窗口填充颜色，颜色用三原色数字列表表示
screen.fill([125, 95, 24])

width = config.default_width
height = config.default_height
white = [255, 255, 255]
black = [0, 0, 0]


def load_records(path):
    if not os.path.exists(path):
        return None
    return pickle.load(open(path, 'rb'))


records = load_records(config.record_path)


def draw(data=None):
    for h in range(width):
        pygame.draw.line(screen, black, [40, 40 + h * 70], [600, 40 + h * 70], 1)

        pygame.draw.line(screen, black, [40 + h * 70, 40], [40 + h * 70, 600], 1)
        # 在棋盘上标出，天元以及另外4个特殊点位
        pygame.draw.circle(screen, black, [320, 320], 5, 0)
        pygame.draw.circle(screen, black, [180, 180], 3, 0)
        pygame.draw.circle(screen, black, [180, 460], 3, 0)
        pygame.draw.circle(screen, black, [460, 180], 3, 0)
        pygame.draw.circle(screen, black, [460, 460], 3, 0)
    if data:
        for action, player in data.items():
            color = black if player == 0 else white
            h, w = action // width, action % width
            pos = [40 + 70 * w + 1, 40 + 70 * h]
            # 画出棋子
            pygame.draw.circle(screen, color, pos, 22, 0)
    pygame.display.flip()  # 刷新窗口显示


draw()
while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            sys.exit()
    if records:
        for record in records:
            record_tmp = {}
            for action, value in record.items():
                record_tmp[action] = value
                draw(record_tmp)
                time.sleep(1)
                # 清空棋盘
            screen.fill([125, 95, 24])
            time.sleep(3)
        break
