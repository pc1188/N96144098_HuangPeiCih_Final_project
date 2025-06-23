import pygame
import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
from setting import *
from game import Game

# ----------- 模型定義 -----------
class DQN(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ----------- 圖像前處理 -----------
def preprocess(image):
    image = np.transpose(image, (1, 0, 2))  # HWC -> WHC
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (84, 84))
    return np.expand_dims(image, axis=0) / 255.0  # shape: [1, 84, 84]

# ----------- 初始化環境 -----------
pygame.init()
pygame.font.init()

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SpaceShip RL Inference")
clock = pygame.time.Clock()
game = Game()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------- 載入模型 -----------
action_dim = 4
policy_net = DQN(action_dim).to(device)
policy_net.load_state_dict(torch.load("best_dqn_space_ship.pth", map_location=device))
policy_net.eval()

# ----------- 推論開始 -----------
running = True
state = preprocess(game.state)
total_reward = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or game.score >= 10000:
            running = False

    if game.running:
        # 模型選擇 action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # [1, 1, 84, 84]
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            action = torch.argmax(q_values).item()

        game.update(action)
        state = preprocess(game.state)
        total_reward = game.score

        # 繪製畫面
        game.draw(screen)
        pygame.display.update()
        clock.tick(FPS)

print(f"模型最終得分：{total_reward}")
pygame.quit()
