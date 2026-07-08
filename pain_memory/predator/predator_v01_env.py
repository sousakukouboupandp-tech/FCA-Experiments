# -*- coding: utf-8 -*-
"""
捕食者実験 v01
4AI（ChatGPT/Gemini/Grok/ディープリサーチ）相談の統合設計に基づく。

- 環境変化(穴1): 捕食者のパトロールルートが、エピソード500を境にA→Bへ予告なく切り替わる
- 経路記憶の対応(穴2): 状態に捕食者の位置+直近の移動方向(位相相当)を含める
- 認識なし(穴3): 局所情報のみ(自分と同じマスにいるかどうかだけ分かる)。位置・方向は分からない
- 共鳴トラップ(穴4): このv01には含めない。別の追加実験として分離する
"""
import numpy as np

GRID = 7
START = (0, 0)
GOAL = (6, 6)
N_EPISODES = 1000
SWITCH_EP = 500
MAX_STEPS = 80
LR = 0.1
GAMMA = 0.95
EPSILON = 0.2
STEP_COST = -0.05
COLLISION_PENALTY = -10.0
GOAL_REWARD = 10.0

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
N_ACTIONS = len(ACTIONS)

def move(pos, a):
    x, y = pos
    if a == "UP": y = min(GRID - 1, y + 1)
    elif a == "DOWN": y = max(0, y - 1)
    elif a == "LEFT": x = max(0, x - 1)
    elif a == "RIGHT": x = min(GRID - 1, x + 1)
    return (x, y)

# パトロールルート: 縦の列を上下往復する
ROUTE_A_X = 2
ROUTE_B_X = 4
def build_patrol(x_col):
    ys = list(range(1, 6)) + list(range(5, 0, -1))  # 1->5->1 往復
    return [(x_col, y) for y in ys]

PATROL_A = build_patrol(ROUTE_A_X)
PATROL_B = build_patrol(ROUTE_B_X)

class PredatorEnv:
    def __init__(self, recognition=False):
        self.recognition = recognition

    def reset(self, ep):
        self.pos = START
        self.ep = ep
        self.t = 0
        self.done = False
        self.route_active = PATROL_A if ep < SWITCH_EP else PATROL_B
        self.pred_idx = 0
        self.pred_pos = self.route_active[0]
        self.pred_prev = self.pred_pos
        return self.get_state()

    def get_state(self):
        if self.recognition:
            dx = self.pred_pos[0] - self.pred_prev[0]
            dy = self.pred_pos[1] - self.pred_prev[1]
            return (self.pos, self.pred_pos, (dx, dy))
        else:
            # 認識なし: 自分と同じマスにいるかどうかだけ分かる(局所情報)
            same = (self.pos == self.pred_pos)
            return (self.pos, same)

    def step(self, action_idx):
        a = ACTIONS[action_idx]
        reward = STEP_COST
        info = {"collided": False, "success": False}

        self.pos = move(self.pos, a)

        # 捕食者移動(パトロール, 1歩/ステップ)
        self.pred_prev = self.pred_pos
        self.pred_idx = (self.pred_idx + 1) % len(self.route_active)
        self.pred_pos = self.route_active[self.pred_idx]

        if self.pos == self.pred_pos:
            reward += COLLISION_PENALTY
            info["collided"] = True
        if self.pos == GOAL:
            reward += GOAL_REWARD
            info["success"] = True
            self.done = True

        self.t += 1
        if self.t >= MAX_STEPS:
            self.done = True

        return self.get_state(), reward, self.done, info

print("PredatorEnv定義完了")
