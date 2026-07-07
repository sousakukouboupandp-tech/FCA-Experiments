# -*- coding: utf-8 -*-
"""
毒シミュレーション v2: 薬物動態モデル(PK)による痛みの山型波形を使用。
geometry修正済み(v01の教訓): POISON_CELLを START->ANTIDOTE の経路上に配置。
"""
import numpy as np

GRID = 7
START = (0, 0)
POISON_CELL = (3, 0)
ANTIDOTE_CELL = (6, 0)
N_EPISODES = 1000
SWITCH_EP = 500
MAX_STEPS = 60
LR = 0.1
GAMMA = 0.95
EPSILON = 0.2
STEP_COST = -0.05

LAMBDA_GLOBAL = 0.965   # エピソード間の長期減衰(exp(-ALPHA)相当。ALPHA=0.0167 -> exp(-0.0167)=0.9834に近い水準だが要検証)
GAMMA_LOCAL = 0.7       # エピソード内EMA蓄積の割引
THETA = 0.3             # FCA固定閾値方式のタブー判定線
P_LIMIT = 0.05

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "ACT"]
N_ACTIONS = len(ACTIONS)

def move(pos, a):
    x, y = pos
    if a == "UP": y = min(GRID - 1, y + 1)
    elif a == "DOWN": y = max(0, y - 1)
    elif a == "LEFT": x = max(0, x - 1)
    elif a == "RIGHT": x = min(GRID - 1, x + 1)
    return (x, y)

def pk_pain_value(t, ka, kel, scale):
    if abs(ka - kel) < 1e-6:
        kel = ka * 1.0001
    c = scale * (ka / (ka - kel)) * (np.exp(-kel * t) - np.exp(-ka * t))
    return -abs(c)

PK_PARAMS = {"ka": 0.4, "kel": 0.15, "scale": 1.537}  # 「毒型」水準(総ダメージ-10相当)

class PoisonEnvV2:
    def __init__(self, recognition=False):
        self.recognition = recognition

    def reset(self, ep):
        self.pos = START
        self.has_antidote = False
        self.dot_timer = None  # 毒を受けた時点からの経過ステップ数。Noneなら効果なし。
        self.poison_status_known = None
        self.ep = ep
        self.done = False
        return self.get_state()

    def poison_active(self):
        return self.ep < SWITCH_EP

    def get_state(self):
        base = (self.pos, self.has_antidote)
        if self.recognition:
            return (base, self.poison_status_known)
        return (base,)

    def step(self, action_idx):
        a = ACTIONS[action_idx]
        reward = STEP_COST
        info = {"poisoned_hit": False, "success": False, "pain_now": 0.0}

        # 持続する毒ダメージ(PK波形)の適用
        if self.dot_timer is not None:
            p = pk_pain_value(self.dot_timer, **PK_PARAMS)
            reward += p
            info["pain_now"] = p
            self.dot_timer += 1
            if self.dot_timer > 20:  # 十分減衰したら打ち切り
                self.dot_timer = None

        if a == "ACT":
            if self.pos == ANTIDOTE_CELL:
                self.has_antidote = True
            elif self.pos == POISON_CELL:
                if self.poison_active():
                    if self.has_antidote:
                        reward += 10.0
                        info["success"] = True
                        self.done = True
                    else:
                        self.dot_timer = 0  # 毒発動、t=0から山が始まる
                        info["poisoned_hit"] = True
                else:
                    reward += 10.0
                    info["success"] = True
                    self.done = True
        else:
            self.pos = move(self.pos, a)
            if self.recognition and self.pos == POISON_CELL:
                self.poison_status_known = self.poison_active()

        return self.get_state(), reward, self.done, info

print("PoisonEnvV2定義完了（PK波形版）")
