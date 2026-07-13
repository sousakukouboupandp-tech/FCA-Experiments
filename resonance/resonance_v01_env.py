# -*- coding: utf-8 -*-
"""
共鳴実験 v01 環境 — predator_v02の地形を継承した二世界版
【設計書】resonance/DESIGN_共鳴実験_v01.md
- 世界A（語り手の世界）: 捕食者がgap_mid入口(x=2, y2-4)を守る。静的。
- 世界B（聞き手の世界）: 危険の配置をΔでずらす。
  Δ0 : Aと同一 (x=2, y2-4)
  Δ小: 1マスずれ (x=2, y3-5)
  Δ大: 逆転 (x=2, y5-6) ＝旧安全路gap_top側が危険に、gap_mid入口が安全に
       → Aの痛み記憶がBには「迷信」になる
【v02からの変更・書記官の自律決定】フェーズ切替(SWITCH_EP)を廃止し静的世界とした。
  理由: 本実験の主題は時間変化への適応ではなく、個体間の記憶の受け渡しであるため。
"""
import numpy as np

GRID = 7
START = (0, 3)
GOAL = (6, 3)
MAX_STEPS = 60
LR = 0.1
GAMMA = 0.95
EPSILON = 0.2
STEP_COST = -0.05
COLLISION_PENALTY = -10.0
GOAL_REWARD = 10.0

WALLS = set()
for y in range(GRID):
    if y not in (3, 6):
        WALLS.add((3, y))

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
N_ACTIONS = len(ACTIONS)

def move(pos, a):
    x, y = pos
    if a == "UP": x2, y2 = x, min(GRID-1, y+1)
    elif a == "DOWN": x2, y2 = x, max(0, y-1)
    elif a == "LEFT": x2, y2 = max(0, x-1), y
    else: x2, y2 = min(GRID-1, x+1), y
    if (x2, y2) in WALLS:
        return pos
    return (x2, y2)

def build_patrol(y_lo, y_hi):
    ys = list(range(y_lo, y_hi+1)) + list(range(y_hi-1, y_lo, -1))
    return [(2, y) for y in ys]

PATROL_WORLD_A  = build_patrol(2, 4)  # 語り手Aの危険: gap_mid入口
PATROL_B_D0     = build_patrol(2, 4)  # Δ0 : 同一
PATROL_B_DSMALL = build_patrol(3, 5)  # Δ小: 1マスずれ
PATROL_B_DBIG   = build_patrol(5, 6)  # Δ大: 逆転(gap_top側が危険)

class ResonanceEnv:
    """認識なし方式固定: state=(pos, adj)。v02で最も綺麗に差が出た土俵。"""
    def __init__(self, patrol):
        self.patrol = patrol

    def reset(self, ep):
        self.pos = START
        self.t = 0
        self.done = False
        self.pidx = 0
        self.pred = self.patrol[0]
        self.pred_prev = self.pred
        return self.get_state()

    def get_state(self):
        adj = abs(self.pos[0]-self.pred[0]) + abs(self.pos[1]-self.pred[1]) <= 1
        return (self.pos, adj)

    def step(self, action_idx):
        a = ACTIONS[action_idx]
        reward = STEP_COST
        info = {"collided": False, "success": False, "collided_at": None}
        new_pos = move(self.pos, a)
        self.pred_prev = self.pred
        self.pidx = (self.pidx + 1) % len(self.patrol)
        self.pred = self.patrol[self.pidx]
        collided = (new_pos == self.pred) or (new_pos == self.pred_prev and self.pos == self.pred)
        if collided:
            reward += COLLISION_PENALTY
            info["collided"] = True
            info["collided_at"] = new_pos
            new_pos = self.pos
        self.pos = new_pos
        if self.pos == GOAL:
            reward += GOAL_REWARD
            info["success"] = True
            self.done = True
        self.t += 1
        if self.t >= MAX_STEPS:
            self.done = True
        return self.get_state(), reward, self.done, info

print("ResonanceEnv定義完了(静的世界・認識なし方式)")
