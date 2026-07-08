# -*- coding: utf-8 -*-
"""
捕食者実験 v02 — v01の3失敗を修正した再設計（最終版）

【v01の3つの失敗と、v02での修正】
1. 危険が機能しない（パトロール列が経路を塞がず、前半で衝突15回のみ）
   → 壁で「短い危険路 vs 長い安全路」を作り、危険を避けられない構造にした（前半衝突30〜75回）
2. 状態爆発（捕食者の絶対座標(x,y)を状態に入れ、960状態に膨張、学習不能）
   → 捕食者情報を「行(row)＋移動方向」だけに圧縮（絶対x座標は常にx=2なので捨てる）。257状態に収まった
3. ボトルネックなし（迂回し放題でPermBanが詰まない）
   → 2ゲート構造で制約

【設計を確定させるまでの試行錯誤（記録として残す）】
- 第1版: 捕食者を通り道(x=2)、Phase Bでy=5,6へ移動 → PermBanが両ゲートの通り道を
  全て禁止して完全に詰んだ(ゴール到達0%=比較にならない)
- 第2版: 捕食者を出口側(x=4)へ → gap_midの遠い出口だけ禁止され、PermBanが
  gap_midの入口までは学習して袋小路にはまった
- 最終版: 捕食者を入口側(x=2)、かつgap_top(y=6経由)を捕食者が絶対に触れない
  「常時安全な逃げ場」にした。これでやかん実験の「ミトン＝常に安全な道」と同じ構造になった

【地形】
  start(0,3), goal(6,3)。x=3が壁、y=3(gap_mid)とy=6(gap_top)のみ通行可
  gap_mid経由=6歩(最短、だが入口の危険地帯を通る)
  gap_top経由=12歩(遠回り、常に安全＝やかんのミトンに相当)

【捕食者】x=2列(gap_midの入口)を上下パトロール
  Phase A(ep<500): y∈{2,3,4} → gap_mid入口が危険 → エージェントはgap_topへ迂回
  Phase B(ep>=500): y∈{0,1}へ退避 → gap_mid入口が安全に(gap_topは常に安全のまま)
  = 以前危険だった最短路が環境変化で安全になる(やかんの火が消えるに相当)
"""
import numpy as np

GRID = 7
START = (0, 3)
GOAL = (6, 3)
N_EPISODES = 1000
SWITCH_EP = 500
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
    elif a == "RIGHT": x2, y2 = min(GRID-1, x+1), y
    if (x2, y2) in WALLS:
        return pos
    return (x2, y2)

def build_patrol(y_lo, y_hi):
    ys = list(range(y_lo, y_hi+1)) + list(range(y_hi-1, y_lo, -1))
    return [(2, y) for y in ys]

PATROL_A = build_patrol(2, 4)   # Phase A: gap_mid入口を守る
PATROL_B = build_patrol(0, 1)   # Phase B: 下へ退避、gap_mid入口が安全化

class PredatorEnvV2:
    def __init__(self, recognition=False):
        self.recognition = recognition

    def reset(self, ep):
        self.pos = START
        self.ep = ep
        self.t = 0
        self.done = False
        self.route = PATROL_A if ep < SWITCH_EP else PATROL_B
        self.pidx = 0
        self.pred = self.route[0]
        self.pred_prev = self.pred
        return self.get_state()

    def get_state(self):
        if self.recognition:
            # 圧縮: 捕食者の行と移動方向だけ(絶対x座標は常にx=2なので捨てる)
            prow = self.pred[1]
            pdir = int(np.sign(self.pred[1] - self.pred_prev[1]))
            return (self.pos, prow, pdir)
        else:
            # 認識なし: 捕食者が隣接マスにいるか(局所情報)だけ
            adj = abs(self.pos[0]-self.pred[0]) + abs(self.pos[1]-self.pred[1]) <= 1
            return (self.pos, adj)

    def step(self, action_idx):
        a = ACTIONS[action_idx]
        reward = STEP_COST
        info = {"collided": False, "success": False}

        new_pos = move(self.pos, a)

        self.pred_prev = self.pred
        self.pidx = (self.pidx + 1) % len(self.route)
        self.pred = self.route[self.pidx]

        # 衝突: 移動後に同じマス、または位置交換(すれ違い)
        collided = (new_pos == self.pred) or (new_pos == self.pred_prev and self.pos == self.pred)
        if collided:
            reward += COLLISION_PENALTY
            info["collided"] = True
            new_pos = self.pos  # 阻まれてその場に留まる(死亡しない=やかんの「痛いが継続」に相当)

        self.pos = new_pos
        if self.pos == GOAL:
            reward += GOAL_REWARD
            info["success"] = True
            self.done = True

        self.t += 1
        if self.t >= MAX_STEPS:
            self.done = True
        return self.get_state(), reward, self.done, info

print("PredatorEnvV2定義完了(捕食者=x=2入口, gap_top常時安全)")
