# -*- coding: utf-8 -*-
"""
捕食者実験 v03 — 「関係的な痛み」の実装

思想の核(2026-07-08 始祖の定義):
  捕食者の痛みは「捕食者との相対的な関係」によって決まる。
  - 距離が縮まっているとき、痛みが立ち上がる(迫られる恐怖)
  - 距離を保てている/開けているとき、痛みは止まる
  - スピード・位置関係・距離の変化の方向、全てがこの一つの概念に含まれる
  - 接触=死 で痛みが最大かつ終端

v02までとの決定的な違い:
  v02は接触した瞬間だけ-10という「絶対的・点の痛み」だった。
  v03は捕食者との関係が刻々と生む「関係的・連続の痛み」。接触しなくても
  死に近づくこと自体が痛い。痛みは対象ではなく「関係」に属する。

痛みの数式(翻訳):
  各ステップの痛み = 近接項 + 接近項
    近接項 = W_prox * f(d)      … 今どれだけ近いか。d小さいほど大。
    接近項 = W_approach * max(0, closing_rate) * g(d)
             … どれだけ速く近づかれているか。距離が縮まっている(closing_rate>0)
                ときだけ立ち上がる。開いている(closing_rate<0)ときは0=止まる。
                g(d)で「近いほど接近の恐怖が跳ね上がる」を表現。
  接触(d=0) → COLLISION_PENALTY(死), エピソード終端。

環境:
  まずシンプルに。開けた空間(壁・ゲートなし)+動く捕食者+関係的な痛みの場。
  幾何学の複雑さは後で足す(何が効いたか分からなくなるのを防ぐ)。
"""
import numpy as np

GRID = 9              # 開けた空間。関係的な痛みを見るため少し広め
START = (0, 4)
GOAL = (8, 4)
N_EPISODES = 1000
SWITCH_EP = 500
MAX_STEPS = 80
LR = 0.1
GAMMA = 0.95
EPSILON = 0.2
STEP_COST = -0.05
COLLISION_PENALTY = -15.0   # 接触=死。関係痛の積分より十分大きく
GOAL_REWARD = 10.0

# 関係的な痛みのパラメータ(細部は診断しながら調整予定)
W_PROX = 0.15        # 近接項の重み(今の近さ由来の警戒痛)
W_APPROACH = 0.6     # 接近項の重み(距離が縮まる恐怖)。関係痛の主役
PAIN_RANGE = 4       # この距離より遠いと痛みほぼ0(逃走開始距離に相当)

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
N_ACTIONS = len(ACTIONS)

def move(pos, a):
    x, y = pos
    if a == "UP": y = min(GRID-1, y+1)
    elif a == "DOWN": y = max(0, y-1)
    elif a == "LEFT": x = max(0, x-1)
    elif a == "RIGHT": x = min(GRID-1, x+1)
    # STAY は動かない
    return (x, y)

def chebyshev(a, b):
    # 相対的な距離(8方向の最短歩数)。捕食者との「関係」の基本量
    return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

def build_patrol(y_lo, y_hi, x_col):
    ys = list(range(y_lo, y_hi+1)) + list(range(y_hi-1, y_lo, -1))
    return [(x_col, y) for y in ys]

# Phase A: 捕食者が経路の中央(x=4)付近を縦にパトロール
# Phase B: 捕食者が上方へ退避し、中央ルートの関係が safe になる
PATROL_A = build_patrol(2, 6, 4)
PATROL_B = build_patrol(7, 8, 4)

class PredatorEnvV3:
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
        self.prev_dist = chebyshev(self.pos, self.pred)
        return self.get_state()

    def get_state(self):
        d = chebyshev(self.pos, self.pred)
        # 関係を状態に圧縮: 相対位置(符号付き)と距離帯、接近中か否か
        rel_x = int(np.sign(self.pred[0] - self.pos[0]))
        rel_y = int(np.sign(self.pred[1] - self.pos[1]))
        dist_band = min(d, PAIN_RANGE)  # 0..PAIN_RANGE でクリップ
        if self.recognition:
            closing = 1 if d < self.prev_dist else 0  # 距離が縮まっているか
            return (self.pos, rel_x, rel_y, dist_band, closing)
        else:
            # 認識なし: 捕食者が隣接(危険圏)にいるかだけ
            adj = 1 if d <= 1 else 0
            return (self.pos, adj)

    def relational_pain(self, d_before, d_after):
        """捕食者との関係が生む痛み(接触前の連続痛)"""
        if d_after > PAIN_RANGE:
            return 0.0  # 逃走開始距離の外=痛みなし
        # 近接項: 今の近さ。近いほど大きい(0..1)
        prox = (PAIN_RANGE - d_after) / PAIN_RANGE
        # 接近項: 距離が縮まっているときだけ。近いほど跳ね上がる
        closing_rate = d_before - d_after   # +なら近づかれた
        approach = max(0, closing_rate) * ((PAIN_RANGE - d_after) / PAIN_RANGE)
        pain = W_PROX * prox + W_APPROACH * approach
        return -pain  # 痛みは負の報酬

    def step(self, action_idx):
        a = ACTIONS[action_idx]
        reward = STEP_COST
        info = {"collided": False, "success": False, "pain": 0.0}

        d_before = chebyshev(self.pos, self.pred)
        new_pos = move(self.pos, a)

        # 捕食者移動
        self.pred_prev = self.pred
        self.pidx = (self.pidx + 1) % len(self.route)
        self.pred = self.route[self.pidx]

        d_after = chebyshev(new_pos, self.pred)

        # 接触=死
        if d_after == 0 or (new_pos == self.pred_prev and self.pos == self.pred):
            reward += COLLISION_PENALTY
            info["collided"] = True
            new_pos = self.pos
            self.done = True
        else:
            # 関係的な痛み(接触前の連続痛)
            p = self.relational_pain(d_before, d_after)
            reward += p
            info["pain"] = p

        self.pos = new_pos
        self.prev_dist = d_after

        if not self.done and self.pos == GOAL:
            reward += GOAL_REWARD
            info["success"] = True
            self.done = True

        self.t += 1
        if self.t >= MAX_STEPS:
            self.done = True
        return self.get_state(), reward, self.done, info

print("PredatorEnvV3定義完了(関係的な痛み)")
