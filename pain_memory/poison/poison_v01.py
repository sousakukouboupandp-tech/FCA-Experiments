# -*- coding: utf-8 -*-
"""
毒シミュレーション v01
やかん実験(v03-v06)と同一の6パターン・20シード・1000エピソード構造。
危険の性質のみ変更：単発の火傷(-10) → 持続ダメージの毒(踏んだ瞬間-3, その後5ステップ-1ずつ)

【重要な注意】このv01は「やかんの構造をそのまま流用しただけ」で、
持続ダメージ特有の性質（時間をかけて効く痛みが記憶方式そのものにどう作用するか）は
実質的に検証できていない、という限界が後で判明した。詳細はpoison/README.mdを参照。
発見・修正したバグ2件も含めてそのまま残す。
"""
import numpy as np
import json

np.random.seed(0)

GRID = 7
START = (0, 0)
POISON_CELL = (3, 0)      # 開始地点→解毒剤の直線経路上に配置（自然な通過点）
ANTIDOTE_CELL = (6, 0)    # 毒地帯のさらに先
N_EPISODES = 1000
SWITCH_EP = 500
MAX_STEPS = 60
LR = 0.1
GAMMA = 0.95
EPSILON = 0.2
STEP_COST = -0.05

# Pain Memory decay params (kettle実験と同一)
P_PEAK = 2.0
P_LIMIT = 0.1
THETA = 0.3
ALPHA = 0.0167

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "ACT"]  # ACT = 毒ゾーンに踏み込む/解毒剤を拾う/ゴールで完了
N_ACTIONS = len(ACTIONS)

def move(pos, a):
    x, y = pos
    if a == "UP": y = min(GRID - 1, y + 1)
    elif a == "DOWN": y = max(0, y - 1)
    elif a == "LEFT": x = max(0, x - 1)
    elif a == "RIGHT": x = min(GRID - 1, x + 1)
    return (x, y)

class PoisonEnv:
    """
    エージェントは毒ゾーン(3,3)で ACT すると+10の報酬（目的達成）を得るが、
    切替前(ep<SWITCH_EP)は「防御なしでACT」すると持続ダメージを受ける。
    防御(has_antidote)を得るには先にANTIDOTE_CELL(6,0)でACTする必要がある(+0, 遠回り)。
    切替後(ep>=SWITCH_EP)は毒が無毒化しており、防御なしでACTしても安全。
    """
    def __init__(self, recognition=False):
        self.recognition = recognition
        self.reset_episode_state()

    def reset_episode_state(self):
        self.has_antidote = False
        self.poison_dot_remaining = 0  # 持続ダメージの残りステップ数
        self.poison_status_known = None  # None=未確認, True=有効(危険), False=無毒化済み(安全) ※recognition条件のみ意味を持つ

    def reset(self, ep):
        self.pos = START
        self.reset_episode_state()
        self.ep = ep
        self.done = False
        return self.get_state()

    def poison_active(self):
        return self.ep < SWITCH_EP

    def get_state(self):
        # 基本状態: 位置 + antidote所持
        base = (self.pos, self.has_antidote)
        if self.recognition:
            # 文脈認識あり: 毒の状態(未確認/危険/安全)を状態に含める
            return (base, self.poison_status_known)
        return (base,)

    def step(self, action_idx):
        a = ACTIONS[action_idx]
        reward = STEP_COST
        info = {"poisoned_hit": False, "success": False, "observed": False}

        # 持続ダメージの処理（毎ステップ、行動に関わらず適用）
        if self.poison_dot_remaining > 0:
            reward += -1.0
            self.poison_dot_remaining -= 1

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
                        # 防御なしでACT -> 即時-3 + 持続-1×5ステップ
                        reward += -3.0
                        self.poison_dot_remaining = 5
                        info["poisoned_hit"] = True
                else:
                    # 毒は無毒化済み。防御なしでも安全にACT可能
                    reward += 10.0
                    info["success"] = True
                    self.done = True
        else:
            self.pos = move(self.pos, a)
            # 認識条件: 毒ゾーンに隣接/到達すると、触れずに状態を確認できる
            if self.recognition and self.pos == POISON_CELL:
                self.poison_status_known = self.poison_active()
                info["observed"] = True

        return self.get_state(), reward, self.done, info


def run_condition(method, recognition, seed, n_episodes=N_EPISODES, max_steps=MAX_STEPS):
    """
    method: "PermBan" | "NoMemory" | "FCA"
    """
    rng = np.random.RandomState(seed)
    env = PoisonEnv(recognition=recognition)
    Q = {}

    def get_Q(s):
        if s not in Q:
            Q[s] = np.zeros(N_ACTIONS)
        return Q[s]

    # Pain Memory: (state_key_without_time) -> pain_intensity (経路禁止のタブーリスト)
    pain_memory = {}  # key: (pos, has_antidote) -> intensity
    perm_ban = set()  # PermBan用: 一度罰を受けた(state)は永久封鎖

    rewards_per_ep = []
    poisoned_hits_pre = 0
    first_safe_ep = -1
    safe_flags_last100 = []

    def fca_key(state):
        # FCA: 経路禁止の対象は状態全体（認識ありの場合は毒の既知状態を含む）。
        # 「危険と知っての行動」と「安全と確認した行動」を別の記憶として扱う。
        return state

    def permban_key(state):
        # PermBan: 対象・行動そのものを禁止し、文脈（認識の有無）は考慮しない。
        # これがPermBanとFCAの定義上の本質的な違いである。
        return (state[0][0], state[0][1])  # (pos, has_antidote) のみ。認識情報は含めない。

    def is_tabooed(state, action_idx):
        a = ACTIONS[action_idx]
        if a != "ACT":
            return False
        if state[0][0] != POISON_CELL:
            return False
        if method == "PermBan":
            return permban_key(state) in perm_ban
        elif method == "FCA":
            return pain_memory.get(fca_key(state), 0.0) > THETA
        return False  # NoMemory

    for ep in range(n_episodes):
        state = env.reset(ep)
        total_r = 0.0
        episode_poisoned = False
        episode_safe_act = False

        for t in range(max_steps):
            s = state
            qvals = get_Q(s).copy()
            # タブー行動をマスク（epsilon-greedy選択時に除外）
            valid_actions = list(range(N_ACTIONS))
            tabooed = [i for i in valid_actions if is_tabooed(s, i)]
            if len(tabooed) < N_ACTIONS:
                for i in tabooed:
                    qvals[i] = -np.inf

            if rng.rand() < EPSILON:
                choices = [i for i in valid_actions if i not in tabooed] or valid_actions
                a_idx = rng.choice(choices)
            else:
                a_idx = int(np.argmax(qvals))

            next_state, r, done, info = env.step(a_idx)
            total_r += r

            # Q学習更新
            best_next = np.max(get_Q(next_state)) if not done else 0.0
            td_target = r + GAMMA * best_next
            get_Q(s)[a_idx] += LR * (td_target - get_Q(s)[a_idx])

            if info["poisoned_hit"]:
                episode_poisoned = True
                intensity = P_PEAK
                if method == "FCA":
                    key = fca_key(s)
                    pain_memory[key] = pain_memory.get(key, 0.0) + intensity
                elif method == "PermBan":
                    perm_ban.add(permban_key(s))
                elif method == "NoMemory":
                    pass  # 記憶しない

            if info["success"]:
                if not env.has_antidote and not env.poison_active():
                    episode_safe_act = True

            state = next_state
            if done:
                break

        # 痛みの減衰（エピソード単位、§4.5の指数減衰を簡略適用）
        for k in list(pain_memory.keys()):
            pain_memory[k] *= np.exp(-ALPHA)
            if pain_memory[k] < P_LIMIT:
                del pain_memory[k]

        rewards_per_ep.append(total_r)
        if ep < SWITCH_EP and episode_poisoned:
            poisoned_hits_pre += 1
        if ep >= SWITCH_EP and episode_safe_act and first_safe_ep < 0:
            first_safe_ep = ep - SWITCH_EP
        if ep >= n_episodes - 100:
            safe_flags_last100.append(1.0 if episode_safe_act else 0.0)

    rewards_per_ep = np.array(rewards_per_ep)
    return {
        "rew_pre": float(np.mean(rewards_per_ep[:SWITCH_EP])),
        "rew_post": float(np.mean(rewards_per_ep[SWITCH_EP:])),
        "rew_post_first50": float(np.mean(rewards_per_ep[SWITCH_EP:SWITCH_EP+50])),
        "rew_last100": float(np.mean(rewards_per_ep[-100:])),
        "poisoned_hits_pre": int(poisoned_hits_pre),
        "safe_last100": float(np.mean(safe_flags_last100)) if safe_flags_last100 else 0.0,
        "first_safe_ep": int(first_safe_ep),
    }

print("PoisonEnvクラス定義・run_condition関数定義 完了")
