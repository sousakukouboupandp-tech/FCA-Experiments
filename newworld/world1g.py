# -*- coding: utf-8 -*-
"""World_1-G 正本エンジン（1歩MDP）
設計書：DESIGN_新世界_v02_World1G_表1と再現仕様.md（凍結コミット 386bf76）

方針（ChatGPT との設計会議 返答30〜32）：
  ・transition は純関数。state + choice + draws -> next_state
  ・抽選は sample_draws に分離する。テストでは draws を直接渡せる
  ・体力は原本の ×20 の整数表現（近似ではなく等価）
"""

from dataclasses import dataclass, replace

# ---- 世界の定数（v02 §1）--------------------------------------------------
E_MAX = 2000          # 体力の上限（原本 100）
UPKEEP = 20           # 基礎消耗 /歩（原本 1.0）
WOUND_COST = 5        # 傷1つの追加消耗 /歩（原本 0.25）
GRASS_GAIN = 14       # 草の摂取 /歩（原本 0.7）
SMALL_GAIN = 600      # 小物の栄養（原本 30）
LARGE_GAIN = 1800     # 大物の栄養（原本 90）

HORIZON = 1000        # 寿命（歩）

SMALL_INIT = 12       # 小物の初期の残り具合
SMALL_FLOOR = 1       # 1 で止まる（絶滅しない残存層）
SMALL_SEARCH_K = 24.0 # 平均探索歩数 = 24 / N_S
CHASE_STEPS = 4       # 追跡の歩数
CATCH_P = 0.7         # 追跡4歩目の捕獲確率

LARGE_INIT = 3
LARGE_MAX = 3
LARGE_BASE_SEARCH = 8.0   # N_L=3 のときの平均探索歩数
LARGE_SEARCH_EXP = 1.5    # k。基準 1.5（感度分析 2.0）
LARGE_RECOVER_P = 1.0 / 250.0   # N_L が 1 または 2 のとき、1歩あたり1頭戻る確率

HEAL_P = 1.0 / 150.0      # 傷1つが1歩で治る確率

# 戦闘の排他的な4結果（v02 §1-7）
COMBAT_KILL = 0.25
COMBAT_ACUTE = 0.04
COMBAT_WOUND = 0.36
COMBAT_MISS = 0.35

# MODE
DECISION = "decision"        # 意思決定の境界。時間を使わない
GRASS = "grass"
SEARCH_SMALL = "search_small"
CHASE_1 = "chase_1"
CHASE_2 = "chase_2"
CHASE_3 = "chase_3"
CHASE_4 = "chase_4"
SEARCH_LARGE = "search_large"
COMBAT = "combat"

CHASE_NEXT = {CHASE_1: CHASE_2, CHASE_2: CHASE_3, CHASE_3: CHASE_4}

# 行動（decision 境界で選ぶ）
ACT_GRASS = "grass"
ACT_SMALL = "small"
ACT_LARGE = "large"

# 終端の理由
ALIVE = None
DEAD_STARVE = "starve"
DEAD_ACUTE = "acute"
DEAD_TENJU = "tenju"


def small_search_p(n_small):
    """小物の1歩あたりの遭遇確率。平均 24/N_S 歩"""
    return n_small / SMALL_SEARCH_K


def large_search_steps(n_large, k=LARGE_SEARCH_EXP):
    """大物の平均探索歩数 8*(3/N)^k"""
    if n_large <= 0:
        return float("inf")
    return LARGE_BASE_SEARCH * (LARGE_MAX / n_large) ** k


def large_search_p(n_large, k=LARGE_SEARCH_EXP):
    t = large_search_steps(n_large, k)
    return 0.0 if t == float("inf") else 1.0 / t


# ---- 状態と抽選結果 -------------------------------------------------------
@dataclass(frozen=True)
class State:
    e: int = E_MAX            # 体力（×20 表現）
    w: int = 0                # 傷の数
    n_small: int = SMALL_INIT
    n_large: int = LARGE_INIT
    t: int = 0                # 経過歩数
    mode: str = DECISION
    dead: object = ALIVE      # ALIVE か終端の理由

    @property
    def alive(self):
        return self.dead is ALIVE

    @property
    def at_decision(self):
        return self.mode == DECISION and self.alive


@dataclass(frozen=True)
class Draws:
    """その歩の抽選結果。MODE ごとに使う項目だけを持つ。
    heal は「その歩に治る傷の数」（歩の始めの W 個から、それぞれ独立に 1/150）。
    recover は「その歩に大物が1頭戻るか」（N_L が 1 か 2 のときだけ引く）。
    find  は探索で遭遇したか。catch は追跡4歩目の捕獲。
    combat は 'kill' / 'acute' / 'wound' / 'miss' のいずれか。
    """
    heal: int = 0
    recover: bool = False
    find: bool = False
    catch: bool = False
    combat: str = None


def legal_actions(s):
    """decision 境界で選べる行動。大物は1頭以上いるときだけ"""
    if not s.at_decision:
        return ()
    acts = [ACT_GRASS, ACT_SMALL]
    if s.n_large >= 1:
        acts.append(ACT_LARGE)
    return tuple(acts)


def start_mode(action):
    """選んだ行動の1歩目の MODE。decision 自体は時間を使わない"""
    if action == ACT_GRASS:
        return GRASS
    if action == ACT_SMALL:
        return SEARCH_SMALL
    if action == ACT_LARGE:
        return SEARCH_LARGE
    raise ValueError("未知の行動: %r" % (action,))


def transition(s, action, d):
    """1歩進める純関数。
    s: 歩の始めの状態（s.mode が DECISION なら action を使い、その歩を行動の1歩目とする）
    action: decision 境界のときだけ使う。それ以外では None を要求する
    d: Draws（その歩の抽選結果）
    戻り値: 次の状態
    """
    if not s.alive:
        raise ValueError("終端状態からは遷移しない")

    if s.mode == DECISION:
        if action not in legal_actions(s):
            raise ValueError("その状態で選べない行動: %r" % (action,))
        mode = start_mode(action)
    else:
        if action is not None:
            raise ValueError("decision 境界以外では行動を選べない")
        mode = s.mode

    # --- その歩の消耗は「歩の始めの傷の数」で決まる（v02 §1-10）
    cost = UPKEEP + WOUND_COST * s.w
    intake = 0

    # --- 傷の治癒：歩の始めにある W 個が対象。新しい傷は次の歩から
    if not (0 <= d.heal <= s.w):
        raise ValueError("治癒数が歩の始めの傷の数を超えている")
    w_next = s.w - d.heal

    # --- 大物の自然回復：N_L が 1 か 2 のときだけ
    recover = 1 if (d.recover and s.n_large in (1, 2)) else 0
    if d.recover and s.n_large not in (1, 2):
        raise ValueError("回復の抽選は N_L が 1 か 2 のときだけ")

    n_small_next = s.n_small
    n_large_next = s.n_large
    kill = 0
    next_mode = mode
    acute = False

    if mode == GRASS:
        intake = GRASS_GAIN
        next_mode = DECISION

    elif mode == SEARCH_SMALL:
        next_mode = CHASE_1 if d.find else SEARCH_SMALL

    elif mode in (CHASE_1, CHASE_2, CHASE_3):
        next_mode = CHASE_NEXT[mode]

    elif mode == CHASE_4:
        if d.catch:
            intake = SMALL_GAIN
            n_small_next = max(SMALL_FLOOR, s.n_small - 1)
        next_mode = DECISION

    elif mode == SEARCH_LARGE:
        if s.n_large < 1:
            raise ValueError("大物がいないのに大物探索はできない")
        next_mode = COMBAT if d.find else SEARCH_LARGE

    elif mode == COMBAT:
        if d.combat == "kill":
            intake = LARGE_GAIN
            kill = 1
            next_mode = DECISION
        elif d.combat == "acute":
            acute = True
            next_mode = COMBAT
        elif d.combat == "wound":
            w_next += 1
            next_mode = COMBAT
        elif d.combat == "miss":
            next_mode = COMBAT
        else:
            raise ValueError("戦闘の抽選結果が不正: %r" % (d.combat,))

    else:
        raise ValueError("未知の MODE: %r" % (mode,))

    # --- 大物の頭数：仕留めで 0 になる歩は、回復を無効にして絶滅を優先（v02 §1-10）
    if kill and s.n_large == 1:
        n_large_next = 0
    else:
        n_large_next = min(LARGE_MAX, s.n_large + recover - kill)

    # --- 体力：その歩の全収支を反映してから飢え死を判定
    e_next = min(E_MAX, s.e - cost + intake)
    t_next = s.t + 1

    # --- 終端判定。急所死は体力に関係なく死
    if acute:
        dead = DEAD_ACUTE
    elif e_next <= 0:
        dead = DEAD_STARVE
    else:
        dead = ALIVE

    if dead is ALIVE and t_next >= HORIZON:
        dead = DEAD_TENJU   # 1000歩目の遷移を生き抜いた場合だけ天寿

    return replace(s, e=e_next, w=w_next, n_small=n_small_next,
                   n_large=n_large_next, t=t_next,
                   mode=(next_mode if dead is ALIVE else s.mode), dead=dead)


# ---- 抽選（乱数を使う側。遷移関数からは分離）-----------------------------
COMBAT_OUTCOMES = ("kill", "acute", "wound", "miss")
COMBAT_P = (COMBAT_KILL, COMBAT_ACUTE, COMBAT_WOUND, COMBAT_MISS)


def sample_draws(s, action, rng, k=LARGE_SEARCH_EXP):
    """その歩の抽選。s.mode が DECISION のときは action で1歩目の MODE を決める。
    同じ歩の別系統の出来事は、歩の始めの状態のもとで互いに独立に引く（v02 §1-10）。
    """
    mode = start_mode(action) if s.mode == DECISION else s.mode

    # 背景：傷の治癒（歩の始めの W 個が独立に 1/150）
    heal = int(rng.binomial(s.w, HEAL_P)) if s.w else 0

    # 背景：大物の自然回復（N_L が 1 か 2 のときだけ）
    recover = bool(rng.random() < LARGE_RECOVER_P) if s.n_large in (1, 2) else False

    find = False
    catch = False
    combat = None

    if mode == SEARCH_SMALL:
        find = bool(rng.random() < small_search_p(s.n_small))
    elif mode == CHASE_4:
        catch = bool(rng.random() < CATCH_P)
    elif mode == SEARCH_LARGE:
        find = bool(rng.random() < large_search_p(s.n_large, k))
    elif mode == COMBAT:
        combat = COMBAT_OUTCOMES[int(rng.choice(len(COMBAT_OUTCOMES), p=COMBAT_P))]

    return Draws(heal=heal, recover=recover, find=find, catch=catch, combat=combat)


def step(s, action, rng, k=LARGE_SEARCH_EXP):
    """抽選して1歩進める。戻り値: (次の状態, 使った抽選結果)"""
    d = sample_draws(s, action, rng, k)
    return transition(s, action, d), d
