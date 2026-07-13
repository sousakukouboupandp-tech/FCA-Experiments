# -*- coding: utf-8 -*-
"""
共鳴実験 v01 配管検査（スモークテスト）
目的は結果ではなく土俵の検証（設計書§9・捕食者v01の教訓）:
 (1) A（語り手）が痛み記憶を蓄積するか
 (2) Δ大でAの記憶がBにとって誤情報（迷信）になっているか
 (3) 4条件のBの受信量が設計どおり違うか（FullCopy=全量 / ResoNaive≈0 / ResoExp>0）
条件: Isolated / FullCopy(誕生時・全量) / ResoNaive(誕生時・類似度ゲート)
      / ResoExp(幼年期100ep後・類似度ゲート)
規模: 3シード × A500ep + B500ep（結果を解釈しない。配管検査である）

【書記官の自律決定（第9条・作業日誌転記用）】
 - 類似度 sim = spatial(位置距離) × ctx(隣接フラグ一致) × dir(移動方向一致)
 - 又聞き減衰 λ_hear=0.94（自前0.97より速い）／η_res=0.5／タブー閾値0.3はv02踏襲
 - 又聞きが自前体験と同キーで衝突したら自前(強度加算・own化)が勝つ
"""
import sys, json, os
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import numpy as np
exec(open(os.path.join(BASE, 'resonance_v01_env.py'), encoding='utf-8').read())

ETA_RES = 0.5
LAMBDA_OWN = 0.97
LAMBDA_HEAR = 0.94
TABOO_TH = 0.3
CHILDHOOD = 100
N_EP_A = 500
N_EP_B = 500
N_SEEDS = 3

def argmax_rt(qvals, rng):
    finite = np.isfinite(qvals)
    if not finite.any(): return rng.randint(len(qvals))
    mv = np.max(qvals[finite]); cand = np.where((qvals==mv)&finite)[0]
    return rng.choice(cand)

def sim_entry(eA, eB):
    """痛み記録同士の類似度(0-1)。e = ((pos, adj), next_pos)【同病相憐れむ項】"""
    (posA, adjA), nA = eA
    (posB, adjB), nB = eB
    d = abs(posA[0]-posB[0]) + abs(posA[1]-posB[1])
    spatial = max(0.0, 1.0 - d/4.0)
    ctx = 1.0 if adjA == adjB else 0.5
    dirA = (nA[0]-posA[0], nA[1]-posA[1])
    dirB = (nB[0]-posB[0], nB[1]-posB[1])
    dr = 1.0 if dirA == dirB else 0.5
    return spatial * ctx * dr

class Agent:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        self.Q = {}
        self.pm = {}   # key -> [intensity, own(bool)]

    def get_Q(self, s):
        if s not in self.Q: self.Q[s] = np.zeros(N_ACTIONS)
        return self.Q[s]

    def own_keys(self):
        return [k for k, v in self.pm.items() if v[1]]

    def is_tabooed(self, state, ai):
        nxt = move(state[0], ACTIONS[ai])
        if nxt == state[0]: return False
        v = self.pm.get((state, nxt))
        return v is not None and v[0] > TABOO_TH

    def record_pain(self, state, intended):
        k = (state, intended)
        if k in self.pm:
            self.pm[k][0] += 1.0
            self.pm[k][1] = True   # 自前体験がown化して勝つ
        else:
            self.pm[k] = [1.0, True]

    def decay(self):
        for k in list(self.pm.keys()):
            lam = LAMBDA_OWN if self.pm[k][1] else LAMBDA_HEAR
            self.pm[k][0] *= lam
            if self.pm[k][0] < 0.05: del self.pm[k]

def run_episodes(agent, env, n_eps, stats):
    for ep in range(n_eps):
        state = env.reset(ep)
        ep_coll, ep_succ, crossed, total_r = False, False, None, 0.0
        for t in range(MAX_STEPS):
            s = state
            qv = agent.get_Q(s).copy()
            tab = [i for i in range(N_ACTIONS) if agent.is_tabooed(s, i)]
            if len(tab) < N_ACTIONS:
                for i in tab: qv[i] = -np.inf
            if agent.rng.rand() < EPSILON:
                ch = [i for i in range(N_ACTIONS) if i not in tab] or list(range(N_ACTIONS))
                a = agent.rng.choice(ch)
            else:
                a = argmax_rt(qv, agent.rng)
            ns, r, done, info = env.step(a)
            total_r += r
            best = np.max(agent.get_Q(ns)) if not done else 0.0
            agent.get_Q(s)[a] += LR*(r + GAMMA*best - agent.get_Q(s)[a])
            if env.pos == (3, 3): crossed = "mid"
            elif env.pos == (3, 6): crossed = "top"
            if info["collided"]:
                ep_coll = True
                intended = move(s[0], ACTIONS[a])
                agent.record_pain(s, intended)
                stats["coll_pos"].append(info["collided_at"])
            if info["success"]: ep_succ = True
            state = ns
            if done: break
        agent.decay()
        stats["rewards"].append(total_r)
        stats["collisions"] += 1 if ep_coll else 0
        stats["succ"].append(1.0 if ep_succ else 0.0)
        stats["route"].append(crossed)

def transfer(A_pm, B, gated):
    """AのPain MemoryをBへ受け渡す。gated=Trueなら同病相憐れむ項を適用。"""
    own = B.own_keys()
    n_received = 0
    for k, (inten, is_own) in A_pm.items():
        if not is_own: continue
        if gated:
            S = max([sim_entry(k, kb) for kb in own], default=0.0)
            I = inten * ETA_RES * S
        else:
            I = inten  # FullCopy: 強度そのまま全量
        if I > 0.05:
            if k in B.pm:
                B.pm[k][0] = max(B.pm[k][0], I)
            else:
                B.pm[k] = [I, False]
            n_received += 1
    return n_received

def new_stats():
    return {"rewards": [], "collisions": 0, "succ": [], "route": [], "coll_pos": []}

def summarize(stats, received, B):
    last = 100
    routes = [r for r in stats["route"][-last:] if r]
    top_coll = sum(1 for p in stats["coll_pos"] if p and p[1] >= 5)
    mid_coll = sum(1 for p in stats["coll_pos"] if p and p[1] <= 4)
    return {
        "coll_total": stats["collisions"],
        "rew_last100": float(np.mean(stats["rewards"][-last:])),
        "succ_last100": float(np.mean(stats["succ"][-last:])),
        "mid": routes.count("mid"), "top": routes.count("top"),
        "received": received,
        "pm_own_end": len(B.own_keys()),
        "pm_hear_end": len(B.pm) - len(B.own_keys()),
        "coll_at_top": top_coll, "coll_at_mid": mid_coll,
    }

def run_B(cond, patrol, A_pm, seed):
    B = Agent(seed + 1000)
    envB = ResonanceEnv(patrol)
    stats = new_stats()
    received = 0
    if cond == "FullCopy":
        received = transfer(A_pm, B, gated=False)
        run_episodes(B, envB, N_EP_B, stats)
    elif cond == "ResoNaive":
        received = transfer(A_pm, B, gated=True)   # 無垢: own空 → S=0のはず
        run_episodes(B, envB, N_EP_B, stats)
    elif cond == "ResoExp":
        run_episodes(B, envB, CHILDHOOD, stats)    # 幼年期: 自前の傷を得る
        received = transfer(A_pm, B, gated=True)
        run_episodes(B, envB, N_EP_B - CHILDHOOD, stats)
    else:  # Isolated
        run_episodes(B, envB, N_EP_B, stats)
    return summarize(stats, received, B)

if __name__ == "__main__":
    worlds = [("D0", PATROL_B_D0), ("Dsmall", PATROL_B_DSMALL), ("Dbig", PATROL_B_DBIG)]
    conds = ["Isolated", "FullCopy", "ResoNaive", "ResoExp"]
    all_out = {}

    print("=== 配管検査(1): 語り手Aの学習 ===")
    A_cache = {}
    for seed in range(N_SEEDS):
        A = Agent(seed)
        envA = ResonanceEnv(PATROL_WORLD_A)
        sA = new_stats()
        run_episodes(A, envA, N_EP_A, sA)
        A_cache[seed] = {k: list(v) for k, v in A.pm.items() if v[1]}
        A_cache[seed] = {k: v for k, v in A.pm.items()}
        print(f"  seed{seed}: 衝突{sA['collisions']} 痛み記録{len(A.pm)}件 "
              f"末尾報酬{np.mean(sA['rewards'][-100:]):.2f} "
              f"記録サンプル{list(A.pm.keys())[:2]}")

    print("=== 配管検査(2)(3): 4条件 × 3世界 × 3シード ===")
    keys = ["coll_total","rew_last100","succ_last100","mid","top","received",
            "pm_own_end","pm_hear_end","coll_at_top","coll_at_mid"]
    for wname, patrol in worlds:
        all_out[wname] = {}
        for cond in conds:
            agg = {k: [] for k in keys}
            for seed in range(N_SEEDS):
                A_pm = {k: [v[0], v[1]] for k, v in A_cache[seed].items()}
                r = run_B(cond, patrol, A_pm, seed)
                for k in keys: agg[k].append(r[k])
            all_out[wname][cond] = {k: float(np.mean(agg[k])) for k in keys}

    json.dump(all_out, open(os.path.join(BASE, 'resonance_v01_smoke_results.json'),
              'w'), ensure_ascii=False, indent=1)

    for wname, _ in worlds:
        print(f"--- 世界 {wname} ---")
        print(f'{"条件":10s} {"衝突計":>6s} {"末尾報酬":>8s} {"成功率":>6s} '
              f'{"mid":>4s} {"top":>4s} {"受信":>5s} {"own":>4s} {"hear":>4s} '
              f'{"衝突top":>7s} {"衝突mid":>7s}')
        for cond in conds:
            r = all_out[wname][cond]
            print(f'{cond:10s} {r["coll_total"]:6.1f} {r["rew_last100"]:8.2f} '
                  f'{r["succ_last100"]:6.2f} {r["mid"]:4.0f} {r["top"]:4.0f} '
                  f'{r["received"]:5.1f} {r["pm_own_end"]:4.1f} {r["pm_hear_end"]:4.1f} '
                  f'{r["coll_at_top"]:7.1f} {r["coll_at_mid"]:7.1f}')

    print()
    print("=== 検査項目の判定材料 ===")
    print("(1) Aの痛み記録件数が0でないこと → 上の配管検査(1)")
    print("(2) Dbig世界でIsolatedの衝突がtop側に集中していること → 迷信化の前提成立")
    print("(3) 受信量: FullCopy=全量 / ResoNaive≈0 / ResoExp>0 となっていること")
