# -*- coding: utf-8 -*-
"""
共鳴実験 v02 本実行（15シード×4条件×3世界）
【v01からの変更＝身に染みる項（設計書§15・始祖の思想注入第2弾）】
 又聞きの痛みの減衰率を、受信時のシンクロ率Sの関数にする:
   λ_hear(S) = λ_base + (λ_own − λ_base) × S
 自分の傷と似た忠告は身に染みて長生きし、経験と無関係な伝聞はすぐ蒸発する。
 FullCopy(融合)はS=1.0固定＝迷信も我が事として長生きする。
 併せてシード別データを保存する（v01の宿題）。
検証対象: 設計書の仮説H1(安全の移転)/H2(融合の毒)/H3(非融合)/H4(同病相憐れむ)
        および負け条件L1/L2/L3
追加計測: 政策距離 pol_dist ＝ AとBが共有する状態のうち、貪欲行動が異なる割合
          (1に近いほどBはAのコピーではなく自分の答えを持つ)
【v01からの修正】設計書§12の欠陥1・2を修理:
 修正1: 生涯の傷アーカイブ(減衰しない)を追加。語り(transfer)はアーカイブから行う
        —— FCA 4.5「記憶は消えない」に実装を合わせた
 修正2: 又聞きの痛みは想起(類似検索)で発動する。自前の傷はv02互換のピンポイント発動のまま
        —— FCA 4.4「想起」をタブー判定にも適用
 修正3: 同病相憐れむゲートのB側照合も生涯アーカイブに対して行う(疼きが引いた傷でも共鳴できる)
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
LAMBDA_HEAR_BASE = 0.94  # S=0(無関係な伝聞)の減衰率。S=1でλ_ownに漸近【身に染みる項】
TABOO_TH = 0.3
CHILDHOOD = 100
N_EP_A = 1000
N_EP_B = 1000
N_SEEDS = 15

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
        self.pm = {}       # key -> [intensity, own(bool), sync] 疼く傷(減衰する作業記憶)
                           # sync = 受信時のシンクロ率S(自前の傷は1.0)【身に染みる項】
        self.archive = {}  # key -> peak_intensity 生涯の傷跡帳(減衰しない)【v01.1修正1】

    def get_Q(self, s):
        if s not in self.Q: self.Q[s] = np.zeros(N_ACTIONS)
        return self.Q[s]

    def own_keys(self):
        return [k for k, v in self.pm.items() if v[1]]

    def is_tabooed(self, state, ai):
        nxt = move(state[0], ACTIONS[ai])
        if nxt == state[0]: return False
        cand = (state, nxt)
        v = self.pm.get(cand)
        if v is not None and v[1] and v[0] > TABOO_TH:
            return True   # 自前の傷: v02互換のピンポイント発動
        # 又聞きの傷は想起(類似検索)で疼く【v01.1修正2】
        for k, v in self.pm.items():
            if v[1]: continue
            if v[0] * sim_entry(cand, k) > TABOO_TH:
                return True
        return False

    def record_pain(self, state, intended):
        k = (state, intended)
        if k in self.pm:
            self.pm[k][0] += 1.0
            self.pm[k][1] = True   # 自前体験がown化して勝つ
            self.pm[k][2] = 1.0
        else:
            self.pm[k] = [1.0, True, 1.0]
        # 生涯の傷跡帳へ(減衰しない・ピーク強度)【v01.1修正1】
        self.archive[k] = max(self.archive.get(k, 0.0), self.pm[k][0])

    def decay(self):
        for k in list(self.pm.keys()):
            if self.pm[k][1]:
                lam = LAMBDA_OWN
            else:
                # 身に染みる項: シンクロ率が高い忠告ほど自分の傷に近い寿命を持つ
                lam = LAMBDA_HEAR_BASE + (LAMBDA_OWN - LAMBDA_HEAR_BASE) * self.pm[k][2]
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

def transfer(A_archive, B, gated):
    """Aの生涯アーカイブをBへ語る【v01.1修正1: 語りは傷跡帳から】。
    gated=Trueなら同病相憐れむ項を適用。B側の照合も生涯アーカイブ【修正3】。"""
    own = list(B.archive.keys())
    n_received = 0
    for k, inten in A_archive.items():
        if gated:
            S = max([sim_entry(k, kb) for kb in own], default=0.0)
            I = inten * ETA_RES * S
        else:
            S = 1.0    # FullCopy: 迷信も「我が事」として身に染みる(融合)
            I = inten  # 強度そのまま全量
        if I > 0.05:
            if k in B.pm:
                B.pm[k][0] = max(B.pm[k][0], I)
                B.pm[k][2] = max(B.pm[k][2], S)
            else:
                B.pm[k] = [I, False, S]
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

def run_B(cond, patrol, A_pm, greedyA, seed):
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
    res = summarize(stats, received, B)
    # 政策距離【H3: 非融合の計測】
    shared = [s for s in B.Q if s in greedyA]
    if shared:
        diff = sum(1 for s in shared if int(np.argmax(B.Q[s])) != greedyA[s])
        res["pol_dist"] = diff / len(shared)
    else:
        res["pol_dist"] = -1.0
    return res

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
        greedyA = {s: int(np.argmax(q)) for s, q in A.Q.items()}
        A_cache[seed] = (dict(A.archive), greedyA)   # 語りは生涯アーカイブから【v01.1】
        print(f"  seed{seed}: 衝突{sA['collisions']} 疼く傷{len(A.pm)}件 "
              f"生涯アーカイブ{len(A.archive)}件 "
              f"末尾報酬{np.mean(sA['rewards'][-100:]):.2f}")

    print("=== 配管検査(2)(3): 4条件 × 3世界 × 3シード ===")
    keys = ["coll_total","rew_last100","succ_last100","mid","top","received",
            "pm_own_end","pm_hear_end","coll_at_top","coll_at_mid","pol_dist"]
    for wname, patrol in worlds:
        all_out[wname] = {}
        for cond in conds:
            agg = {k: [] for k in keys}
            for seed in range(N_SEEDS):
                A_pm = dict(A_cache[seed][0])
                r = run_B(cond, patrol, A_pm, A_cache[seed][1], seed)
                for k in keys: agg[k].append(r[k])
            all_out[wname][cond] = {
                "mean": {k: float(np.mean(agg[k])) for k in keys},
                "std":  {k: float(np.std(agg[k])) for k in keys},
                "per_seed": {k: [float(x) for x in agg[k]] for k in keys},
            }

    json.dump(all_out, open(os.path.join(BASE, 'resonance_v02_full_results_15seed.json'),
              'w'), ensure_ascii=False, indent=1)

    for wname, _ in worlds:
        print(f"--- 世界 {wname} ---")
        print(f'{"条件":10s} {"衝突計":>6s} {"末尾報酬":>8s} {"成功率":>6s} '
              f'{"mid":>4s} {"top":>4s} {"受信":>5s} '
              f'{"衝突top":>7s} {"衝突mid":>7s} {"政策距離":>8s}')
        for cond in conds:
            r = all_out[wname][cond]["mean"]
            sd = all_out[wname][cond]["std"]
            print(f'{cond:10s} {r["coll_total"]:6.1f} {r["rew_last100"]:8.2f} '
                  f'{r["succ_last100"]:6.2f} {r["mid"]:4.0f} {r["top"]:4.0f} '
                  f'{r["received"]:5.1f} '
                  f'{r["coll_at_top"]:7.1f} {r["coll_at_mid"]:7.1f} {r["pol_dist"]:8.2f} '
                  f'(衝突σ{sd["coll_total"]:.1f})')

    print()
    print("=== 検査項目の判定材料 ===")
    print("(1) Aの痛み記録件数が0でないこと → 上の配管検査(1)")
    print("(2) Dbig世界でIsolatedの衝突がtop側に集中していること → 迷信化の前提成立")
    print("(3) 受信量: FullCopy=全量 / ResoNaiveはほぼ0 / ResoExp>0 となっていること")
