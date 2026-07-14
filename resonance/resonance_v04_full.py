# -*- coding: utf-8 -*-
"""
共鳴実験 v04 本実行（30シード×4条件×4世界）—— H2「融合の毒」決着戦
事前登録: PREREG_v04_H2決着.md §1-6(登録) + §7(第1R監査・凍結) + §8(第2R監査・最終凍結)
実行後の変更はバグ例外条項(§7-6)を除き行わない。

【v03からの変更（すべて事前凍結済み）】
 1. シード 15→30
 2. 世界にDbig+を追加: 捕食者が(1,6)と(2,6)を横往復＝遠回り路(y=6)を2マス封鎖。
    最短路(y=3)は完全安全。「仮説検証用に意図的に設計された敵対的環境」(§7-7)
 3. 条件: Isolated / FullCopy / ResoExp / FullCopyDelayed(探索的・幼年期100ep後に
    全量S=1受信、主要結論は導かない §8-10)。ResoNaiveは凍結構成から除外
 4. 主要指標: 衝突エピソード数(§8-2)。主要検定は2本のみ:
    P1 = Dbig+ FullCopy vs Isolated (1-1000ep)
    P2 = Dbig+ ResoExp  vs Isolated (101-1000ep)
 5. 統計: 対応ありウィルコクソン(zero_method='pratt' §8-5)+ホルム補正、
    参考に対応ありt検定、効果サイズ rank-biserial(§8-9)
 6. 記録: ResoExp受信S値分布(平均/最小/最大/中央値/四分位 §8-8)、S>0シード数(§8-3)、
    100ep時点の最短路Q値スナップショット(§8-7・免疫の定量)、シード別生データ全保存
 7. 全シードでS=0ならP2は「測定不能」と報告(§8-3)。全ペア差分ゼロなら検定不能(§8-5)
 パラメータはv03全凍結: ETA_RES=1.0, λ_own=0.97, λ_hear_base=0.94,
 taboo=0.3, 幼年期=100ep, A=1000ep(据え置き §7-1), B=1000ep
"""
import sys, json, os
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import numpy as np
from scipy import stats as spstats
exec(open(os.path.join(BASE, 'resonance_v01_env.py'), encoding='utf-8').read())

# Dbig+: y=6行の(1,6)と(2,6)を往復する巡回リスト（遠回り路2マス封鎖・最短路は完全安全）
PATROL_B_DBIGPLUS = [(1, 6), (2, 6)]

ETA_RES = 1.0            # v03凍結: 門番はシンクロ率Sのみ
LAMBDA_OWN = 0.97
LAMBDA_HEAR_BASE = 0.94  # 身に染みる項: λ_hear(S)=base+(own-base)*S
TABOO_TH = 0.3
CHILDHOOD = 100
N_EP_A = 1000            # §7-1: A延長は撤回、1000ep据え置き
N_EP_B = 1000
N_SEEDS = 30             # §2: 15→30

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
        self.pm = {}       # key -> [intensity, own(bool), sync] 疼く傷(減衰する)
        self.archive = {}  # key -> peak_intensity 生涯の傷跡帳(減衰しない・語りの源)

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
            return True   # 自前の傷: ピンポイント発動
        for k, v in self.pm.items():
            if v[1]: continue
            if v[0] * sim_entry(cand, k) > TABOO_TH:
                return True  # 又聞きの傷: 想起(類似検索)で疼く
        return False

    def record_pain(self, state, intended):
        k = (state, intended)
        if k in self.pm:
            self.pm[k][0] += 1.0
            self.pm[k][1] = True
            self.pm[k][2] = 1.0
        else:
            self.pm[k] = [1.0, True, 1.0]
        self.archive[k] = max(self.archive.get(k, 0.0), self.pm[k][0])

    def decay(self):
        for k in list(self.pm.keys()):
            if self.pm[k][1]:
                lam = LAMBDA_OWN
            else:
                lam = LAMBDA_HEAR_BASE + (LAMBDA_OWN - LAMBDA_HEAR_BASE) * self.pm[k][2]
            self.pm[k][0] *= lam
            if self.pm[k][0] < 0.05: del self.pm[k]

def new_stats():
    return {"rewards": [], "collisions": 0, "succ": [], "route": [],
            "coll_pos": [], "coll_flags": []}

def run_episodes(agent, env, n_eps, stats):
    for ep in range(n_eps):
        state = env.reset(ep)
        ep_coll, crossed, total_r = False, None, 0.0
        ep_succ = False
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
        stats["coll_flags"].append(1 if ep_coll else 0)
        stats["succ"].append(1.0 if ep_succ else 0.0)
        stats["route"].append(crossed)

def transfer(A_archive, B, gated):
    """Aの生涯アーカイブをBへ語る。gated=Trueなら同病相憐れむゲート適用。
    戻り値: (受信件数, 提示されたS全件, 受理されたS全件) §8-3/§8-8"""
    own = list(B.archive.keys())
    n_received = 0
    S_presented, S_accepted = [], []
    for k, inten in A_archive.items():
        if gated:
            S = max([sim_entry(k, kb) for kb in own], default=0.0)
            I = inten * ETA_RES * S
        else:
            S = 1.0    # FullCopy系: 迷信も「我が事」として身に染みる(融合)
            I = inten
        S_presented.append(S)
        if I > 0.05:
            if k in B.pm:
                B.pm[k][0] = max(B.pm[k][0], I)
                B.pm[k][2] = max(B.pm[k][2], S)
            else:
                B.pm[k] = [I, False, S]
            n_received += 1
            S_accepted.append(S)
    return n_received, S_presented, S_accepted

def mid_route_q(B):
    """最短路(y=3)上の既知状態のmaxQ平均。100ep時点の「免疫」の定量(§8-7・Gemini対策)"""
    vals = [float(np.max(q)) for s, q in B.Q.items() if s[0][1] == 3]
    return float(np.mean(vals)) if vals else 0.0

def s_summary(S_list):
    """S値分布の要約(§8-8: 平均/最小/最大/中央値/四分位)"""
    if not S_list:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0,
                "median": 0.0, "q1": 0.0, "q3": 0.0}
    a = np.array(S_list)
    return {"n": int(len(a)), "mean": float(a.mean()), "min": float(a.min()),
            "max": float(a.max()), "median": float(np.median(a)),
            "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75))}

def summarize(stats, received, B):
    last = 100
    routes = [r for r in stats["route"][-last:] if r]
    top_coll = sum(1 for p in stats["coll_pos"] if p and p[1] >= 5)
    mid_coll = sum(1 for p in stats["coll_pos"] if p and p[1] <= 4)
    return {
        "coll_total": stats["collisions"],  # 主要指標(§8-2): 衝突エピソード数(1-1000ep)
        "coll_ep_101_1000": int(sum(stats["coll_flags"][CHILDHOOD:])),  # P2区間
        "coll_events": len(stats["coll_pos"]),  # 副次: 衝突イベント総数
        "rew_last100": float(np.mean(stats["rewards"][-last:])),
        "succ_last100": float(np.mean(stats["succ"][-last:])),
        "mid": routes.count("mid"), "top": routes.count("top"),
        "received": received,
        "pm_own_end": len(B.own_keys()),
        "pm_hear_end": len(B.pm) - len(B.own_keys()),
        "coll_at_top": top_coll, "coll_at_mid": mid_coll,
    }

def run_B(cond, patrol, A_pm, greedyA, seed):
    B = Agent(seed + 1000)   # 対応あり設計: 全条件で同一seed+1000(Grokペア保証)
    envB = ResonanceEnv(patrol)
    st = new_stats()
    received, S_pre, S_acc, q_snap100 = 0, [], [], None
    if cond == "FullCopy":
        received, S_pre, S_acc = transfer(A_pm, B, gated=False)
        run_episodes(B, envB, N_EP_B, st)
    elif cond == "ResoExp":
        run_episodes(B, envB, CHILDHOOD, st)     # 幼年期: 自前の傷を得る
        q_snap100 = mid_route_q(B)               # §8-7: 免疫スナップショット
        received, S_pre, S_acc = transfer(A_pm, B, gated=True)
        run_episodes(B, envB, N_EP_B - CHILDHOOD, st)
    elif cond == "FullCopyDelayed":              # §7-9: 探索的第4条件
        run_episodes(B, envB, CHILDHOOD, st)
        q_snap100 = mid_route_q(B)               # §8-7: 免疫スナップショット
        received, S_pre, S_acc = transfer(A_pm, B, gated=False)
        run_episodes(B, envB, N_EP_B - CHILDHOOD, st)
    else:  # Isolated
        run_episodes(B, envB, N_EP_B, st)
    res = summarize(st, received, B)
    res["n_S_pos"] = int(sum(1 for s in S_pre if s > 0))  # §8-3: S>0の提示件数
    res["S_acc"] = s_summary(S_acc)
    res["S_pre"] = s_summary(S_pre)
    res["q_mid_100ep"] = q_snap100 if q_snap100 is not None else -1.0
    shared = [s for s in B.Q if s in greedyA]
    if shared:
        diff = sum(1 for s in shared if int(np.argmax(B.Q[s])) != greedyA[s])
        res["pol_dist"] = diff / len(shared)
    else:
        res["pol_dist"] = -1.0
    return res

def paired_wilcoxon_pratt(x, y):
    """対応ありウィルコクソン(zero_method='pratt' §8-5)+rank-biserial(§8-9)。
    全ペア差分ゼロなら None = 検定不能(§8-5)"""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    d = x - y
    if np.all(d == 0):
        return None
    w, p = spstats.wilcoxon(x, y, zero_method='pratt')
    dz = d[d != 0]
    ranks = spstats.rankdata(np.abs(dz))
    wp = float(ranks[dz > 0].sum()); wm = float(ranks[dz < 0].sum())
    rb = (wp - wm) / (wp + wm) if (wp + wm) > 0 else 0.0
    t_stat, t_p = spstats.ttest_rel(x, y)  # 参考値: 対応ありt
    return {"W": float(w), "p": float(p), "rank_biserial": rb,
            "t_ref": float(t_stat), "t_p_ref": float(t_p),
            "mean_diff": float(np.mean(d)), "median_diff": float(np.median(d))}

def holm_2(p1, p2):
    """ホルム＝ボンフェローニ補正(主要2仮説専用 §7-3)"""
    if p1 is None or p2 is None:
        return p1, p2
    if p1 <= p2:
        a1 = min(1.0, 2*p1); a2 = min(1.0, max(a1, p2))
    else:
        a2 = min(1.0, 2*p2); a1 = min(1.0, max(a2, p1))
    return a1, a2

def iqr_str(vals):
    a = np.array(vals, dtype=float)
    return "中央値%.1f IQR[%.1f-%.1f]" % (np.median(a),
           np.percentile(a, 25), np.percentile(a, 75))

if __name__ == "__main__":
    worlds = [("D0", PATROL_B_D0), ("Dsmall", PATROL_B_DSMALL),
              ("Dbig", PATROL_B_DBIG), ("DbigPlus", PATROL_B_DBIGPLUS)]
    conds = ["Isolated", "FullCopy", "ResoExp", "FullCopyDelayed"]
    all_out = {}

    print("=== (1) 語り手Aの学習 30シード ===")
    A_cache = {}
    for seed in range(N_SEEDS):
        A = Agent(seed)
        envA = ResonanceEnv(PATROL_WORLD_A)
        sA = new_stats()
        run_episodes(A, envA, N_EP_A, sA)
        greedyA = {s: int(np.argmax(q)) for s, q in A.Q.items()}
        A_cache[seed] = (dict(A.archive), greedyA)
        if seed < 3 or seed == N_SEEDS-1:
            print("  seed%d: 衝突ep%d 生涯アーカイブ%d件" % (
                seed, sA['collisions'], len(A.archive)))

    print("=== (2) 4条件 x 4世界 x 30シード ===")
    scalar_keys = ["coll_total","coll_ep_101_1000","coll_events","rew_last100",
                   "succ_last100","mid","top","received","pm_own_end",
                   "pm_hear_end","coll_at_top","coll_at_mid","n_S_pos",
                   "q_mid_100ep","pol_dist"]

    for wname, patrol in worlds:
        all_out[wname] = {}
        for cond in conds:
            agg = {k: [] for k in scalar_keys}
            s_accs, s_pres = [], []
            for seed in range(N_SEEDS):
                A_pm = dict(A_cache[seed][0])
                r = run_B(cond, patrol, A_pm, A_cache[seed][1], seed)
                for k in scalar_keys: agg[k].append(r[k])
                s_accs.append(r["S_acc"]); s_pres.append(r["S_pre"])
            all_out[wname][cond] = {
                "mean": {k: float(np.mean(agg[k])) for k in scalar_keys},
                "std":  {k: float(np.std(agg[k])) for k in scalar_keys},
                "per_seed": {k: [float(x) for x in agg[k]] for k in scalar_keys},
                "S_acc_per_seed": s_accs,
                "S_pre_per_seed": s_pres,
            }
        print("  世界 %s 完了" % wname)

    # ===== 主要検定(§7-3/§8): Dbig+のP1・P2の2本のみ =====
    iso = all_out["DbigPlus"]["Isolated"]["per_seed"]
    fc  = all_out["DbigPlus"]["FullCopy"]["per_seed"]
    rex = all_out["DbigPlus"]["ResoExp"]["per_seed"]

    P1 = paired_wilcoxon_pratt(fc["coll_total"], iso["coll_total"])
    # §8-3: 測定不能規則 —— 全30シードでS=0(提示すら非ゼロなし)ならP2は測定不能
    n_seeds_S_pos = int(sum(1 for n in rex["n_S_pos"] if n > 0))
    if n_seeds_S_pos == 0:
        P2 = "MEASUREMENT_IMPOSSIBLE"
    else:
        P2 = paired_wilcoxon_pratt(rex["coll_ep_101_1000"], iso["coll_ep_101_1000"])

    # ホルム補正(§7-3)
    if isinstance(P2, dict) and P1 is not None:
        p1h, p2h = holm_2(P1["p"], P2["p"])
        P1["p_holm"] = p1h; P2["p_holm"] = p2h
    elif P1 is not None:
        P1["p_holm"] = P1["p"]  # P2測定不能/検定不能時はP1単独(補正なし=1本)

    # 参考解析(主要結論には数えない): P3再現チェック(D0)と相互区間、Delayed
    ref = {}
    d0i = all_out["D0"]["Isolated"]["per_seed"]
    ref["P3_D0_FullCopy"] = paired_wilcoxon_pratt(
        all_out["D0"]["FullCopy"]["per_seed"]["coll_total"], d0i["coll_total"])
    ref["P3_D0_ResoExp"] = paired_wilcoxon_pratt(
        all_out["D0"]["ResoExp"]["per_seed"]["coll_total"], d0i["coll_total"])
    ref["P1_alt_101_1000"] = paired_wilcoxon_pratt(
        fc["coll_ep_101_1000"], iso["coll_ep_101_1000"])
    ref["P2_alt_1_1000"] = paired_wilcoxon_pratt(
        rex["coll_total"], iso["coll_total"])
    ref["Delayed_vs_Iso_101_1000"] = paired_wilcoxon_pratt(
        all_out["DbigPlus"]["FullCopyDelayed"]["per_seed"]["coll_ep_101_1000"],
        iso["coll_ep_101_1000"])

    all_out["_tests"] = {
        "P1_DbigPlus_FullCopy_vs_Isolated_1_1000": P1,
        "P2_DbigPlus_ResoExp_vs_Isolated_101_1000": P2,
        "P2_n_seeds_S_pos": n_seeds_S_pos,
        "reference_not_primary": ref,
    }
    json.dump(all_out, open(os.path.join(BASE,
        'resonance_v04_full_results_30seed.json'), 'w'), ensure_ascii=False, indent=1)
    print("JSON保存完了: resonance_v04_full_results_30seed.json")

    # ===== レポート出力 =====
    for wname, _ in worlds:
        print("--- 世界 %s ---" % wname)
        print("%-16s %6s %8s %6s %4s %4s %5s %6s %8s" % (
            "条件","衝突ep","101-1000","成功率","mid","top","受信","S>0件","政策距離"))
        for cond in conds:
            r = all_out[wname][cond]["mean"]
            sd = all_out[wname][cond]["std"]
            print("%-16s %6.1f %8.1f %6.2f %4.0f %4.0f %5.1f %6.1f %8.2f (sd%.1f)" % (
                cond, r["coll_total"], r["coll_ep_101_1000"], r["succ_last100"],
                r["mid"], r["top"], r["received"], r["n_S_pos"],
                r["pol_dist"], sd["coll_total"]))

    print()
    print("=== 主要検定(事前登録・2本のみ) ===")
    print("[P1] Dbig+ FullCopy vs Isolated (衝突ep数 1-1000)")
    print("     FullCopy: %s / Isolated: %s" % (
        iqr_str(fc["coll_total"]), iqr_str(iso["coll_total"])))
    if P1 is None:
        print("     全ペア差分ゼロ: 検定不能(§8-5)")
    else:
        print("     Wilcoxon(pratt) W=%.1f p=%.4f p_holm=%.4f rb=%.3f (参考t p=%.4f)" % (
            P1["W"], P1["p"], P1["p_holm"], P1["rank_biserial"], P1["t_p_ref"]))
    print("[P2] Dbig+ ResoExp vs Isolated (衝突ep数 101-1000)")
    print("     S>0だったシード数: %d / %d" % (n_seeds_S_pos, N_SEEDS))
    if P2 == "MEASUREMENT_IMPOSSIBLE":
        print("     全シードS=0: P2は測定不能と報告(§8-3)")
    elif P2 is None:
        print("     全ペア差分ゼロ: 検定不能(§8-5)")
    else:
        print("     ResoExp: %s / Isolated: %s" % (
            iqr_str(rex["coll_ep_101_1000"]), iqr_str(iso["coll_ep_101_1000"])))
        print("     Wilcoxon(pratt) W=%.1f p=%.4f p_holm=%.4f rb=%.3f (参考t p=%.4f)" % (
            P2["W"], P2["p"], P2["p_holm"], P2["rank_biserial"], P2["t_p_ref"]))
    print()
    print("判定はPREREG_v04_H2決着.mdの事前登録基準のみで行うこと。")
