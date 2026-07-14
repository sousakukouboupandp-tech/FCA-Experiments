# -*- coding: utf-8 -*-
"""
幼年期スイープ実験 v05 本実行（S(t)＝受信タイミングとシンクロ率）
事前登録: DESIGN_幼年期スイープ_v04_凍結.md（2026年7月15日凍結・3ラウンド監査完了）
実行後の変更はバグ例外条項（共鳴実験v04 PREREG §7-6継承）を除き行わない。

主要検定（2本のみ・ホルム補正・D0）:
 P1(非飽和) = ResoExp: S̄_pre(τ=400) vs S̄_pre(τ=25) 対応ありウィルコクソン(pratt)
 P2(薄い経験の門) = ResoExp(τ=25) vs Isolated 衝突ep数[26,1000] 同上
記録: 提示/受理S分布・S>0シード数・固定窓[τ+1,τ+600]・絶対削減度・
      Delayed q_midスナップショット・シード別生データ(coll_flags)全保存
実行時QA: 提示件数==Aアーカイブ件数（全run検証・不一致で停止）
使い方: python resonance_v05_sweep.py diag  → 単一シード診断
        python resonance_v05_sweep.py full  → 30シード本実行
"""
import sys, json, os, time
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import numpy as np
from scipy import stats as spstats
exec(open(os.path.join(BASE, 'resonance_v01_env.py'), encoding='utf-8').read())

PATROL_B_DBIGPLUS = [(1, 6), (2, 6)]   # v04凍結地形（土俵ゲート合格済み・流用）
ETA_RES = 1.0
LAMBDA_OWN = 0.97
LAMBDA_HEAR_BASE = 0.94
TABOO_TH = 0.3
TAUS = [0, 25, 50, 100, 200, 400]
TAUS_DELAYED = [25, 50, 100, 200, 400]
N_EP_A = 1000
N_EP_B = 1000
N_SEEDS = 30
FIXED_WIN = 600

def argmax_rt(qvals, rng):
    finite = np.isfinite(qvals)
    if not finite.any(): return rng.randint(len(qvals))
    mv = np.max(qvals[finite]); cand = np.where((qvals==mv)&finite)[0]
    return rng.choice(cand)

def sim_entry(eA, eB):
    """痛み記録同士の類似度(0-1)。e = ((pos, adj), next_pos)【同病相憐れむ項】v04凍結流用"""
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
        self.pm = {}       # key -> [intensity, own(bool), sync]
        self.archive = {}  # key -> peak_intensity（生涯の傷跡帳・語りの源）

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
            return True
        for k, v in self.pm.items():
            if v[1]: continue
            if v[0] * sim_entry(cand, k) > TABOO_TH:
                return True
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
    return {"rewards": [], "succ": [], "route": [], "coll_pos": [], "coll_flags": []}

def mid_route_q(B):
    """最短路(y=3)上の既知状態のmaxQ平均（免疫の定量・v04 §8-7）"""
    vals = [float(np.max(q)) for s, q in B.Q.items() if s[0][1] == 3]
    return float(np.mean(vals)) if vals else 0.0

def run_episodes(agent, env, n_eps, stats, snaps=None, snap_offset=0):
    """v04と同一のエピソードループ。snaps指定時は100epごとにq_mid記録（Delayed用）。
    2フェーズ構造（幼年期→受信後）はv04と同一：env.reset(ep)のepは各呼び出し内で0始まり。"""
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
        stats["coll_flags"].append(1 if ep_coll else 0)
        stats["succ"].append(1.0 if ep_succ else 0.0)
        stats["route"].append(crossed)
        if snaps is not None and (ep + 1) % 100 == 0:
            snaps.append([snap_offset + ep + 1, mid_route_q(agent)])

def transfer(A_archive, B, gated):
    """Aの生涯アーカイブをBへ語る。gated=Trueなら同病相憐れむゲート適用（v04凍結流用）。
    戻り値: (受信件数, 提示S全件, 受理S全件)"""
    own = list(B.archive.keys())
    n_received = 0
    S_presented, S_accepted = [], []
    for k, inten in A_archive.items():
        if gated:
            S = max([sim_entry(k, kb) for kb in own], default=0.0)
            I = inten * ETA_RES * S
        else:
            S = 1.0
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

def s_summary(S_list):
    if not S_list:
        return {"n": 0, "mean": 0.0, "min": 0.0, "max": 0.0,
                "median": 0.0, "q1": 0.0, "q3": 0.0}
    a = np.array(S_list)
    return {"n": int(len(a)), "mean": float(a.mean()), "min": float(a.min()),
            "max": float(a.max()), "median": float(np.median(a)),
            "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75))}

def pack(st, received, S_pre, S_acc, q_snaps):
    last = 100
    routes = [r for r in st["route"][-last:] if r]
    return {
        "coll_flags": st["coll_flags"],  # 生データ: ep1..1000の衝突フラグ（区間計算の源）
        "succ_last100": float(np.mean(st["succ"][-last:])),
        "mid": routes.count("mid"), "top": routes.count("top"),
        "received": received,
        "n_presented": len(S_pre),
        "n_S_pos": int(sum(1 for s in S_pre if s > 0)),
        "S_pre_mean": (float(np.mean(S_pre)) if S_pre else None),   # シード別S̄_pre（P1の検定単位）
        "S_acc_mean": (float(np.mean(S_acc)) if S_acc else None),   # 欠測=None（§3）
        "S_pre_sum": s_summary(S_pre), "S_acc_sum": s_summary(S_acc),
        "S_pre_raw": [float(s) for s in S_pre],
        "q_snaps": q_snaps,
    }

def run_B_sweep(cond, patrol, A_pm, seed, tau):
    """条件×τの1本。Isolatedはtau無視で1000ep素通し。"""
    B = Agent(seed + 1000)   # 対応あり設計: 全条件・全τで同一seed+1000
    envB = ResonanceEnv(patrol)
    st = new_stats()
    if cond == "Isolated":
        run_episodes(B, envB, N_EP_B, st)
        return pack(st, 0, [], [], None)
    if tau > 0:
        run_episodes(B, envB, tau, st)   # 幼年期（v04と同一の2フェーズ構造）
    q_before = mid_route_q(B)
    gated = (cond == "ResoExp")
    received, S_pre, S_acc = transfer(A_pm, B, gated=gated)
    q_after = mid_route_q(B)
    snaps = [[tau, q_before], [tau, q_after]] if cond == "FullCopyDelayed" else None
    run_episodes(B, envB, N_EP_B - tau, st, snaps=snaps, snap_offset=tau)
    return pack(st, received, S_pre, S_acc, snaps)

def qa_or_die(r, expected, label, need_full_receive=False):
    """実行時QA（凍結§3）: 提示件数==Aアーカイブ件数。Delayedは受信件数も一致（ロードQA）。"""
    if r["n_presented"] != expected:
        print("!!! QA不一致(提示件数): %s 提示%d 期待%d → 停止" % (label, r["n_presented"], expected))
        sys.exit(1)
    if need_full_receive and r["received"] != expected:
        print("!!! ロードQA不一致(受信件数): %s 受信%d 期待%d → 停止" % (label, r["received"], expected))
        sys.exit(1)

def learn_A(seed):
    A = Agent(seed)
    envA = ResonanceEnv(PATROL_WORLD_A)
    sA = new_stats()
    run_episodes(A, envA, N_EP_A, sA)
    return dict(A.archive)

def paired_wilcoxon_pratt(x, y):
    """対応ありウィルコクソン(pratt)+rank-biserial+参考t。全ペア差分ゼロならNone=検定不能。"""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    d = x - y
    if np.all(d == 0):
        return None
    w, p = spstats.wilcoxon(x, y, zero_method='pratt')
    dz = d[d != 0]
    ranks = spstats.rankdata(np.abs(dz))
    wp = float(ranks[dz > 0].sum()); wm = float(ranks[dz < 0].sum())
    rb = (wp - wm) / (wp + wm) if (wp + wm) > 0 else 0.0
    t_stat, t_p = spstats.ttest_rel(x, y)
    return {"W": float(w), "p": float(p), "rank_biserial": rb,
            "t_ref": float(t_stat), "t_p_ref": float(t_p),
            "mean_diff": float(np.mean(d)), "median_diff": float(np.median(d))}

def holm_2(p1, p2):
    if p1 is None or p2 is None:
        return p1, p2
    if p1 <= p2:
        a1 = min(1.0, 2*p1); a2 = min(1.0, max(a1, p2))
    else:
        a2 = min(1.0, 2*p2); a1 = min(1.0, max(a2, p1))
    return a1, a2

def iqr_str(vals):
    a = np.array(vals, dtype=float)
    return "中央値%.2f IQR[%.2f-%.2f]" % (np.median(a), np.percentile(a,25), np.percentile(a,75))

def coll_interval(flags, lo, hi):
    """区間[lo,hi]（両端含む・1始まり）の衝突ep数"""
    return int(sum(flags[lo-1:hi]))

def diag():
    """単一シード診断（実験前の必須手順）: seed0で構造チェック。統計はしない。"""
    print("=== 診断モード（seed0・統計なし・構造チェックのみ） ===")
    A_pm = learn_A(0)
    na = len(A_pm)
    print("Aアーカイブ件数:", na)
    res = {}
    for tau in (25, 400):
        r = run_B_sweep("ResoExp", PATROL_B_D0, dict(A_pm), 0, tau)
        qa_or_die(r, na, "diag ResoExp tau=%d" % tau)
        res[tau] = r
        print("D0 ResoExp τ=%d: 提示%d S̄_pre=%.4f S>0件=%d 受理%d 衝突ep[τ+1,1000]=%d" % (
            tau, r["n_presented"], r["S_pre_mean"], r["n_S_pos"], r["received"],
            coll_interval(r["coll_flags"], tau+1, 1000)))
    if res[400]["S_pre_mean"] < res[25]["S_pre_mean"] - 1e-12:
        print("!!! 設計により真のはずの単調非減少が破れた → 実装バグ。停止"); sys.exit(1)
    if res[400]["S_pre_raw"] == res[25]["S_pre_raw"]:
        print("注意: τ=25と400で提示S全件が完全一致（差分ゼロ）。分布を確認せよ")
    rD = run_B_sweep("FullCopyDelayed", PATROL_B_DBIGPLUS, dict(A_pm), 0, 100)
    qa_or_die(rD, na, "diag Delayed tau=100", need_full_receive=True)
    print("Dbig+ Delayed τ=100: 受信%d==アーカイブ%d ロードQA合格 q_snaps点数=%d" % (
        rD["received"], na, len(rD["q_snaps"])))
    rI = run_B_sweep("Isolated", PATROL_B_D0, dict(A_pm), 0, 0)
    print("D0 Isolated: 衝突ep[1,1000]=%d 成功率(末尾100)=%.2f flags長=%d" % (
        coll_interval(rI["coll_flags"], 1, 1000), rI["succ_last100"], len(rI["coll_flags"])))
    for tau in (25, 400):
        assert len(res[tau]["coll_flags"]) == 1000, "flags長異常"
    print("=== 診断: 全チェック合格 ===")

def full():
    t0 = time.time()
    print("=== (1) 語り手Aの学習 30シード ===")
    A_cache, A_counts = {}, {}
    for seed in range(N_SEEDS):
        A_cache[seed] = learn_A(seed)
        A_counts[seed] = len(A_cache[seed])
    print("Aアーカイブ件数: min%d max%d" % (min(A_counts.values()), max(A_counts.values())))

    worlds = [("D0", PATROL_B_D0), ("DbigPlus", PATROL_B_DBIGPLUS)]
    out = {"A_archive_counts": {str(k): v for k, v in A_counts.items()}}
    print("=== (2) スイープ本実行 ===")
    for wname, patrol in worlds:
        out[wname] = {}
        conds = [("Isolated", [None])] + [("ResoExp", TAUS)]
        if wname == "DbigPlus":
            conds.append(("FullCopyDelayed", TAUS_DELAYED))
        for cond, taus in conds:
            for tau in taus:
                key = cond if cond == "Isolated" else "%s_%d" % (cond, tau)
                runs = []
                for seed in range(N_SEEDS):
                    r = run_B_sweep(cond, patrol, dict(A_cache[seed]), seed,
                                    0 if tau is None else tau)
                    if cond != "Isolated":
                        qa_or_die(r, A_counts[seed], "%s %s seed%d" % (wname, key, seed),
                                  need_full_receive=(cond == "FullCopyDelayed"))
                    runs.append(r)
                out[wname][key] = runs
                print("  %s %s 完了 (%.0f秒)" % (wname, key, time.time()-t0))
    return out, t0

def analyze(out):
    d0 = out["D0"]
    # ===== 主要検定（凍結§4・2本のみ） =====
    sp25 = [r["S_pre_mean"] for r in d0["ResoExp_25"]]
    sp400 = [r["S_pre_mean"] for r in d0["ResoExp_400"]]
    P1 = paired_wilcoxon_pratt(sp400, sp25)  # 方向: mean_diff>0 ならτ=400>τ=25
    iso = d0["Isolated"]
    c_iso25 = [coll_interval(r["coll_flags"], 26, 1000) for r in iso]
    c_r25 = [coll_interval(r["coll_flags"], 26, 1000) for r in d0["ResoExp_25"]]
    n_seeds_S_pos_25 = sum(1 for r in d0["ResoExp_25"] if r["n_S_pos"] > 0)
    if n_seeds_S_pos_25 == 0:
        P2 = "MEASUREMENT_IMPOSSIBLE"
    else:
        P2 = paired_wilcoxon_pratt(c_r25, c_iso25)
    if isinstance(P2, dict) and P1 is not None:
        p1h, p2h = holm_2(P1["p"], P2["p"])
        P1["p_holm"] = p1h; P2["p_holm"] = p2h
    elif P1 is not None:
        P1["p_holm"] = P1["p"]
    tests = {"P1_D0_Spre_400_vs_25": P1,
             "P2_D0_Reso25_vs_Isolated_26_1000": P2,
             "P2_n_seeds_S_pos_tau25": n_seeds_S_pos_25,
             "P1_data": {"sp25": sp25, "sp400": sp400},
             "P2_data": {"reso25": c_r25, "iso": c_iso25}}
    # ===== 参考（主要結論に数えない）: 各τの対Isolated・固定窓・絶対削減度・Delayed =====
    ref = {}
    for wname in ("D0", "DbigPlus"):
        w = out[wname]; isoW = w["Isolated"]
        for tau in TAUS:
            key = "ResoExp_%d" % tau
            ci = [coll_interval(r["coll_flags"], tau+1, 1000) for r in isoW]
            cr = [coll_interval(r["coll_flags"], tau+1, 1000) for r in w[key]]
            L = 1000 - tau
            fi = [coll_interval(r["coll_flags"], tau+1, tau+FIXED_WIN) for r in isoW]
            fr = [coll_interval(r["coll_flags"], tau+1, tau+FIXED_WIN) for r in w[key]]
            ref["%s_tau%d" % (wname, tau)] = {
                "wilcoxon_vs_iso": paired_wilcoxon_pratt(cr, ci),
                "coll_reso_mean": float(np.mean(cr)), "coll_iso_mean": float(np.mean(ci)),
                "abs_reduction_mean": float(np.mean([(a-b)/L for a, b in zip(ci, cr)])),
                "fixedwin_reso_mean": float(np.mean(fr)), "fixedwin_iso_mean": float(np.mean(fi)),
                "S_pre_mean_of_means": float(np.mean([r["S_pre_mean"] for r in w[key]])),
                "S_pre_median_of_means": float(np.median([r["S_pre_mean"] for r in w[key]])),
                "n_seeds_S_pos": int(sum(1 for r in w[key] if r["n_S_pos"] > 0)),
                "S_acc_missing": int(sum(1 for r in w[key] if r["S_acc_mean"] is None)),
                "S_acc_mean_of_means": (float(np.mean([r["S_acc_mean"] for r in w[key]
                    if r["S_acc_mean"] is not None]))
                    if any(r["S_acc_mean"] is not None for r in w[key]) else None),
            }
        if wname == "DbigPlus":
            for tau in TAUS_DELAYED:
                key = "FullCopyDelayed_%d" % tau
                ci = [coll_interval(r["coll_flags"], tau+1, 1000) for r in isoW]
                cd = [coll_interval(r["coll_flags"], tau+1, 1000) for r in w[key]]
                ref["Delayed_tau%d_vs_iso" % tau] = {
                    "wilcoxon": paired_wilcoxon_pratt(cd, ci),
                    "coll_delayed_mean": float(np.mean(cd)), "coll_iso_mean": float(np.mean(ci))}
    return tests, ref

def report(out, tests, ref):
    print()
    for wname in ("D0", "DbigPlus"):
        print("--- 世界 %s（記述） ---" % wname)
        print("%-10s %8s %8s %8s %8s %6s %8s" % (
            "τ", "S̄pre平均", "S̄pre中央", "S>0シード", "衝突Reso", "衝突Iso", "絶対削減度"))
        for tau in TAUS:
            k = ref["%s_tau%d" % (wname, tau)]
            print("%-10s %8.4f %8.4f %8d %8.1f %6.1f %+8.4f" % (
                "tau=%d" % tau, k["S_pre_mean_of_means"], k["S_pre_median_of_means"],
                k["n_seeds_S_pos"], k["coll_reso_mean"], k["coll_iso_mean"],
                k["abs_reduction_mean"]))
    print()
    print("=== 主要検定（事前登録・2本のみ・判定は凍結§4/§6の基準のみ） ===")
    P1 = tests["P1_D0_Spre_400_vs_25"]
    print("[P1] D0 S̄_pre: τ=400 vs τ=25")
    print("     τ=400: %s / τ=25: %s" % (
        iqr_str(tests["P1_data"]["sp400"]), iqr_str(tests["P1_data"]["sp25"])))
    if P1 is None:
        print("     全ペア差分ゼロ: 検定不能")
    else:
        print("     Wilcoxon(pratt) W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.4f (参考t p=%.5f)" % (
            P1["W"], P1["p"], P1["p_holm"], P1["rank_biserial"], P1["mean_diff"], P1["t_p_ref"]))
    P2 = tests["P2_D0_Reso25_vs_Isolated_26_1000"]
    print("[P2] D0 ResoExp(τ=25) vs Isolated 衝突ep数[26,1000]")
    print("     S>0だったシード数(τ=25): %d / %d" % (tests["P2_n_seeds_S_pos_tau25"], N_SEEDS))
    if P2 == "MEASUREMENT_IMPOSSIBLE":
        print("     全シードS=0: P2は測定不能と報告")
    elif P2 is None:
        print("     全ペア差分ゼロ: 検定不能")
    else:
        print("     ResoExp: %s / Isolated: %s" % (
            iqr_str(tests["P2_data"]["reso25"]), iqr_str(tests["P2_data"]["iso"])))
        print("     Wilcoxon(pratt) W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.4f (参考t p=%.5f)" % (
            P2["W"], P2["p"], P2["p_holm"], P2["rank_biserial"], P2["mean_diff"], P2["t_p_ref"]))
    print()
    print("--- 参考: Delayed vs Isolated（Dbig+・探索的・主張しない） ---")
    for tau in TAUS_DELAYED:
        k = ref["Delayed_tau%d_vs_iso" % tau]; wv = k["wilcoxon"]
        if wv is None:
            print("  τ=%d: 差分全ゼロ" % tau)
        else:
            print("  τ=%d: Delayed%.1f vs Iso%.1f p=%.4f rb=%.3f" % (
                tau, k["coll_delayed_mean"], k["coll_iso_mean"], wv["p"], wv["rank_biserial"]))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if mode == "diag":
        diag()
    else:
        out, t0 = full()
        tests, ref = analyze(out)
        out["_tests"] = tests; out["_reference_not_primary"] = ref
        json.dump(out, open(os.path.join(BASE,
            'resonance_v05_sweep_results_30seed.json'), 'w'), ensure_ascii=False)
        print("JSON保存完了: resonance_v05_sweep_results_30seed.json (%.0f秒)" % (time.time()-t0))
        report(out, tests, ref)
        print()
        print("判定はDESIGN_幼年期スイープ_v04_凍結.mdの事前登録基準のみで行うこと。")
