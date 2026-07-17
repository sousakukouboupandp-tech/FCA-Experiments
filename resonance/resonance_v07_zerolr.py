# -*- coding: utf-8 -*-
"""
学習率ゼロベースライン実験 v07 本実行（成熟vs被覆の分離）
事前登録: DESIGN_学習率ゼロ基線_v02.md（2026年7月17日凍結・第2R監査4AI全員同意）
主要検定（2本のみ・ホルム補正・D0・新シード30-59）:
 P1(被覆単独の非飽和) = ZeroChild: S̄_pre(τ=400) vs S̄_pre(τ=25)
 P2(綱引きの向き)     = τ=400: ZeroChild S̄_pre vs LearnChild S̄_pre
天井規程: ZeroChild τ=25で S̄_pre>=0.95 が15/30以上 → P1は測定不能と報告(L1(a)不発動)
使い方: python resonance_v07_zerolr.py ut / diag / full
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

ETA_RES = 1.0
LAMBDA_OWN = 0.97
LAMBDA_HEAR_BASE = 0.94
TABOO_TH = 0.3
TAUS_Z = [10, 25, 50, 100, 200, 400]
N_EP = 1000
SEEDS = list(range(30, 60))   # 新シード（凍結§2・循環回避）
CEIL_TH = 0.95
CEIL_N = 15

def argmax_rt(qvals, rng):
    finite = np.isfinite(qvals)
    if not finite.any(): return rng.randint(len(qvals))
    mv = np.max(qvals[finite]); cand = np.where((qvals==mv)&finite)[0]
    return rng.choice(cand)

def sim_entry(eA, eB):
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
        self.pm = {}
        self.archive = {}
    def get_Q(self, s):
        if s not in self.Q: self.Q[s] = np.zeros(N_ACTIONS)
        return self.Q[s]
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
            self.pm[k][0] += 1.0; self.pm[k][1] = True; self.pm[k][2] = 1.0
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

def run_eps(agent, env, n_eps, st, learn=True, track=None):
    """v05のrun_episodesを継承。learn=FalseでTD更新1行のみスキップ（凍結§2）。
    track指定時: 訪問位置/状態・総歩数・毎ep傷跡帳件数を記録（凍結§3）。"""
    for ep in range(n_eps):
        state = env.reset(ep)
        ep_coll, total_r, ep_succ = False, 0.0, False
        for t in range(MAX_STEPS):
            s = state
            if track is not None:
                track["pos"].add(s[0]); track["st"].add(s); track["steps"] += 1
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
            if learn:
                best = np.max(agent.get_Q(ns)) if not done else 0.0
                agent.get_Q(s)[a] += LR*(r + GAMMA*best - agent.get_Q(s)[a])
            if info["collided"]:
                ep_coll = True
                intended = move(s[0], ACTIONS[a])
                agent.record_pain(s, intended)
            if info["success"]: ep_succ = True
            state = ns
            if done: break
        agent.decay()
        st["coll_flags"].append(1 if ep_coll else 0)
        st["succ"].append(1.0 if ep_succ else 0.0)
        if track is not None:
            track["arch_traj"].append(len(agent.archive))

def transfer(A_archive, B):
    own = list(B.archive.keys())
    n_received = 0
    S_presented, S_accepted = [], []
    for k, inten in A_archive.items():
        S = max([sim_entry(k, kb) for kb in own], default=0.0)
        I = inten * ETA_RES * S
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

def learn_A(seed):
    A = Agent(seed)
    envA = ResonanceEnv(PATROL_WORLD_A)
    sA = {"coll_flags": [], "succ": []}
    run_eps(A, envA, N_EP, sA, learn=True)
    return dict(A.archive)

def run_B(cond, seed, tau, A_pm):
    """条件×τの1本。D0のみ。幼年期learn=ZeroChildでFalse、受信後は両条件learn=True（凍結§2）。"""
    B = Agent(seed + 1000)
    envB = ResonanceEnv(PATROL_B_D0)
    st = {"coll_flags": [], "succ": []}
    track = {"pos": set(), "st": set(), "steps": 0, "arch_traj": []}
    run_eps(B, envB, tau, st, learn=(cond == "LearnChild"), track=track)
    rx = {"arch_at_rx": len(B.archive), "uniq_pos_at_rx": len(track["pos"]),
          "uniq_state_at_rx": len(track["st"]), "steps_at_rx": track["steps"],
          "child_coll_eps": int(sum(st["coll_flags"]))}
    received, S_pre, S_acc = transfer(A_pm, B)
    run_eps(B, envB, N_EP - tau, st, learn=True, track=track)
    return {"cond": cond, "tau": tau, "seed": seed,
            "n_presented": len(S_pre),
            "n_S_pos": int(sum(1 for s in S_pre if s > 0)),
            "S_pre_mean": float(np.mean(S_pre)) if S_pre else None,
            "S_acc_mean": float(np.mean(S_acc)) if S_acc else None,
            "S_pre_raw": [float(s) for s in S_pre],
            "coll_flags": st["coll_flags"], "rx": rx,
            "arch_traj": track["arch_traj"]}

def qa_or_die(r, expected, label):
    if r["n_presented"] != expected:
        print("!!! QA不一致(提示件数): %s 提示%d 期待%d → 停止" % (label, r["n_presented"], expected))
        sys.exit(1)

def paired_wilcoxon_pratt(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    d = x - y
    if np.all(d == 0): return None
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
    if p1 is None or p2 is None: return p1, p2
    if p1 <= p2:
        a1 = min(1.0, 2*p1); a2 = min(1.0, max(a1, p2))
    else:
        a2 = min(1.0, 2*p2); a1 = min(1.0, max(a2, p1))
    return a1, a2

def iqr_str(vals):
    a = np.array(vals, dtype=float)
    return "中央値%.4f IQR[%.4f-%.4f]" % (np.median(a), np.percentile(a,25), np.percentile(a,75))

def ut():
    """UT-Z1〜Z4（凍結§6）。不合格時の修正は仕様一致目的のバグ修正のみ。"""
    print("=== UT-Z1: 学習停止の配線確認（seed30 ZeroChild τ=25） ===")
    B = Agent(30 + 1000); envB = ResonanceEnv(PATROL_B_D0)
    st = {"coll_flags": [], "succ": []}
    run_eps(B, envB, 25, st, learn=False)
    bad = [s for s, q in B.Q.items() if not np.all(q == 0.0)]
    if bad:
        print("!!! UT-Z1不合格: 非ゼロQが%d状態 → 配線破損" % len(bad)); sys.exit(1)
    print("合格: 全%d訪問状態でQ厳密ゼロ" % len(B.Q))
    print("=== UT-Z2: 傷の記録継続（同run） ===")
    n_arch = len(B.archive)
    if n_arch >= 1:
        print("合格: 傷跡帳%d件・幼年期衝突ep=%d" % (n_arch, int(sum(st["coll_flags"]))))
    else:
        print("0件 → 凍結§6の規程どおり別シード診断を実施")
        B2 = Agent(31 + 1000); st2 = {"coll_flags": [], "succ": []}
        run_eps(B2, ResonanceEnv(PATROL_B_D0), 25, st2, learn=False)
        print("seed31: 傷跡帳%d件（0件が続く場合record_pain経路を点検し停止）" % len(B2.archive))
        if len(B2.archive) == 0: sys.exit(1)
    print("=== UT-Z3: LearnChild配線確認（seed30 τ=25 learn=True） ===")
    BL = Agent(30 + 1000); stL = {"coll_flags": [], "succ": []}
    run_eps(BL, ResonanceEnv(PATROL_B_D0), 25, stL, learn=True)
    nz = sum(1 for q in BL.Q.values() if not np.all(q == 0.0))
    print("合格（機能確認）: 学習ONで非ゼロQが%d状態（更新経路は生きている）" % nz if nz > 0
          else "!!! UT-Z3不合格: 学習ONでQが全ゼロ")
    if nz == 0: sys.exit(1)
    print("=== UT-Z4: 一様性の配線確認（Q≡0で1000回選択・分布をログ保存） ===")
    rng = np.random.RandomState(777)
    counts = np.zeros(N_ACTIONS, dtype=int)
    for _ in range(1000):
        counts[argmax_rt(np.zeros(N_ACTIONS), rng)] += 1
    print("行動別選択回数:", counts.tolist(), "（期待値 各%.0f・配線確認であり検定ではない）"
          % (1000.0/N_ACTIONS))
    json.dump({"utz4_counts": counts.tolist()},
              open(os.path.join(BASE, 'v07_utz4_log.json'), 'w'))
    if counts.min() == 0:
        print("!!! UT-Z4不合格: 選ばれない行動が存在 → argmax_rt配線点検"); sys.exit(1)
    print("=== UT全合格 ===")

def diag():
    """単一シード診断（seed30・統計なし・構造チェックのみ）"""
    print("=== 診断モード（seed30） ===")
    A_pm = learn_A(30); na = len(A_pm)
    print("Aアーカイブ件数:", na)
    res = {}
    for cond in ("ZeroChild", "LearnChild"):
        for tau in (25, 400):
            r = run_B(cond, 30, tau, dict(A_pm))
            qa_or_die(r, na, "diag %s tau=%d" % (cond, tau))
            res[(cond, tau)] = r
            print("%s τ=%d: S̄_pre=%.4f S>0件=%d 傷跡帳=%d 訪問位置=%d 幼年期衝突ep=%d" % (
                cond, tau, r["S_pre_mean"], r["n_S_pos"], r["rx"]["arch_at_rx"],
                r["rx"]["uniq_pos_at_rx"], r["rx"]["child_coll_eps"]))
    for cond in ("ZeroChild", "LearnChild"):
        if res[(cond, 400)]["S_pre_mean"] < res[(cond, 25)]["S_pre_mean"] - 1e-12:
            print("!!! 単調非減少の破れ（%s）→ 実装バグ。停止" % cond); sys.exit(1)
    allsame = all(res[k]["S_pre_mean"] == res[(("ZeroChild",25))]["S_pre_mean"] for k in res)
    if allsame:
        print("注意: 4条件のS̄_preが完全一致 → 実装バグ疑い（3手法同一結果の教訓）")
    print("=== 診断: 合格 ===")

def full():
    t0 = time.time()
    print("=== (1) 語り手A学習 新シード30-59 ===")
    A_cache = {s: learn_A(s) for s in SEEDS}
    A_counts = {s: len(A_cache[s]) for s in SEEDS}
    print("Aアーカイブ件数: min%d max%d" % (min(A_counts.values()), max(A_counts.values())))
    out = {"A_archive_counts": {str(k): v for k, v in A_counts.items()}}
    print("=== (2) 本実行 2条件×6τ×30シード ===")
    for cond in ("LearnChild", "ZeroChild"):
        for tau in TAUS_Z:
            key = "%s_%d" % (cond, tau)
            runs = []
            for seed in SEEDS:
                r = run_B(cond, seed, tau, dict(A_cache[seed]))
                qa_or_die(r, A_counts[seed], "%s seed%d" % (key, seed))
                runs.append(r)
            out[key] = runs
            print("  %s 完了 (%.0f秒)" % (key, time.time() - t0))
    return out, t0

def analyze(out):
    z25 = [r["S_pre_mean"] for r in out["ZeroChild_25"]]
    z400 = [r["S_pre_mean"] for r in out["ZeroChild_400"]]
    l400 = [r["S_pre_mean"] for r in out["LearnChild_400"]]
    n_ceil = sum(1 for v in z25 if v >= CEIL_TH)
    ceiling = (n_ceil >= CEIL_N)
    P1 = paired_wilcoxon_pratt(z400, z25)
    P2 = paired_wilcoxon_pratt(z400, l400)
    if P1 is not None and P2 is not None:
        p1h, p2h = holm_2(P1["p"], P2["p"])
        P1["p_holm"] = p1h; P2["p_holm"] = p2h
    tests = {"P1": P1, "P2": P2, "ceiling_n": n_ceil, "ceiling": ceiling,
             "data": {"z25": z25, "z400": z400, "l400": l400}}
    desc = {}
    for cond in ("LearnChild", "ZeroChild"):
        for tau in TAUS_Z:
            rs = out["%s_%d" % (cond, tau)]
            desc["%s_%d" % (cond, tau)] = {
                "S_pre_mean": float(np.mean([r["S_pre_mean"] for r in rs])),
                "S_pre_median": float(np.median([r["S_pre_mean"] for r in rs])),
                "n_seeds_S_pos": int(sum(1 for r in rs if r["n_S_pos"] > 0)),
                "arch_at_rx": float(np.mean([r["rx"]["arch_at_rx"] for r in rs])),
                "uniq_pos": float(np.mean([r["rx"]["uniq_pos_at_rx"] for r in rs])),
                "steps": float(np.mean([r["rx"]["steps_at_rx"] for r in rs])),
                "child_coll": float(np.mean([r["rx"]["child_coll_eps"] for r in rs]))}
    return tests, desc

def report(tests, desc):
    print()
    print("--- 記述（各条件×τ・30シード平均） ---")
    print("%-16s %8s %8s %8s %8s %8s %8s" % ("条件_τ", "S̄pre平均", "S̄pre中央", "傷跡帳", "訪問位置", "総歩数", "幼年衝突"))
    for cond in ("LearnChild", "ZeroChild"):
        for tau in TAUS_Z:
            d = desc["%s_%d" % (cond, tau)]
            print("%-16s %8.4f %8.4f %8.1f %8.1f %8.0f %8.1f" % (
                "%s_%d" % (cond[:5], tau), d["S_pre_mean"], d["S_pre_median"],
                d["arch_at_rx"], d["uniq_pos"], d["steps"], d["child_coll"]))
    print()
    print("=== 天井規程（凍結§5・判定より先に確認） ===")
    print("ZeroChild τ=25 S̄_pre>=%.2f のシード数: %d/30（基準%d本以上で天井）→ %s" % (
        CEIL_TH, tests["ceiling_n"], CEIL_N, "天井到達" if tests["ceiling"] else "非該当"))
    print()
    print("=== 主要検定（事前登録・2本のみ） ===")
    P1, P2 = tests["P1"], tests["P2"]
    print("[P1] ZeroChild S̄_pre: τ=400 %s vs τ=25 %s" % (
        iqr_str(tests["data"]["z400"]), iqr_str(tests["data"]["z25"])))
    if P1 is None: print("     全ペア差分ゼロ: 検定不能")
    else: print("     W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.4f" % (
        P1["W"], P1["p"], P1["p_holm"], P1["rank_biserial"], P1["mean_diff"]))
    if tests["ceiling"]:
        print("     ※天井到達につき凍結§5によりP1の結論は『測定不能』。L1(a)は発動しない。")
    print("[P2] τ=400: ZeroChild %s vs LearnChild %s" % (
        iqr_str(tests["data"]["z400"]), iqr_str(tests["data"]["l400"])))
    if P2 is None: print("     全ペア差分ゼロ: 検定不能")
    else: print("     W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.4f" % (
        P2["W"], P2["p"], P2["p_holm"], P2["rank_biserial"], P2["mean_diff"]))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if mode == "ut":
        ut()
    elif mode == "diag":
        diag()
    else:
        out, t0 = full()
        tests, desc = analyze(out)
        out["_tests"] = tests; out["_desc"] = desc
        json.dump(out, open(os.path.join(BASE,
            'resonance_v07_zerolr_results_30seed.json'), 'w'), ensure_ascii=False)
        print("JSON保存完了: resonance_v07_zerolr_results_30seed.json (%.0f秒)" % (time.time()-t0))
        report(tests, desc)
        print()
        print("判定はDESIGN_学習率ゼロ基線_v02.mdの事前登録基準のみで行うこと。")
