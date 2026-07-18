# -*- coding: utf-8 -*-
"""
認識と傷の分離実験 v08 本実行（言葉の痛覚化・用量反応）
事前登録: DESIGN_認識と傷の分離_v02.md（§11凍結追記まで・2026年7月18日凍結・第2R 3AI全員同意）
条件: Iso / η=0(配線確認) / η=0.03,0.05(記述専用) / η=0.25,0.50,1.0(本体)
主要検定: P1 = η1.0 vs Iso（受信後衝突・確認的） / P2 = η1.0 vs η0.25（用量）
使い方: python resonance_v08_eta.py ut / full
"""
import sys, json, os, time
import numpy as np
from scipy import stats as spstats
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from resonance_v07_zerolr import (Agent, ResonanceEnv, PATROL_B_D0, learn_A, sim_entry,
    ETA_RES, argmax_rt, EPSILON, N_ACTIONS, ACTIONS, move, MAX_STEPS, LR, GAMMA,
    paired_wilcoxon_pratt, holm_2, iqr_str)

TAU = 150
N_EP = 1000
SEEDS = list(range(60, 90))
THRESH = 0.05
ETAS_MAIN = [0.25, 0.50, 1.0]
ETAS_DESC = [0.03, 0.05]
CONDS = ["Iso", "eta0.00", "eta0.03", "eta0.05", "eta0.25", "eta0.50", "eta1.00"]

def run_eps_watch(agent, env, n_eps, st, learn=True, watch=None, death_log=None):
    """v07 run_epsの複製＋傷の寿命フック（凍結§11定義: 書き込みから削除条件<0.05を初めて満たすまでのep数）。
    フックはpm参照のみで乱数を消費しない。"""
    for ep in range(n_eps):
        state = env.reset(ep)
        ep_coll = False
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
            if learn:
                best = np.max(agent.get_Q(ns)) if not done else 0.0
                agent.get_Q(s)[a] += LR*(r + GAMMA*best - agent.get_Q(s)[a])
            if info["collided"]:
                ep_coll = True
                intended = move(s[0], ACTIONS[a])
                agent.record_pain(s, intended)
            if info["success"]: pass
            state = ns
            if done: break
        agent.decay()
        st["coll_flags"].append(1 if ep_coll else 0)
        if watch:
            for k in list(watch):
                if k not in agent.pm:
                    death_log[repr(k)] = ep + 1
                    watch.discard(k)

def transfer_eta(A_pm, B, eta):
    """凍結§4: I_eff = η×(intensity_A×η_res×S)。閾値0.05はI_effに適用。η=1.0は従来仕様と一致。
    S算出は決定論的（rng非使用・コードレビュー済み）。"""
    own = list(B.archive.keys())
    S_pre, written = [], []
    for k, inten in A_pm.items():
        S = max([sim_entry(k, kb) for kb in own], default=0.0)
        S_pre.append(S)
        I_eff = eta * (inten * ETA_RES * S)
        if I_eff > THRESH:
            if k in B.pm:
                B.pm[k][0] = max(B.pm[k][0], I_eff)
                B.pm[k][2] = max(B.pm[k][2], S)
            else:
                B.pm[k] = [I_eff, False, S]
            written.append(k)
    return S_pre, written

def run_cond(cond, seed, A_pm):
    B = Agent(seed + 1000)
    envB = ResonanceEnv(PATROL_B_D0)
    st = {"coll_flags": [], "succ": []}
    run_eps_watch(B, envB, TAU, st, learn=True)
    pre_coll = int(sum(st["coll_flags"]))
    S_pre, written = [], []
    if cond != "Iso":
        eta = float(cond.replace("eta", ""))
        S_pre, written = transfer_eta(A_pm, B, eta)
    watch = set(written)
    death_log = {}
    st2 = {"coll_flags": [], "succ": []}
    run_eps_watch(B, envB, N_EP - TAU, st2, learn=True, watch=watch, death_log=death_log)
    n_promoted = sum(1 for k in written if k in B.pm and B.pm[k][1])
    return {"cond": cond, "seed": seed,
            "n_presented": len(S_pre),
            "S_pre_mean": float(np.mean(S_pre)) if S_pre else None,
            "n_written": len(written),
            "post_coll": int(sum(st2["coll_flags"])),
            "pre_coll": pre_coll,
            "coll_flags_post": st2["coll_flags"],
            "lifetimes": sorted(death_log.values()),
            "n_survived": len(watch), "n_promoted": int(n_promoted),
            "arch_end": len(B.archive)}

def qa(r, expected):
    exp = 0 if r["cond"] == "Iso" else expected
    if r["n_presented"] != exp:
        print("!!! QA不一致: %s seed%d 提示%d 期待%d → 停止" % (r["cond"], r["seed"], r["n_presented"], exp))
        sys.exit(1)

def jonckheere(groups):
    """JT検定（昇順用量で衝突減少を予測→降順傾向）。正規近似・記述用。"""
    k = len(groups)
    J = 0.0
    for i in range(k):
        for j in range(i+1, k):
            for x in groups[i]:
                for y in groups[j]:
                    J += (1.0 if y > x else (0.5 if y == x else 0.0))
    ns = [len(g) for g in groups]; N = sum(ns)
    mean = (N*N - sum(n*n for n in ns)) / 4.0
    var = (N*N*(2*N+3) - sum(n*n*(2*n+3) for n in ns)) / 72.0
    z = (J - mean) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - spstats.norm.cdf(abs(z)))
    return {"J": float(J), "z": float(z), "p_two_sided": float(p)}

def ut():
    print("=== UT-E1: 設計により真の機械検証（seed60 Iso vs η=0・受信後衝突列の全一致） ===")
    A_pm = learn_A(60)
    r_iso = run_cond("Iso", 60, A_pm)
    r_e0 = run_cond("eta0.00", 60, A_pm)
    if r_iso["coll_flags_post"] != r_e0["coll_flags_post"]:
        print("!!! UT-E1不合格: 軌道分岐＝乱数消費の混入 → 停止"); sys.exit(1)
    print("合格: 850ep全一致（受信後衝突 %d=%d）" % (r_iso["post_coll"], r_e0["post_coll"]))
    print("=== UT-E2: 従来仕様との一致（seed60 η=1.0） ===")
    r1 = run_cond("eta1.00", 60, A_pm)
    print("合格: n_written=%d (≥1)" % r1["n_written"] if r1["n_written"] >= 1 else "!!! 不合格")
    if r1["n_written"] < 1: sys.exit(1)
    print("=== UT-E3: 用量配線（seed60・単調非減少・配線確認であり理論検証ではない） ===")
    ns = []
    for cond in ["eta0.00", "eta0.03", "eta0.05", "eta0.25", "eta0.50", "eta1.00"]:
        ns.append(run_cond(cond, 60, A_pm)["n_written"])
    print("n_written:", ns)
    if any(ns[i] > ns[i+1] for i in range(len(ns)-1)):
        print("!!! UT-E3不合格: 単調性の破れ"); sys.exit(1)
    print("合格")
    print("=== UT全合格 ===")

def full():
    t0 = time.time()
    out = {}
    print("=== 本実行 7条件×30シード（新シード60-89） ===")
    A_cache = {s: learn_A(s) for s in SEEDS}
    for cond in CONDS:
        runs = []
        for seed in SEEDS:
            r = run_cond(cond, seed, dict(A_cache[seed]))
            qa(r, len(A_cache[seed]))
            runs.append(r)
        out[cond] = runs
        print("  %s 完了 (%.0f秒)" % (cond, time.time() - t0))
    return out, t0

def analyze(out):
    iso = [r["post_coll"] for r in out["Iso"]]
    e025 = [r["post_coll"] for r in out["eta0.25"]]
    e050 = [r["post_coll"] for r in out["eta0.50"]]
    e100 = [r["post_coll"] for r in out["eta1.00"]]
    P1 = paired_wilcoxon_pratt(e100, iso)      # 予測: 負（η1.0の方が少ない）
    P2 = paired_wilcoxon_pratt(e100, e025)     # 予測: 負
    if P1 is not None and P2 is not None:
        p1h, p2h = holm_2(P1["p"], P2["p"])
        P1["p_holm"] = p1h; P2["p_holm"] = p2h
    aux_jt = jonckheere([iso, e025, e050, e100])
    desc = {}
    for cond in CONDS:
        rs = out[cond]
        lts = [lt for r in rs for lt in r["lifetimes"]]
        desc[cond] = {
            "post_coll_mean": float(np.mean([r["post_coll"] for r in rs])),
            "post_coll_median": float(np.median([r["post_coll"] for r in rs])),
            "n_written_median": float(np.median([r["n_written"] for r in rs])),
            "write_rate": float(np.mean([r["n_written"]/r["n_presented"] for r in rs if r["n_presented"]>0])) if cond != "Iso" else None,
            "lifetime_median": float(np.median(lts)) if lts else None,
            "n_survived_mean": float(np.mean([r["n_survived"] for r in rs])),
            "n_promoted_mean": float(np.mean([r["n_promoted"] for r in rs])),
            "S_pre_mean": float(np.mean([r["S_pre_mean"] for r in rs])) if cond != "Iso" else None}
    return {"P1": P1, "P2": P2, "aux_jt": aux_jt,
            "data": {"iso": iso, "e025": e025, "e050": e050, "e100": e100}}, desc

def report(tests, desc):
    print()
    print("--- 記述（条件別・30シード） ---")
    print("%-9s %10s %10s %8s %8s %8s %8s" % ("条件", "受信後衝突平均", "同中央値", "書込中央", "書込率", "寿命中央", "実体験化"))
    for cond in CONDS:
        d = desc[cond]
        print("%-9s %10.1f %10.1f %8s %8s %8s %8.1f" % (
            cond, d["post_coll_mean"], d["post_coll_median"],
            "%.0f" % d["n_written_median"],
            ("%.2f" % d["write_rate"]) if d["write_rate"] is not None else "-",
            ("%.0f" % d["lifetime_median"]) if d["lifetime_median"] is not None else "-",
            d["n_promoted_mean"]))
    print()
    print("=== 主要検定（事前登録・2本のみ・ホルム補正） ===")
    P1, P2 = tests["P1"], tests["P2"]
    print("[P1・確認的] 受信後衝突: η1.0 %s vs Iso %s" % (
        iqr_str(tests["data"]["e100"]), iqr_str(tests["data"]["iso"])))
    if P1 is None: print("     全ペア差分ゼロ: 検定不能")
    else: print("     W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.2f" % (
        P1["W"], P1["p"], P1["p_holm"], P1["rank_biserial"], P1["mean_diff"]))
    print("[P2・用量] 受信後衝突: η1.0 %s vs η0.25 %s" % (
        iqr_str(tests["data"]["e100"]), iqr_str(tests["data"]["e025"])))
    if P2 is None: print("     全ペア差分ゼロ: 検定不能")
    else: print("     W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.2f" % (
        P2["W"], P2["p"], P2["p_holm"], P2["rank_biserial"], P2["mean_diff"]))
    print("[補助・記述] JT順序性(Iso→0.25→0.50→1.0): z=%.2f p=%.5f（主要検定に数えない）" % (
        tests["aux_jt"]["z"], tests["aux_jt"]["p_two_sided"]))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ut"
    if mode == "ut":
        ut()
    else:
        out, t0 = full()
        tests, desc = analyze(out)
        out["_tests"] = tests; out["_desc"] = desc
        json.dump(out, open(os.path.join(BASE,
            'resonance_v08_eta_results_30seed.json'), 'w'), ensure_ascii=False)
        print("JSON保存完了 (%.0f秒)" % (time.time() - t0))
        report(tests, desc)
        print()
        print("判定はDESIGN_認識と傷の分離_v02.md（凍結版）の事前登録基準のみで行うこと。")
