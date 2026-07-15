# -*- coding: utf-8 -*-
"""
遅延丸呑みの害曲線 v06 本実行（確認的再現・新シード30-59）
事前登録: DESIGN_遅延丸呑み害曲線_v03_凍結.md（2026年7月15日凍結・2ラウンド監査完了）
主要検定（2本のみ・ホルム・Dbig+）:
 P1 = FullCopyDelayed(τ=25) vs Isolated 衝突ep数[26,1000]（害=Delayedが多い）
 P2 = excess(τ=25) > excess(τ=400)  excess(τ)=固定窓[τ+1,τ+600]の衝突ep数差(FCD−Iso)
頑健性(判定に使わない): excess_norm(τ)=excess/(600−C_Iso)
使い方: python resonance_v06_harmcurve.py diag | full
"""
import sys, os, json, time
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
from scipy import stats as spstats
BASE = os.path.dirname(os.path.abspath(__file__))
V05 = os.path.join(BASE, 'resonance_v05_sweep.py')
g = {'__name__': 'v05lib', '__file__': V05}
exec(open(V05, encoding='utf-8').read(), g)

# --- 最小差分パッチ: run_episodes にエピソード別ステップ数記録を追加（力学は完全同一） ---
PATCH = '''
def new_stats():
    return {"rewards": [], "succ": [], "route": [], "coll_pos": [], "coll_flags": [], "steps": []}

def run_episodes(agent, env, n_eps, stats, snaps=None, snap_offset=0):
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
        stats["steps"].append(t + 1)
        if snaps is not None and (ep + 1) % 100 == 0:
            snaps.append([snap_offset + ep + 1, mid_route_q(agent)])

_orig_pack = pack
def pack(st, received, S_pre, S_acc, q_snaps):
    r = _orig_pack(st, received, S_pre, S_acc, q_snaps)
    r["steps"] = st["steps"]
    return r
'''
exec(PATCH, g)

SEEDS = list(range(30, 60))       # 新シード30-59（凍結§2）
TAUS6 = [0, 25, 50, 100, 200, 400]
FW = 600
run_B = g['run_B_sweep']; learn_A = g['learn_A']; qa = g['qa_or_die']
coll_int = g['coll_interval']; pw = g['paired_wilcoxon_pratt']; holm2 = g['holm_2']; iqr = g['iqr_str']
PATROL = g['PATROL_B_DBIGPLUS']

def diag():
    print("=== 診断（seed30・構造チェックのみ） ===")
    A = learn_A(30); na = len(A)
    print("Aアーカイブ件数:", na)
    for tau in (0, 25, 400):
        r = run_B("FullCopyDelayed", PATROL, dict(A), 30, tau)
        qa(r, na, "diag FCD tau=%d" % tau, need_full_receive=True)
        fw_c = coll_int(r["coll_flags"], tau+1, tau+FW)
        print("FCD τ=%d: 受信%d==na QA合格 衝突ep[τ+1,1000]=%d 固定窓=%d flags長=%d steps長=%d" % (
            tau, r["received"], coll_int(r["coll_flags"], tau+1, 1000), fw_c,
            len(r["coll_flags"]), len(r["steps"])))
    rI = run_B("Isolated", PATROL, dict(A), 30, 0)
    print("Isolated: 衝突ep[1,1000]=%d 成功率=%.2f" % (
        coll_int(rI["coll_flags"], 1, 1000), rI["succ_last100"]))
    print("=== 診断: 全チェック合格 ===")

def full():
    t0 = time.time()
    print("=== (1) A学習 シード30-59 ===")
    A_cache = {s: learn_A(s) for s in SEEDS}
    counts = {s: len(A_cache[s]) for s in SEEDS}
    print("Aアーカイブ: min%d max%d" % (min(counts.values()), max(counts.values())))
    out = {"A_archive_counts": {str(k): v for k, v in counts.items()}, "DbigPlus": {}}
    print("=== (2) 本実行 ===")
    out["DbigPlus"]["Isolated"] = []
    for s in SEEDS:
        out["DbigPlus"]["Isolated"].append(run_B("Isolated", PATROL, dict(A_cache[s]), s, 0))
    print("  Isolated 完了 (%.0f秒)" % (time.time()-t0))
    for tau in TAUS6:
        key = "FullCopyDelayed_%d" % tau
        runs = []
        for s in SEEDS:
            r = run_B("FullCopyDelayed", PATROL, dict(A_cache[s]), s, tau)
            qa(r, counts[s], "%s seed%d" % (key, s), need_full_receive=True)
            runs.append(r)
        out["DbigPlus"][key] = runs
        print("  %s 完了 (%.0f秒)" % (key, time.time()-t0))
    return out, t0

def analyze(out):
    w = out["DbigPlus"]; iso = w["Isolated"]
    # P1: [26,1000] FCD(25) vs Iso（害=FCDが多い→mean_diff>0）
    c_f25 = [coll_int(r["coll_flags"], 26, 1000) for r in w["FullCopyDelayed_25"]]
    c_i25 = [coll_int(r["coll_flags"], 26, 1000) for r in iso]
    P1 = pw(c_f25, c_i25)
    # P2: excess(25) vs excess(400)（固定窓・凍結§3）
    def excess(tau):
        fcd = w["FullCopyDelayed_%d" % tau]
        return [coll_int(a["coll_flags"], tau+1, tau+FW) - coll_int(b["coll_flags"], tau+1, tau+FW)
                for a, b in zip(fcd, iso)]
    e25, e400 = excess(25), excess(400)
    P2 = pw(e25, e400)
    if P1 is not None and P2 is not None:
        p1h, p2h = holm2(P1["p"], P2["p"]); P1["p_holm"] = p1h; P2["p_holm"] = p2h
    # 記述: 害曲線・excess_norm・固定窓平均ステップ
    ref = {}
    for tau in TAUS6:
        ex = excess(tau)
        en = [e / (FW - coll_int(b["coll_flags"], tau+1, tau+FW)) for e, b in zip(ex, iso)]
        st_f = [float(np.mean(a["steps"][tau:tau+FW])) for a in w["FullCopyDelayed_%d" % tau]]
        st_i = [float(np.mean(b["steps"][tau:tau+FW])) for b in iso]
        ref["tau%d" % tau] = {"excess_mean": float(np.mean(ex)), "excess_median": float(np.median(ex)),
            "excess_norm_mean": float(np.mean(en)),
            "steps_fcd_mean": float(np.mean(st_f)), "steps_iso_mean": float(np.mean(st_i)),
            "coll_full_mean": float(np.mean([coll_int(a["coll_flags"], tau+1, 1000) for a in w["FullCopyDelayed_%d" % tau]])),
            "coll_iso_mean": float(np.mean([coll_int(b["coll_flags"], tau+1, 1000) for b in iso]))}
    return {"P1": P1, "P2": P2, "P1_data": {"fcd25": c_f25, "iso": c_i25},
            "P2_data": {"e25": e25, "e400": e400}}, ref

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if mode == "diag":
        diag(); sys.exit()
    out, t0 = full()
    tests, ref = analyze(out)
    out["_tests"] = tests; out["_reference_not_primary"] = ref
    json.dump(out, open(os.path.join(BASE, 'resonance_v06_harmcurve_results_30seed.json'), 'w'), ensure_ascii=False)
    print("JSON保存完了 (%.0f秒)" % (time.time()-t0))
    print()
    print("--- 記述: 害曲線（検定しない） ---")
    print("%-8s %10s %12s %10s %10s" % ("τ", "excess平均", "excess_norm", "FCD歩数", "Iso歩数"))
    for tau in TAUS6:
        k = ref["tau%d" % tau]
        print("%-8s %10.2f %12.5f %10.1f %10.1f" % ("tau=%d" % tau, k["excess_mean"],
            k["excess_norm_mean"], k["steps_fcd_mean"], k["steps_iso_mean"]))
    print()
    print("=== 主要検定（凍結§3・§4の基準のみ） ===")
    P1, P2 = tests["P1"], tests["P2"]
    print("[P1] FCD(τ=25) vs Isolated 衝突ep数[26,1000]")
    print("     FCD: %s / Iso: %s" % (iqr(tests["P1_data"]["fcd25"]), iqr(tests["P1_data"]["iso"])))
    if P1 is None: print("     全ペア差分ゼロ: 検定不能")
    else: print("     W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.2f (参考t p=%.5f)" % (
        P1["W"], P1["p"], P1["p_holm"], P1["rank_biserial"], P1["mean_diff"], P1["t_p_ref"]))
    print("[P2] excess(τ=25) vs excess(τ=400) 固定窓600ep")
    print("     e25: %s / e400: %s" % (iqr(tests["P2_data"]["e25"]), iqr(tests["P2_data"]["e400"])))
    if P2 is None: print("     全ペア差分ゼロ: 検定不能")
    else: print("     W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.2f (参考t p=%.5f)" % (
        P2["W"], P2["p"], P2["p_holm"], P2["rank_biserial"], P2["mean_diff"], P2["t_p_ref"]))
    print()
    print("判定はDESIGN_遅延丸呑み害曲線_v03_凍結.mdの事前登録基準のみで行うこと。")
