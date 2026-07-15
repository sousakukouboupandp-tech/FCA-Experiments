# -*- coding: utf-8 -*-
"""
遅延丸呑み害曲線 v06 本実行（確認的再現・新シード30-59）
事前登録: DESIGN_遅延丸呑み害曲線_v02_凍結.md（2026年7月15日凍結・2ラウンド監査・3AI全員同意）
方式: resonance_v05_sweep.py の関数群を全流用(exec)。構成変更はシード範囲・世界・条件のみ。
run_episodes/new_stats/pack/run_B_sweep をログ追加のため上書き（乱数消費・状態遷移への影響なし：
steps記録とlen(B.archive)読み取りは観測のみ）。
主要検定（2本・ホルム・Dbig+）:
 P1 = FullCopyDelayed(τ=25) vs Isolated 衝突ep数[26,1000]（害の方向=Fullが多い）
 P2 = excess(τ=25) > excess(τ=400)  excess(τ)=固定窓[τ+1,τ+600]の衝突ep数差(Full−Iso)
使い方: python resonance_v06_gulp.py diag | full
"""
import sys, os, json, time
_BASE6 = os.path.dirname(os.path.abspath(__file__))
_v05_code = open(os.path.join(_BASE6, 'resonance_v05_sweep.py'), encoding='utf-8').read()
_v05_code = _v05_code.split('if __name__')[0]  # v05のmain部を切り落として関数群のみ取り込む
exec(compile(_v05_code, 'resonance_v05_sweep.py', 'exec'), globals())

SEEDS6 = list(range(30, 60))          # 新シード（事前登録の最重要設計）
TAUS_G = [0, 25, 50, 100, 200, 400]   # τ=0=新生児丸呑み対照（記述のみ）

def new_stats():
    return {"rewards": [], "succ": [], "route": [], "coll_pos": [], "coll_flags": [], "steps": []}

def run_episodes(agent, env, n_eps, stats, snaps=None, snap_offset=0):
    """v05と同一＋各エピソードの実ステップ数を記録（観測のみ・乱数消費なし）"""
    for ep in range(n_eps):
        state = env.reset(ep)
        ep_coll, crossed, total_r = False, None, 0.0
        ep_succ = False
        t = -1
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

def pack(st, received, S_pre, S_acc, q_snaps):
    last = 100
    routes = [r for r in st["route"][-last:] if r]
    return {
        "coll_flags": st["coll_flags"],
        "steps": st["steps"],
        "succ_last100": float(np.mean(st["succ"][-last:])),
        "mid": routes.count("mid"), "top": routes.count("top"),
        "received": received,
        "n_presented": len(S_pre),
        "n_S_pos": int(sum(1 for s in S_pre if s > 0)),
        "S_pre_mean": (float(np.mean(S_pre)) if S_pre else None),
        "S_acc_mean": (float(np.mean(S_acc)) if S_acc else None),
        "q_snaps": q_snaps,
    }

def run_B_sweep(cond, patrol, A_pm, seed, tau):
    B = Agent(seed + 1000)
    envB = ResonanceEnv(patrol)
    st = new_stats()
    if cond == "Isolated":
        run_episodes(B, envB, N_EP_B, st)
        r = pack(st, 0, [], [], None); r["b_arch_at_recv"] = None
        return r
    if tau > 0:
        run_episodes(B, envB, tau, st)
    b_arch = len(B.archive)                    # 受信時のBの傷跡帳件数（記述・凍結§5）
    q_before = mid_route_q(B)
    gated = (cond == "ResoExp")
    received, S_pre, S_acc = transfer(A_pm, B, gated=gated)
    q_after = mid_route_q(B)
    snaps = [[tau, q_before], [tau, q_after]] if cond == "FullCopyDelayed" else None
    run_episodes(B, envB, N_EP_B - tau, st, snaps=snaps, snap_offset=tau)
    r = pack(st, received, S_pre, S_acc, snaps); r["b_arch_at_recv"] = b_arch
    return r

def fw_excess(fcd_run, iso_run, tau):
    """excess(τ) = 固定窓[τ+1, τ+600]の衝突エピソード数差（Full − Iso）凍結§3"""
    cf = coll_interval(fcd_run["coll_flags"], tau+1, tau+FIXED_WIN)
    ci = coll_interval(iso_run["coll_flags"], tau+1, tau+FIXED_WIN)
    return cf - ci, ci

def diag6():
    print("=== 診断モード（seed30・統計なし・構造チェックのみ） ===")
    A_pm = learn_A(30)
    na = len(A_pm)
    print("A(seed30)アーカイブ件数:", na)
    res = {}
    for tau in (25, 400):
        r = run_B_sweep("FullCopyDelayed", PATROL_B_DBIGPLUS, dict(A_pm), 30, tau)
        qa_or_die(r, na, "diag FCD tau=%d" % tau, need_full_receive=True)
        res[tau] = r
        print("Dbig+ FCD τ=%d: 受信%d==アーカイブ%d B受信時傷跡%d 衝突ep[τ+1,1000]=%d snaps点数=%d" % (
            tau, r["received"], na, r["b_arch_at_recv"],
            coll_interval(r["coll_flags"], tau+1, 1000), len(r["q_snaps"])))
    rI = run_B_sweep("Isolated", PATROL_B_DBIGPLUS, dict(A_pm), 30, 0)
    e25, _ = fw_excess(res[25], rI, 25)
    e400, _ = fw_excess(res[400], rI, 400)
    print("Dbig+ Isolated: 衝突ep[1,1000]=%d flags長=%d steps長=%d" % (
        coll_interval(rI["coll_flags"], 1, 1000), len(rI["coll_flags"]), len(rI["steps"])))
    print("excess(25)=%d excess(400)=%d（seed30単体・構造確認のみ）" % (e25, e400))
    for tau in (25, 400):
        assert len(res[tau]["coll_flags"]) == 1000 and len(res[tau]["steps"]) == 1000, "長さ異常"
    if res[25]["coll_flags"] == res[400]["coll_flags"]:
        print("注意: τ=25と400でflags完全一致。実装確認を"); sys.exit(1)
    print("=== 診断: 全チェック合格 ===")

def full6():
    t0 = time.time()
    print("=== (1) 語り手Aの学習 シード30-59 ===")
    A_cache, A_counts = {}, {}
    for seed in SEEDS6:
        A_cache[seed] = learn_A(seed)
        A_counts[seed] = len(A_cache[seed])
    print("Aアーカイブ件数: min%d max%d" % (min(A_counts.values()), max(A_counts.values())))
    out = {"A_archive_counts": {str(k): v for k, v in A_counts.items()}}
    print("=== (2) 本実行 Dbig+ ===")
    key_list = [("Isolated", None)] + [("FullCopyDelayed", tau) for tau in TAUS_G]
    for cond, tau in key_list:
        key = cond if tau is None else "%s_%d" % (cond, tau)
        runs = []
        for seed in SEEDS6:
            r = run_B_sweep(cond, PATROL_B_DBIGPLUS, dict(A_cache[seed]), seed,
                            0 if tau is None else tau)
            if cond == "FullCopyDelayed":
                qa_or_die(r, A_counts[seed], "%s seed%d" % (key, seed), need_full_receive=True)
            runs.append(r)
        out[key] = runs
        print("  %s 完了 (%.0f秒)" % (key, time.time()-t0))
    return out, t0

def analyze6(out):
    iso = out["Isolated"]
    # P1: FCD(25) vs Isolated 区間[26,1000]
    c_iso25 = [coll_interval(r["coll_flags"], 26, 1000) for r in iso]
    c_f25 = [coll_interval(r["coll_flags"], 26, 1000) for r in out["FullCopyDelayed_25"]]
    P1 = paired_wilcoxon_pratt(c_f25, c_iso25)  # mean_diff>0 = 害の方向
    # P2: excess(25) vs excess(400)
    exc = {}
    for tau in TAUS_G:
        vals = []
        for rF, rI in zip(out["FullCopyDelayed_%d" % tau], iso):
            e, ci = fw_excess(rF, rI, tau)
            vals.append((e, ci))
        exc[tau] = vals
    e25 = [v[0] for v in exc[25]]; e400 = [v[0] for v in exc[400]]
    P2 = paired_wilcoxon_pratt(e25, e400)  # mean_diff>0 = 25の方が害大
    if P1 is not None and P2 is not None:
        p1h, p2h = holm_2(P1["p"], P2["p"]); P1["p_holm"] = p1h; P2["p_holm"] = p2h
    tests = {"P1": P1, "P2": P2, "P1_data": {"f25": c_f25, "iso": c_iso25},
             "P2_data": {"e25": e25, "e400": e400}}
    ref = {}
    for tau in TAUS_G:
        es = [v[0] for v in exc[tau]]; cis = [v[1] for v in exc[tau]]
        en = [e / (FIXED_WIN - ci) for e, ci in exc[tau]]
        fcd = out["FullCopyDelayed_%d" % tau]
        fw_steps_f = [float(np.mean(r["steps"][tau:tau+FIXED_WIN])) for r in fcd]
        fw_steps_i = [float(np.mean(r["steps"][tau:tau+FIXED_WIN])) for r in iso]
        ref["tau%d" % tau] = {
            "excess_mean": float(np.mean(es)), "excess_median": float(np.median(es)),
            "excess_norm_mean": float(np.mean(en)),
            "iso_fw_mean": float(np.mean(cis)),
            "b_arch_mean": float(np.mean([r["b_arch_at_recv"] for r in fcd])),
            "fw_steps_fcd": float(np.mean(fw_steps_f)), "fw_steps_iso": float(np.mean(fw_steps_i)),
            "wilcoxon_excess_vs_zero_ref": None}
    return tests, ref

def report6(tests, ref):
    print()
    print("--- 害曲線（記述・τ=0含む） ---")
    print("%-8s %10s %10s %12s %8s %10s %10s" % (
        "τ", "excess平均", "excess中央", "excess_norm", "Iso固定窓", "B傷跡平均", "FW歩数F/I"))
    for tau in TAUS_G:
        k = ref["tau%d" % tau]
        print("%-8s %10.2f %10.1f %12.5f %8.2f %10.1f %6.1f/%.1f" % (
            "tau=%d" % tau, k["excess_mean"], k["excess_median"], k["excess_norm_mean"],
            k["iso_fw_mean"], k["b_arch_mean"], k["fw_steps_fcd"], k["fw_steps_iso"]))
    print()
    print("=== 主要検定（事前登録・2本のみ・判定は凍結§3/§4の基準のみ） ===")
    P1 = tests["P1"]
    print("[P1] Dbig+ FCD(τ=25) vs Isolated 衝突ep数[26,1000]")
    print("     FCD: %s / Iso: %s" % (iqr_str(tests["P1_data"]["f25"]), iqr_str(tests["P1_data"]["iso"])))
    if P1 is None:
        print("     全ペア差分ゼロ: 検定不能")
    else:
        print("     Wilcoxon(pratt) W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.2f (参考t p=%.5f)" % (
            P1["W"], P1["p"], P1["p_holm"], P1["rank_biserial"], P1["mean_diff"], P1["t_p_ref"]))
    P2 = tests["P2"]
    print("[P2] excess(τ=25) vs excess(τ=400)（固定窓600ep・同一シード対応）")
    if P2 is None:
        print("     全ペア差分ゼロ: 検定不能")
    else:
        print("     e25: %s / e400: %s" % (iqr_str(tests["P2_data"]["e25"]), iqr_str(tests["P2_data"]["e400"])))
        print("     Wilcoxon(pratt) W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.2f (参考t p=%.5f)" % (
            P2["W"], P2["p"], P2["p_holm"], P2["rank_biserial"], P2["mean_diff"], P2["t_p_ref"]))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if mode == "diag":
        diag6()
    else:
        out, t0 = full6()
        tests, ref = analyze6(out)
        out["_tests"] = tests; out["_reference_not_primary"] = ref
        json.dump(out, open(os.path.join(_BASE6,
            'resonance_v06_gulp_results_30seed.json'), 'w'), ensure_ascii=False)
        print("JSON保存完了 (%.0f秒)" % (time.time()-t0))
        report6(tests, ref)
        print()
        print("判定はDESIGN_遅延丸呑み害曲線_v02_凍結.mdの事前登録基準のみで行うこと。")
