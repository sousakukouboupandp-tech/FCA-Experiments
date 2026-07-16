# -*- coding: utf-8 -*-
"""
NEC 4.0 最小実装（LERI骨格）とアブレーション 本実行
事前登録: DESIGN_NEC4最小実装_v03_凍結.md（確定凍結2026年7月16日・§9エラッタ反映済み）
主要検定: P1 = ΔD（コア−非コア生存率差）のμ=0.2 vs μ=0（同時演算）
          P2 = 累積G改善の同時演算 vs 最良交互対照（c∈{1,10,60}）
使い方: python nec4_min.py ut | full
"""
import sys, os, json, time
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
from scipy import stats as spstats

BASE = os.path.dirname(os.path.abspath(__file__))
N, DIM, T, NCORE = 32, 8, 3000, 8
THR, MU_BASE, EPS = 0.1, 0.2, 1e-4
W_SURV = 0.3

def normalize(x):
    n = np.linalg.norm(x)
    return x / n if n >= 1e-9 else np.zeros_like(x)

def xi_all(w, V):
    """Ξ_i: 局所パターン(自要素+リング±1,±2)と全体縮約パターンのcos（負は0）凍結§2.4"""
    m = normalize((w[:, None] * V).sum(0))
    Xi = np.zeros(N)
    if not m.any():
        return Xi
    for i in range(N):
        nb = V[(i - 2) % N] + V[(i - 1) % N] + V[i] + V[(i + 1) % N] + V[(i + 2) % N]
        l = normalize(nb)
        Xi[i] = max(0.0, float(l @ m)) if l.any() else 0.0
    return Xi

def init_structure(rng):
    """凍結§2.2: m0→コア(0..7)→非コア(8..31)。UT0: コア平均Ξ≥0.7、不合格ならコアのみ引き直し(最大100回)"""
    m0 = normalize(rng.randn(DIM))
    Vn = None
    for attempt in range(1, 101):
        Vc = np.array([normalize(m0 + 0.05 * rng.randn(DIM)) for _ in range(NCORE)])
        if Vn is None:
            Vn = np.array([normalize(rng.randn(DIM)) for _ in range(N - NCORE)])
        V = np.vstack([Vc, Vn])
        w = np.array([0.8] * NCORE + [0.5] * (N - NCORE))
        Xi = xi_all(w, V)
        if Xi[:NCORE].mean() >= 0.7:
            return w, V, attempt
    print("!!! UT0不合格（100回試行）→ 停止"); sys.exit(1)

def gen_P_sequence(rng):
    """凍結§2.3: P(0)=4·球面一様。間隔U[160,240]、4次元にN(0,1)加算。イベント系列を事前生成"""
    P = 4.0 * normalize(rng.randn(DIM))
    seq = np.zeros((T + 1, DIM))
    events = []
    t = 0
    while True:
        dt = rng.randint(160, 241)
        t += dt
        if t >= T:
            break
        dims = rng.choice(DIM, 4, replace=False)
        deltas = rng.randn(4) * 1.0
        events.append((t, dims, deltas))
    ev = {e[0]: (e[1], e[2]) for e in events}
    cur = P.copy()
    for tt in range(T + 1):
        if tt in ev:
            d, dl = ev[tt]
            cur = cur.copy(); cur[d] += dl
        seq[tt] = cur
    return seq

def run_system(seed, mode, c=None, mu=MU_BASE, lam_zero=False, t_max=T,
               phi_off=False, lam_fix=None, ut1_check=False):
    """1本の実行。mode='sim'(同時)|'alt'(交互・周期c)。凍結§2の全式を実装。
    状態評価タイミング（§2.1）: 開始時状態から全量を一度だけ算出→全要素共有→一括適用。"""
    rng = np.random.RandomState(seed)
    w, V, ut0_attempts = init_structure(rng)
    Pseq = gen_P_sequence(rng)
    w0 = w.copy()
    Gs = []
    pain_times = []
    conv_ct = diss_ct = 0
    per_conv = np.zeros(N, int); per_diss = np.zeros(N, int)
    w100 = None
    lam_log = []
    ut1_ok = True
    ut4_probes = []
    for t in range(t_max):
        P = Pseq[t]
        y = (w[:, None] * V).sum(0)
        G = float(np.sum((y - P) ** 2))
        if t > 0 and G > Gs[-1] * 1.05:
            pain_times.append(t)
        Gs.append(G)
        G50 = Gs[t - 50] if t >= 50 else Gs[0]
        imp = float(np.clip((G50 - G) / (G50 + 1e-9), -1, 1))
        pain_sum = sum(np.exp(-(t - te) / 100.0) for te in pain_times)
        Pain = float(np.tanh(pain_sum / 3.0))
        phi = 0.0 if phi_off else float(np.clip(imp - 0.5 * Pain, -1, 1))

        r = V @ (P - y)
        mr = float(np.abs(r).max())
        psi = r / mr if mr > 1e-12 else np.zeros(N)
        g = np.empty(N)
        for i in range(N):
            Gp = float(np.sum((y + EPS * V[i] - P) ** 2))
            Gm = float(np.sum((y - EPS * V[i] - P) ** 2))
            g[i] = (Gp - Gm) / (2 * EPS)
        mg = float(np.abs(g).max())
        grn = g / mg if mg > 1e-12 else np.zeros(N)
        Xi = xi_all(w, V)
        Xig = float(Xi.mean())
        G200 = Gs[t - 200] if t >= 200 else Gs[0]
        RQ = float(np.clip((G200 - G) / (G200 + 1e-9), 0, 1))
        if lam_zero:
            lam = 0.0
        elif lam_fix is not None:
            lam = lam_fix
        else:
            lam = max(0.0, 0.7 - 0.3 * RQ * Xig)
        lam_log.append(lam)
        if t in (100, 1000, 2500):
            lam2 = max(0.0, 0.7 - 0.3 * np.clip((G200 - G) / (G200 + 1e-9), 0, 1) * np.mean(xi_all(w, V)))
            phi2 = np.clip(np.clip((G50 - G) / (G50 + 1e-9), -1, 1) - 0.5 * np.tanh(pain_sum / 3.0), -1, 1)
            ut4_probes.append((abs((0.0 if lam_zero else (lam_fix if lam_fix is not None else lam2)) - lam),
                               abs((0.0 if phi_off else phi2) - phi)))
        LERI = phi * psi - lam * grn + mu * Xi
        conv = LERI > THR
        diss = LERI < -THR
        if ut1_check:
            if not (np.array_equal(conv, (phi * psi) > THR) and np.array_equal(diss, (phi * psi) < -THR)):
                ut1_ok = False
        if mode == 'alt':
            phase = (t // c) % 2  # 0=収束モード（開始は収束）
            if phase == 0: diss[:] = False
            else: conv[:] = False
        d = P - y
        nd = float(np.linalg.norm(d))
        dhat = d / nd if nd >= 1e-9 else None
        wi = np.where(conv)[0]
        w[wi] = np.minimum(1.0, w[wi] + 0.05)
        if dhat is not None:
            for i in wi:
                V[i] = normalize(0.9 * V[i] + 0.1 * dhat)
        di = np.where(diss)[0]
        w[di] *= 0.5
        for i in di:
            V[i] = normalize(rng.randn(DIM))
        conv_ct += len(wi); diss_ct += len(di)
        per_conv[wi] += 1; per_diss[di] += 1
        if t == 99:
            w100 = w.copy()

    yF = (w[:, None] * V).sum(0)
    Gs.append(float(np.sum((yF - Pseq[t_max]) ** 2)))
    cum_imp = float(sum(max(0.0, Gs[k - 1] - Gs[k]) for k in range(1, len(Gs))))
    surv = w > W_SURV
    S_core = float(surv[:NCORE].mean()); S_non = float(surv[NCORE:].mean())
    early = None
    if w100 is not None:
        early = float(((w100[:NCORE] - w0[:NCORE]).mean() - (w100[NCORE:] - w0[NCORE:]).mean()) / 100.0)
    return {"S_core": S_core, "S_non": S_non, "D": S_core - S_non,
            "cum_imp": cum_imp, "conv_ct": int(conv_ct), "diss_ct": int(diss_ct),
            "events": int(conv_ct + diss_ct),
            "per_conv_core": int(per_conv[:NCORE].sum()), "per_conv_non": int(per_conv[NCORE:].sum()),
            "per_diss_core": int(per_diss[:NCORE].sum()), "per_diss_non": int(per_diss[NCORE:].sum()),
            "early_growth": early, "ut0_attempts": int(ut0_attempts),
            "lam_min": float(min(lam_log)), "lam_max": float(max(lam_log)),
            "lam_nonconst": bool(max(lam_log) - min(lam_log) > 1e-12),
            "ut1_ok": bool(ut1_ok),
            "ut4_maxerr": float(max(max(a, b) for a, b in ut4_probes)) if ut4_probes else None,
            "G_final": Gs[-1]}

def paired_wilcoxon_pratt(x, y):
    x, y = np.array(x, float), np.array(y, float)
    d = x - y
    if np.all(d == 0):
        return None
    wst, p = spstats.wilcoxon(x, y, zero_method='pratt')
    dz = d[d != 0]
    ranks = spstats.rankdata(np.abs(dz))
    wp = float(ranks[dz > 0].sum()); wm = float(ranks[dz < 0].sum())
    rb = (wp - wm) / (wp + wm) if (wp + wm) > 0 else 0.0
    ts, tp = spstats.ttest_rel(x, y)
    return {"W": float(wst), "p": float(p), "rb": rb, "mean_diff": float(np.mean(d)),
            "median_diff": float(np.median(d)), "t_p_ref": float(tp)}

def holm_2(p1, p2):
    if p1 is None or p2 is None:
        return p1, p2
    if p1 <= p2:
        a1 = min(1.0, 2 * p1); a2 = min(1.0, max(a1, p2))
    else:
        a2 = min(1.0, 2 * p2); a1 = min(1.0, max(a2, p1))
    return a1, a2

def iqr_str(v):
    a = np.array(v, float)
    return "中央値%.3f IQR[%.3f-%.3f]" % (np.median(a), np.percentile(a, 25), np.percentile(a, 75))

def run_ut():
    print("=== ユニットテスト（凍結§2.6・seed0） ===")
    t0 = time.time()
    # UT0+UT1: φψ単独（λ=μ=0）配線: 判定がφψのみに一致
    r1 = run_system(0, 'sim', mu=0.0, lam_zero=True, t_max=200, ut1_check=True)
    print("UT0: 初期化試行%d回で合格" % r1["ut0_attempts"])
    print("UT1(φψ単独・判定一致):", "合格" if r1["ut1_ok"] else "不合格→停止")
    if not r1["ut1_ok"]: sys.exit(1)
    # UT2(エラッタ版): λ配線確認 (a)非定数[0,0.7] (b)λ有効vsλ恒等0で収束実行数が異なる
    r2a = run_system(0, 'sim', mu=MU_BASE)
    r2b = run_system(0, 'sim', mu=MU_BASE, lam_zero=True)
    ok2 = r2a["lam_nonconst"] and 0.0 <= r2a["lam_min"] and r2a["lam_max"] <= 0.7 and r2a["conv_ct"] != r2b["conv_ct"]
    print("UT2(λ配線): λ範囲[%.3f,%.3f] 非定数=%s 収束数 %d vs %d(λ=0) →" % (
        r2a["lam_min"], r2a["lam_max"], r2a["lam_nonconst"], r2a["conv_ct"], r2b["conv_ct"]),
        "合格" if ok2 else "不合格→停止")
    if not ok2: sys.exit(1)
    # UT3: μΞ単独（φ=λ=0）: Ξ≥0.8要素が散逸閾値を下回らない（LERI=0.2Ξ≥0で構造的に保証・配線確認）
    r3 = run_system(0, 'sim', mu=MU_BASE, lam_zero=True, phi_off=True, t_max=500)
    print("UT3(μΞ単独): 散逸実行数=%d（μΞ≥0のため0であるべき）→" % r3["diss_ct"],
        "合格" if r3["diss_ct"] == 0 else "不合格→停止")
    if r3["diss_ct"] != 0: sys.exit(1)
    # UT4: 数式一致プローブ（t=100,1000,2500でλ・φを再計算し誤差<1e-9）
    ok4 = r2a["ut4_maxerr"] is not None and r2a["ut4_maxerr"] < 1e-9
    print("UT4(数式一致): 最大誤差=%.2e →" % (r2a["ut4_maxerr"] or -1), "合格" if ok4 else "不合格→停止")
    if not ok4: sys.exit(1)
    # UT5: 交互c=1の実行イベント総数が同時の50%〜150%
    r5 = run_system(0, 'alt', c=1, mu=MU_BASE)
    ratio = r5["events"] / max(1, r2a["events"])
    ok5 = 0.5 <= ratio <= 1.5
    print("UT5(等価性): 交互c=1イベント%d / 同時%d = %.2f →" % (r5["events"], r2a["events"], ratio),
        "合格" if ok5 else "不合格→停止")
    if not ok5: sys.exit(1)
    print("=== 全UT合格 (%.0f秒) ===" % (time.time() - t0))

def run_full():
    t0 = time.time()
    conds = [("sim_mu02", "sim", None, MU_BASE), ("sim_mu00", "sim", None, 0.0),
             ("alt_c1", "alt", 1, MU_BASE), ("alt_c10", "alt", 10, MU_BASE), ("alt_c60", "alt", 60, MU_BASE)]
    out = {}
    for name, mode, c, mu in conds:
        runs = [run_system(s, mode, c=c, mu=mu) for s in range(30)]
        out[name] = runs
        print("  %s 完了 (%.0f秒)" % (name, time.time() - t0))
    return out, t0

def analyze(out):
    D02 = [r["D"] for r in out["sim_mu02"]]
    D00 = [r["D"] for r in out["sim_mu00"]]
    P1 = paired_wilcoxon_pratt(D02, D00)  # mean_diff>0 = 選択的保護あり
    cum_sim = [r["cum_imp"] for r in out["sim_mu02"]]
    alt_means = {c: float(np.mean([r["cum_imp"] for r in out["alt_c%d" % c]])) for c in (1, 10, 60)}
    best_c = max(alt_means, key=alt_means.get)  # 事後最良（最も不利な比較・凍結§3）
    cum_alt = [r["cum_imp"] for r in out["alt_c%d" % best_c]]
    P2 = paired_wilcoxon_pratt(cum_sim, cum_alt)  # mean_diff>0 = 同時が優位
    if P1 is not None and P2 is not None:
        h1, h2 = holm_2(P1["p"], P2["p"]); P1["p_holm"] = h1; P2["p_holm"] = h2
    return {"P1": P1, "P2": P2, "best_alt_c": best_c, "alt_means": alt_means,
            "P1_data": {"D02": D02, "D00": D00},
            "P2_data": {"sim": cum_sim, "alt_best": cum_alt}}

def report(out, tests):
    print()
    print("--- 記述（30シード平均） ---")
    print("%-10s %8s %8s %8s %10s %8s %8s" % ("条件", "S_core", "S_non", "D", "累積G改善", "収束数", "散逸数"))
    for name in ("sim_mu02", "sim_mu00", "alt_c1", "alt_c10", "alt_c60"):
        rs = out[name]
        print("%-10s %8.3f %8.3f %8.3f %10.1f %8.0f %8.0f" % (
            name, np.mean([r["S_core"] for r in rs]), np.mean([r["S_non"] for r in rs]),
            np.mean([r["D"] for r in rs]), np.mean([r["cum_imp"] for r in rs]),
            np.mean([r["conv_ct"] for r in rs]), np.mean([r["diss_ct"] for r in rs])))
    print("early_growth(sim_mu02平均): %.5f / 遷移分布(コア収束/非コア収束/コア散逸/非コア散逸 mu02): %d/%d/%d/%d" % (
        np.mean([r["early_growth"] for r in out["sim_mu02"]]),
        sum(r["per_conv_core"] for r in out["sim_mu02"]), sum(r["per_conv_non"] for r in out["sim_mu02"]),
        sum(r["per_diss_core"] for r in out["sim_mu02"]), sum(r["per_diss_non"] for r in out["sim_mu02"])))
    print()
    print("=== 主要検定（凍結§3・2本のみ・判定は§3/§4基準のみ） ===")
    P1 = tests["P1"]
    print("[P1] ΔD = D(μ=0.2) − D(μ=0)（同時演算・選択的防波堤）")
    print("     D(μ=0.2): %s / D(μ=0): %s" % (iqr_str(tests["P1_data"]["D02"]), iqr_str(tests["P1_data"]["D00"])))
    if P1 is None:
        print("     全ペア差分ゼロ: 検定不能")
    else:
        print("     Wilcoxon(pratt) W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.4f (参考t p=%.5f)" % (
            P1["W"], P1["p"], P1["p_holm"], P1["rb"], P1["mean_diff"], P1["t_p_ref"]))
    P2 = tests["P2"]
    print("[P2] 累積G改善: 同時(μ=0.2) vs 最良交互 c=%d（交互平均: c1=%.1f c10=%.1f c60=%.1f）" % (
        tests["best_alt_c"], tests["alt_means"][1], tests["alt_means"][10], tests["alt_means"][60]))
    if P2 is None:
        print("     全ペア差分ゼロ: 検定不能")
    else:
        print("     同時: %s / 最良交互: %s" % (iqr_str(tests["P2_data"]["sim"]), iqr_str(tests["P2_data"]["alt_best"])))
        print("     Wilcoxon(pratt) W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.1f (参考t p=%.5f)" % (
            P2["W"], P2["p"], P2["p_holm"], P2["rb"], P2["mean_diff"], P2["t_p_ref"]))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ut"
    if mode == "ut":
        run_ut()
    else:
        out, t0 = run_full()
        tests = analyze(out)
        out["_tests"] = tests
        json.dump(out, open(os.path.join(BASE, 'nec4_min_results_30seed.json'), 'w'), ensure_ascii=False)
        print("JSON保存完了 (%.0f秒)" % (time.time() - t0))
        report(out, tests)
        print()
        print("判定はDESIGN_NEC4最小実装_v03_凍結.mdの事前登録基準のみで行うこと。")
