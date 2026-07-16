# -*- coding: utf-8 -*-
"""
μΞ用量反応実験 本実行（駆動源仮説の確認的検定・新シード30-59）
事前登録: DESIGN_μ用量反応_v02_凍結.md（2026年7月16日凍結）
方式: nec4_min.py（NEC4凍結仕様の実装・UT合格済み）のソースを読み込み、
      観測専用ログ（ステップ別イベント数）のみを注入してexec。ロジックは一切変更しない。
主要検定: P1 = 累積G改善 μ=0.2 vs μ=0 ／ P2 = μ=0.1 vs μ=0（ホルム・新シード）
使い方: python mu_dose.py ut | full
"""
import sys, os, json, time
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_BASE, 'nec4_min.py'), encoding='utf-8').read()
_src = _src.split('if __name__')[0]
# 観測専用ログの注入（2箇所・状態と乱数に無影響）
_a = "    lam_log = []"
_b = "    lam_log = []\n    step_ev = []; step_diss = []"
assert _src.count(_a) == 1
_src = _src.replace(_a, _b)
_c = "        conv_ct += len(wi); diss_ct += len(di)"
_d = "        conv_ct += len(wi); diss_ct += len(di)\n        step_ev.append(len(wi) + len(di)); step_diss.append(len(di))"
assert _src.count(_c) == 1
_src = _src.replace(_c, _d)
_e = '"G_final": Gs[-1]}'
_f = '"G_final": Gs[-1], "step_ev": step_ev, "step_diss": step_diss}'
assert _src.count(_e) == 1
_src = _src.replace(_e, _f)
# UTのシードを30に差し替え（凍結§2: UTはシード30で再実行）
_g = "run_system(0, 'sim'"
assert _src.count(_g) == 4
_src = _src.replace(_g, "run_system(UT_SEED, 'sim'")
_h = "run_system(0, 'alt'"
assert _src.count(_h) == 1
_src = _src.replace(_h, "run_system(UT_SEED, 'alt'")
UT_SEED = 30
exec(compile(_src, 'nec4_min.py(patched)', 'exec'), globals())

MUS = [0.0, 0.1, 0.2, 0.4]
SEEDS_MU = list(range(30, 60))

def anomaly_metrics(r):
    """凍結§5: 固着（遷移0が連続200step以上）と発振（散逸>16のstep数）。観測のみ"""
    ev = r["step_ev"]
    max_run = cur = 0
    stall_epochs = 0
    for e in ev:
        if e == 0:
            cur += 1
            if cur == 200:
                stall_epochs += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    osc = sum(1 for d in r["step_diss"] if d > 16)
    return {"stall_runs200": stall_epochs, "stall_max": max_run, "osc_steps": osc}

def full_mu():
    t0 = time.time()
    out = {}
    for mu in MUS:
        key = "mu_%s" % ("%.1f" % mu).replace(".", "")
        runs = []
        for s in SEEDS_MU:
            r = run_system(s, 'sim', mu=mu)
            r.update(anomaly_metrics(r))
            del r["step_ev"]; del r["step_diss"]
            runs.append(r)
        out[key] = runs
        print("  mu=%.1f 完了 (%.0f秒)" % (mu, time.time() - t0))
    return out, t0

def analyze_mu(out):
    c00 = [r["cum_imp"] for r in out["mu_00"]]
    c01 = [r["cum_imp"] for r in out["mu_01"]]
    c02 = [r["cum_imp"] for r in out["mu_02"]]
    P1 = paired_wilcoxon_pratt(c02, c00)
    P2 = paired_wilcoxon_pratt(c01, c00)
    if P1 is not None and P2 is not None:
        h1, h2 = holm_2(P1["p"], P2["p"]); P1["p_holm"] = h1; P2["p_holm"] = h2
    return {"P1": P1, "P2": P2,
            "data": {"c00": c00, "c01": c01, "c02": c02,
                     "c04": [r["cum_imp"] for r in out["mu_04"]]}}

def report_mu(out, tests):
    print()
    print("--- 用量曲線（記述・30シード平均） ---")
    print("%-8s %10s %8s %8s %8s %8s %8s %10s %8s" % (
        "μ", "累積G改善", "収束数", "散逸数", "S_core", "S_non", "D", "固着最長", "発振step"))
    for mu in MUS:
        key = "mu_%s" % ("%.1f" % mu).replace(".", "")
        rs = out[key]
        print("%-8s %10.1f %8.0f %8.0f %8.3f %8.3f %8.3f %10.0f %8.0f" % (
            "%.1f" % mu, np.mean([r["cum_imp"] for r in rs]),
            np.mean([r["conv_ct"] for r in rs]), np.mean([r["diss_ct"] for r in rs]),
            np.mean([r["S_core"] for r in rs]), np.mean([r["S_non"] for r in rs]),
            np.mean([r["D"] for r in rs]),
            np.mean([r["stall_max"] for r in rs]), np.mean([r["osc_steps"] for r in rs])))
    print()
    print("=== 主要検定（凍結§3・2本のみ・判定は§3/§4基準のみ） ===")
    for name, P, a, b in (("P1(駆動の確認: μ0.2 vs μ0)", tests["P1"], "c02", "c00"),
                          ("P2(薄い用量: μ0.1 vs μ0)", tests["P2"], "c01", "c00")):
        print("[%s]" % name)
        print("     %s / %s" % (iqr_str(tests["data"][a]), iqr_str(tests["data"][b])))
        if P is None:
            print("     全ペア差分ゼロ: 検定不能")
        else:
            print("     Wilcoxon(pratt) W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.1f (参考t p=%.5f)" % (
                P["W"], P["p"], P["p_holm"], P["rb"], P["mean_diff"], P["t_p_ref"]))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ut"
    if mode == "ut":
        run_ut()
    else:
        out, t0 = full_mu()
        tests = analyze_mu(out)
        out["_tests"] = tests
        json.dump(out, open(os.path.join(_BASE, 'mu_dose_results_30seed.json'), 'w'), ensure_ascii=False)
        print("JSON保存完了 (%.0f秒)" % (time.time() - t0))
        report_mu(out, tests)
        print()
        print("判定はDESIGN_μ用量反応_v02_凍結.mdの事前登録基準のみで行うこと。")
