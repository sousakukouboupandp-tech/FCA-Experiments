# -*- coding: utf-8 -*-
"""P1e：canonical 171回の全数分類（ChatGPT / AIスタジオ 両方の指示）。
★新しい仮説を立てない。同じ Q と凍結規則から記述統計だけ取り直す★
★新しい閾値を作らない。凍結済みの ε グリッド7点で全部報告する★

canonical = policy.dat と一致した「Q ＋ 共通規則（同点あり・固定順序）」

出すもの
  1. 171回の切り替わりの全数分類
       ① strict → strict（前後とも optimal_set が単独で、その行動が変わった）
       ② tie-mediated（前後どちらかに複数の最適行動がある）
  2. optimal_set 自体の変化の回数（全1999境界）
  3. ε グリッド7点で「両側とも g_rel > ε」を満たす switch の数
  4. ★171回版の間隔の分布★（93回版とは別に測る）／interval==20 の件数／E mod 20
"""
import os, sys, json, csv
from collections import Counter
import numpy as np
import dense_index as dx
from P1d_policy_audit import selected_common
import psutil

_proc = psutil.Process(os.getpid())
D, S = "kernel_full", "solve_D0"
EPS = [1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]   # ★凍結値★
NAMES = {0: "-", 1: "草", 2: "小物", 3: "大物"}


def bits_str(b):
    return "".join(n for k, n in ((1, "草"), (2, "小"), (4, "大")) if b & k) or "-"


def main():
    print("P1e：canonical 171回の全数分類")
    print("★凍結した ε グリッド7点をそのまま使う。新しい閾値を作らない★")
    print()
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    n_pairs = man["n_pairs"]
    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob, t_prob = A["succ"], A["prob"], A["t_prob"]
    state_idx = A["state_idx"].astype(np.int64)
    act_id = A["act_id"]
    V1 = np.load(os.path.join(S, "V1.npy"))
    owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(offs))
    t_owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(t_offs))
    acc = np.bincount(owner, weights=prob * (1.0 + V1[succ]), minlength=n_pairs)
    acc += np.bincount(t_owner, weights=t_prob, minlength=n_pairs)
    is_dec = state_idx < dx.STRIDE_M
    dec = np.nonzero(is_dec)[0]
    Q = np.full((dx.STRIDE_M, 4), np.nan, dtype=np.float64)
    Q[state_idx[dec], act_id[dec]] = acc[dec]
    print(f"t=0 の Q を作った  RSS {_proc.memory_info().rss/1024**3:.2f}GB")
    print()

    # --- canonical を E=1..2000 で作る
    rec = []
    for e in range(1, 2001):
        i = dx.decision_index_of(e, 0, 12, 3)
        q3 = Q[i, 1:]
        best = np.nanmax(q3)
        sel, bits = selected_common(q3, best)
        srt = np.sort(np.nan_to_num(q3, nan=-np.inf))[::-1]
        gap = srt[0] - srt[1]
        g_rel = gap / max(1.0, abs(srt[0]), abs(srt[1]))
        n_opt = bin(bits).count("1")
        rec.append(dict(e=e, q=q3.copy(), sel=sel, bits=bits, gap=gap,
                        g_rel=g_rel, n_opt=n_opt))

    # --- 1. 切り替わりの全数分類
    sws = []
    for k in range(1, len(rec)):
        a, b = rec[k - 1], rec[k]
        if a["sel"] != b["sel"]:
            sws.append((a, b))
    print(f"【1. canonical の切り替わり回数】{len(sws)}")
    strict, tie_med = [], []
    for (a, b) in sws:
        if a["n_opt"] == 1 and b["n_opt"] == 1:
            strict.append((a, b))
        else:
            tie_med.append((a, b))
    print(f"  ① strict → strict（前後とも単独）: {len(strict)}")
    print(f"  ② tie-mediated（どちらかに複数）  : {len(tie_med)}")
    print()

    # --- 2. optimal_set 自体の変化（全1999境界）
    n_bits_change = sum(1 for k in range(1, len(rec))
                        if rec[k - 1]["bits"] != rec[k]["bits"])
    print(f"【2. optimal_set 自体が変わった境界】{n_bits_change} / 1999")
    print(f"  （selected が変わった境界は {len(sws)}）")
    print()

    # --- 3. ε グリッドで「両側とも g_rel > ε」
    print("【3. 両側とも g_rel > ε を満たす switch の数（凍結した7点）】")
    for eps in EPS:
        n = sum(1 for (a, b) in sws if a["g_rel"] > eps and b["g_rel"] > eps)
        ns = sum(1 for (a, b) in strict if a["g_rel"] > eps and b["g_rel"] > eps)
        print(f"  g_rel > {eps:.0e} : 全体 {n:3d} / 171   （うち strict {ns:3d}）")
    print()

    # --- 4. ★171回版の間隔の分布★
    pos = [b["e"] for (a, b) in sws]
    d = np.diff(pos) if len(pos) > 1 else np.array([])
    print("【4. ★171回版★の切り替わりの間隔】（93回版とは別に測る）")
    if len(d):
        print(f"  最小 {d.min()} / 最大 {d.max()} / 中央 {int(np.median(d))}"
              f" / 平均 {d.mean():.1f}")
        c = Counter(d.tolist())
        print(f"  最頻 {c.most_common(5)}")
        print(f"  ★interval == 20 の件数: {c.get(20, 0)}★")
        print(f"  間隔の分布: {dict(sorted(c.items()))}")
    print()
    print("【E mod 20 の分布（切り替わりが起きた E）】")
    cm = Counter([e % 20 for e in pos])
    print(f"  {dict(sorted(cm.items()))}")
    exp = len(pos) / 20
    print(f"  （一様なら各 {exp:.1f} 件）最大 {max(cm.values())} 最小 {min(cm.values()) if len(cm)==20 else 0}")
    print()

    # --- CSV に全 switch を出す
    out = os.path.join(S, "P1e_switches.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wc = csv.writer(f)
        wc.writerow(["E_left", "E_right", "sel_left", "sel_right",
                     "optset_left", "optset_right",
                     "Qg_left", "Qs_left", "Ql_left",
                     "Qg_right", "Qs_right", "Ql_right",
                     "g_rel_left", "g_rel_right", "kind"])
        prev_e = None
        for (a, b) in sws:
            kind = "strict" if (a["n_opt"] == 1 and b["n_opt"] == 1) else "tie"
            wc.writerow([a["e"], b["e"], NAMES[a["sel"]], NAMES[b["sel"]],
                         bits_str(a["bits"]), bits_str(b["bits"]),
                         *[f"{x:.9f}" for x in a["q"]],
                         *[f"{x:.9f}" for x in b["q"]],
                         f"{a['g_rel']:.6e}", f"{b['g_rel']:.6e}", kind])
    print(f"全 {len(sws)} 件を {out} に出した")
    print()
    print("【strict switch の先頭10件】")
    print("  E_left→E_right | 行動 | optset | g_rel左 | g_rel右")
    for (a, b) in strict[:10]:
        print(f"  {a['e']:5d}→{b['e']:5d} | {NAMES[a['sel']]}→{NAMES[b['sel']]}"
              f" | {bits_str(a['bits'])}→{bits_str(b['bits'])}"
              f" | {a['g_rel']:.2e} | {b['g_rel']:.2e}")


if __name__ == "__main__":
    main()
