# -*- coding: utf-8 -*-
"""P1d：policy.dat ↔ 直接 Q の整合性監査（93 vs 171 の決着）。
★観測装置の整合性確認。科学的解釈はしない★

比較するもの（t=0, W=0, N_S=12, N_L=3, E=1..2000）
  (a) policy.dat の値（solve_D0 が保存したもの）
  (b) Q から★solve_D0 と同じ規則★で作り直した方策（共通関数）
  (c) Q から単純な argmax で作った方策（P1c がやっていたもの）
不一致は CSV に出す。
"""
import os, sys, json, time, hashlib, csv
import numpy as np
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
D, S = "kernel_full", "solve_D0"
ATOL, RTOL = 1e-9, 1e-12
HORIZON = 1000


def selected_common(q3, best):
    """★solve_D0 と完全に同じ規則★
    tie = q >= best - (ATOL + RTOL*max(1,|best|))
    selected = 草 → 小物 → 大物 の固定順序
    戻り値: (selected, optimal_set のビット)
    """
    thr = ATOL + RTOL * max(1.0, abs(best))
    bits = 0
    for k in range(3):
        if not np.isnan(q3[k]) and q3[k] >= best - thr:
            bits |= (1 << k)
    sel = 1 if bits & 1 else (2 if bits & 2 else (3 if bits & 4 else 0))
    return sel, bits


def selected_argmax(q3):
    """★P1c がやっていたもの（同点の幅を持たない）★"""
    return int(np.argsort(-np.nan_to_num(q3, nan=-np.inf))[0]) + 1


def main():
    print("P1d：policy.dat ↔ 直接 Q の整合性監査（93 vs 171）")
    print()
    # --- ファイルの同一性
    pth = os.path.join(S, "policy.dat")
    st = os.stat(pth)
    h = hashlib.sha256()
    with open(pth, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    print("【policy.dat の同一性】")
    print(f"  サイズ {st.st_size:,} バイト")
    print(f"  更新日時 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}")
    print(f"  SHA-256 {h.hexdigest()[:32]}")
    with open(os.path.join(S, "summary.json"), encoding="utf-8") as f:
        sm = json.load(f)
    print(f"  summary: V0={sm['V_init']:.6f}, kernel={sm['kernel_sha'][:16]}")
    print()

    # --- kernel と V1 を読む
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    n_pairs = man["n_pairs"]
    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob, t_prob = A["succ"], A["prob"], A["t_prob"]
    state_idx = A["state_idx"].astype(np.int64)
    act_id = A["act_id"]
    V1 = np.load(os.path.join(S, "V1.npy"))
    print(f"kernel と V1 を読んだ  RSS {_proc.memory_info().rss/1024**3:.2f}GB")
    assert man["sha256_all_files_full"] == sm["kernel_sha"], "kernel が違う"
    print("  ★kernel の SHA-256 が summary と一致★")

    # --- t=0 の Q を作る（solve_D0 と同じ式）
    owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(offs))
    t_owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(t_offs))
    acc = np.bincount(owner, weights=prob * (1.0 + V1[succ]), minlength=n_pairs)
    acc += np.bincount(t_owner, weights=t_prob, minlength=n_pairs)
    is_dec = state_idx < dx.STRIDE_M
    dec_pairs = np.nonzero(is_dec)[0]
    dec_state = state_idx[dec_pairs]
    dec_act = act_id[dec_pairs]
    Q = np.full((dx.STRIDE_M, 4), np.nan, dtype=np.float64)
    Q[dec_state, dec_act] = acc[dec_pairs]
    print("t=0 の Q を作った")
    print()

    # --- 突き合わせ
    pol = np.memmap(pth, dtype=np.uint8, mode="r", shape=(HORIZON, dx.STRIDE_M))
    opt = np.memmap(os.path.join(S, "optimal_set.dat"), dtype=np.uint8, mode="r",
                    shape=(HORIZON, dx.STRIDE_M))
    rows = []
    n_ng_common, n_ng_argmax = 0, 0
    sw_pol, sw_com, sw_arg = 0, 0, 0
    prev_p = prev_c = prev_a = None
    for e in range(1, 2001):
        i = dx.decision_index_of(e, 0, 12, 3)
        q3 = Q[i, 1:]
        best = np.nanmax(q3)
        sel_c, bits_c = selected_common(q3, best)
        sel_a = selected_argmax(q3)
        sel_p = int(pol[0][i])
        bits_p = int(opt[0][i])
        if sel_p != prev_p:
            sw_pol += 1; prev_p = sel_p
        if sel_c != prev_c:
            sw_com += 1; prev_c = sel_c
        if sel_a != prev_a:
            sw_arg += 1; prev_a = sel_a
        if sel_p != sel_c:
            n_ng_common += 1
            rows.append((e, sel_p, sel_c, sel_a, q3[0], q3[1], q3[2],
                         best - np.sort(q3)[-2], bits_p, bits_c))
        if sel_p != sel_a:
            n_ng_argmax += 1

    print("【切り替わりの回数（t=0, W=0, N_S=12, N_L=3, E=1..2000）】")
    print(f"  (a) policy.dat          : {sw_pol}")
    print(f"  (b) Q＋共通規則（同点あり）: {sw_com}")
    print(f"  (c) Q＋単純 argmax       : {sw_arg}")
    print()
    print("【不一致の件数】")
    print(f"  policy.dat ↔ 共通規則 : {n_ng_common}")
    print(f"  policy.dat ↔ argmax  : {n_ng_argmax}")
    print()
    if rows:
        out = os.path.join(S, "P1d_mismatch.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["E", "policy", "common", "argmax",
                           "Q_grass", "Q_small", "Q_large", "gap",
                           "bits_policy", "bits_common"])
            wcsv.writerows(rows)
        print(f"  不一致を {out} に出した（{len(rows)} 行）")
        print("  先頭5件:")
        for r in rows[:5]:
            print(f"    E={r[0]:4d} policy={r[1]} common={r[2]} argmax={r[3]}"
                  f"  Q=({r[4]:.6f}, {r[5]:.6f}, {r[6]:.6f}) gap={r[7]:.3e}"
                  f"  bits {r[8]:03b}/{r[9]:03b}")
    else:
        print("  ★policy.dat と共通規則は全一致★")
    print()
    print("【判定】")
    if n_ng_common == 0 and n_ng_argmax > 0:
        print("  ★A: 共通の選択規則を使えば一致した。原因は P1c の単純 argmax。solver は無罪★")
        print(f"  ★採用すべき切り替わりの回数は (a)=(b)={sw_pol}★")
    elif n_ng_common == 0 and n_ng_argmax == 0:
        print("  3通りとも一致。93 vs 171 の差は別の場所にある")
    else:
        print("  ★C: 共通規則でも不一致。solver の方策書き出しと Bellman の再構成が不整合★")
    del pol, opt


if __name__ == "__main__":
    main()
