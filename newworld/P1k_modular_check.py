# -*- coding: utf-8 -*-
"""P1k：modular energy dynamics の確認（HYPOTHESIS_P1k、52d309f で凍結）
  P1k-1 物理レベル：kernel の全枝の ΔE mod 20 を行動別に数える（★方策に依存しない★）
  P1k-2 W=1 を mod20 と mod25 の両方で見る（★8件なので探索的★）
  P1k-3 Q_grass − Q_small の符号が E と E+20 で一致する割合（E+19, E+21 と比較）
★統計検定は持ち込まない。件数30未満には「偶然と区別できない」と明記★
"""
import os, sys, json
from collections import Counter
import numpy as np
import world1g as w
from world1g import DECISION, E_MAX
import dense_index as dx
from P1d_policy_audit import selected_common

D, S = "kernel_full", "solve_D0"
ACT = {1: "草", 2: "小物", 3: "大物", 0: "強制MODE"}


def main():
    print("P1k：modular energy dynamics の確認")
    print("★仮説は 52d309f で凍結済み。数字を見てから書き換えない★")
    print()
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    offs = A["offs"]; succ = A["succ"]
    state_idx = A["state_idx"].astype(np.int64); act_id = A["act_id"]

    # ---- P1k-1：物理レベル（全枝の ΔE mod 20）
    print("【P1k-1】kernel の全枝の ΔE mod 20（★方策に依存しない物理の性質★）")
    n_pairs = man["n_pairs"]
    # ★メモリを節約する：int32 で作り、使い終わった配列はすぐ消す★
    owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(offs))
    e_from = ((state_idx[owner] // dx.STRIDE_E) % dx.N_E).astype(np.int16)
    w_from = ((state_idx[owner] // dx.STRIDE_W) % dx.N_W).astype(np.int8)
    act_of_edge = act_id[owner].astype(np.int8)
    del owner
    e_to = ((succ // dx.STRIDE_E) % dx.N_E).astype(np.int16)
    d_mod20 = ((e_to.astype(np.int32) - e_from.astype(np.int32)) % 20).astype(np.int8)
    clip = (e_to == E_MAX)
    del e_to, e_from
    import gc
    gc.collect()

    print(f"  全枝 {len(succ):,} 本（うち上限クリップ {int(clip.sum()):,} 本）")
    print()
    print("  ◆W=0 の状態から出る枝、クリップを除く◆")
    m = (w_from == 0) & (~clip)
    for a in (1, 2, 3, 0):
        sel = m & (act_of_edge == a)
        n = int(sel.sum())
        if n == 0:
            continue
        c = Counter(d_mod20[sel].tolist())
        top = c.most_common(3)
        print(f"    {ACT[a]:8s} {n:>11,} 本  ΔE mod 20 の分布 "
              f"{[(int(k), int(v)) for k, v in top]}"
              f"{'' if len(c) <= 3 else f' …他{len(c)-3}種'}")
        # 支配的な値の割合
        k0, v0 = top[0]
        print(f"             ★最頻 {int(k0)} が {v0/n*100:5.1f}%★  種類 {len(c)}")
    print()
    print("  ◆強制 MODE は行動 id=0。decision の1歩目だけを見る◆")
    print("  （強制 MODE の枝は、探索・追跡・戦闘の途中。摂取がないので −20 ≡ 0 mod 20 になるはず）")
    print()

    # ---- P1k-2：W=1 を mod20 と mod25 で
    print("【P1k-2】W=1 の断面を mod20 と mod25 の両方で見る（★探索的★）")
    pol = np.memmap(os.path.join(S, "policy.dat"), dtype=np.uint8, mode="r",
                    shape=(1000, dx.STRIDE_M))
    for wnd in (0, 1):
        sels = [int(pol[0][dx.decision_index_of(e, wnd, 12, 3)])
                for e in range(1, 2001)]
        sw = [k + 1 for k in range(1, 2000) if sels[k] != sels[k - 1]]
        c20 = Counter(e % 20 for e in sw)
        c25 = Counter(e % 25 for e in sw)
        print(f"  W={wnd}  switch {len(sw)} 件")
        print(f"    mod20: {dict(sorted((int(a),int(b)) for a,b in c20.items()))}"
              f"  （種類 {len(c20)}）")
        print(f"    mod25: {dict(sorted((int(a),int(b)) for a,b in c25.items()))}"
              f"  （種類 {len(c25)}）")
        if len(sw) and len(sw) < 30:
            print("    ★件数30未満。偶然と区別できない★")
        # 上位2位相が占める割合で比べる
        for nm, c in (("mod20", c20), ("mod25", c25)):
            if len(sw):
                t2 = sum(v for _, v in c.most_common(2))
                print(f"    {nm} の上位2位相が占める割合: {t2}/{len(sw)}"
                      f" = {t2/len(sw)*100:5.1f}%")
    print()

    # ---- P1k-3：優位性の周期性
    print("【P1k-3】Q_grass − Q_small の符号の周期性（t=0, W=0, N_S=12, N_L=3）")
    del d_mod20, clip, w_from, act_of_edge
    gc.collect()
    V1 = np.load(os.path.join(S, "V1.npy"))
    t_offs = A["t_offs"]; prob = A["prob"]; t_prob = A["t_prob"]
    owner2 = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(offs))
    t_owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(t_offs))
    acc = np.bincount(owner2, weights=prob * (1.0 + V1[succ]), minlength=n_pairs)
    acc += np.bincount(t_owner, weights=t_prob, minlength=n_pairs)
    del owner2, t_owner
    gc.collect()
    dec = np.nonzero(state_idx < dx.STRIDE_M)[0]
    Q = np.full((dx.STRIDE_M, 4), np.nan, dtype=np.float64)
    Q[state_idx[dec], act_id[dec]] = acc[dec]
    adv = np.full(2001, np.nan)
    for e in range(1, 2001):
        i = dx.decision_index_of(e, 0, 12, 3)
        adv[e] = Q[i, 1] - Q[i, 2]      # 草 − 小物
    sgn = np.sign(adv)
    print("   ずらし幅 | 符号が一致する割合")
    for lag in (19, 20, 21, 40, 1):
        ok = tot = 0
        for e in range(1, 2001 - lag):
            if np.isnan(adv[e]) or np.isnan(adv[e + lag]):
                continue
            tot += 1
            if sgn[e] == sgn[e + lag]:
                ok += 1
        mark = "★" if lag in (20, 40) else "  "
        print(f"   {mark}E と E+{lag:2d} | {ok}/{tot} = {ok/tot*100:5.1f}%")
    print()
    print("  （lag=1 は隣接。lag=19/21 は20との比較。lag=40 は20の倍数）")
    del pol


if __name__ == "__main__":
    main()
