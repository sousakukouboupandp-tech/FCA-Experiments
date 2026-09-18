# -*- coding: utf-8 -*-
"""P1c：実寸で t=0 の Q を直接測る。
solve_D0 は V_0 しか保存していないので、★V_1 を保存しながら後ろ向きを回す★。
そのあと t=0 の Q（行動ごと）を体力方向に並べ、
振動が「実質的な差」か「numerical tie 近傍」かを判定する。
"""
import os, sys, json, time, gc
import numpy as np
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
D, S = "kernel_full", "solve_D0"
ATOL, RTOL = 1e-9, 1e-12
EPS_NEAR = [1e-9, 1e-7, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
HORIZON = 1000


def rss():
    return _proc.memory_info().rss / 1024**3


def main():
    print("P1c：実寸で t=0 の Q を直接測る（V_1 を保存しながら後ろ向きを回す）")
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    n_pairs = man["n_pairs"]
    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob, t_prob = A["succ"], A["prob"], A["t_prob"]
    state_idx = A["state_idx"].astype(np.int64)
    act_id = A["act_id"]
    owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(offs))
    t_owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(t_offs))
    is_dec = state_idx < dx.STRIDE_M
    dec_pairs = np.nonzero(is_dec)[0]
    dec_state = state_idx[dec_pairs]
    dec_act = act_id[dec_pairs]
    print(f"  読み込み完了  RSS {rss():.2f}GB")

    V_next = np.zeros(dx.TOTAL, dtype=np.float64)
    t0 = time.perf_counter()
    for t in range(HORIZON - 1, 0, -1):     # ★t=1 まで（t=0 は解かない）★
        if t == HORIZON - 1:
            acc = np.bincount(owner, weights=prob, minlength=n_pairs)
            acc += np.bincount(t_owner, weights=t_prob, minlength=n_pairs)
        else:
            acc = np.bincount(owner, weights=prob * (1.0 + V_next[succ]),
                              minlength=n_pairs)
            acc += np.bincount(t_owner, weights=t_prob, minlength=n_pairs)
        V_cur = np.zeros(dx.TOTAL, dtype=np.float64)
        V_cur[state_idx[~is_dec]] = acc[~is_dec]
        best = np.full(dx.STRIDE_M, -np.inf, dtype=np.float64)
        np.maximum.at(best, dec_state, acc[dec_pairs])
        V_cur[:dx.STRIDE_M] = np.where(np.isfinite(best), best, 0.0)
        V_next = V_cur
        if t % 200 == 0:
            print(f"  t={t:4d} {time.perf_counter()-t0:.0f}秒 RSS {rss():.2f}GB", flush=True)
    print(f"  ★V_1 まで到達（{time.perf_counter()-t0:.0f}秒）★")
    np.save(os.path.join(S, "V1.npy"), V_next)

    # ★t=0 の Q を作る★
    acc0 = np.bincount(owner, weights=prob * (1.0 + V_next[succ]), minlength=n_pairs)
    acc0 += np.bincount(t_owner, weights=t_prob, minlength=n_pairs)
    Q = np.full((dx.STRIDE_M, 4), np.nan, dtype=np.float64)
    Q[dec_state, dec_act] = acc0[dec_pairs]
    print()
    print("【t=0 の Q（W=0, N_S=12, N_L=3）】体力を刻んで")
    print("   体力 |       Q草 |     Q小物 |     Q大物 | 選択 |  1位-2位 |    g_rel | tie?")
    names = {1: "草", 2: "小物", 3: "大物"}
    rows = []
    for e in list(range(1, 2001)):
        i = dx.decision_index_of(e, 0, 12, 3)
        q = Q[i, 1:]
        if np.all(np.isnan(q)):
            continue
        order = np.argsort(-np.nan_to_num(q, nan=-np.inf))
        sel = int(order[0]) + 1
        v = np.sort(np.nan_to_num(q, nan=-np.inf))[::-1]
        gap = v[0] - v[1]
        denom = max(1.0, abs(v[0]), abs(v[1]))
        rows.append((e, q.copy(), sel, gap, gap / denom,
                     ATOL + RTOL * denom))
    prev, n_sw, shown = None, 0, 0
    for (e, q, sel, gap, g_rel, thr) in rows:
        if sel != prev:
            n_sw += 1
            if shown < 16:
                print(f" {e:6d} | {q[0]:9.5f} | {q[1]:9.5f} | {q[2]:9.5f}"
                      f" | {names[sel]:4s} | {gap:8.2e} | {g_rel:8.2e}"
                      f" | {'★tie★' if gap <= thr else ''}")
                shown += 1
            prev = sel
    print(f"  ★切り替わりの回数 {n_sw}★")
    gaps = np.array([r[3] for r in rows])
    grels = np.array([r[4] for r in rows])
    thrs = np.array([r[5] for r in rows])
    print()
    print("【判定】")
    print(f"  numerical tie（gap <= 閾値）: {int((gaps <= thrs).sum())}/{len(rows)}")
    for eps in EPS_NEAR:
        print(f"  g_rel <= {eps:.0e}: {int((grels <= eps).sum()):>5,}/{len(rows)}")
    print(f"  gap の中央値 {np.median(gaps):.3e}  最大 {gaps.max():.3e}")
    print(f"  g_rel の中央値 {np.median(grels):.3e}")
    sws = [r[0] for r in rows]
    # 切り替わり位置の間隔
    pos, prev = [], None
    for (e, q, sel, gap, g_rel, thr) in rows:
        if sel != prev:
            pos.append(e); prev = sel
    if len(pos) > 2:
        d = np.diff(pos)
        print(f"  切り替わりの間隔: 最小{d.min()} 最大{d.max()} 中央{int(np.median(d))}")
        print(f"    ★消耗 20 との一致を見る★")
        u, c = np.unique(d, return_counts=True)
        print("    間隔の分布:", {int(a): int(b) for a, b in zip(u, c)})


if __name__ == "__main__":
    main()
