# -*- coding: utf-8 -*-
"""D0 の前向き占有率（★実際の方策 policy.dat を使う。dummy ではない★）
SPEC_D0_DS比較（a4d7304）で凍結した順番の第1項目。

測るもの
  ・★E[T] = 998.520049 を独立に再構成できるか★（後ろ向きの検証）
  ・質量の保存（生存＋飢え死＋急所死＋天寿 = 1）
  ・天寿／飢え死／急所死の最終質量
  ・★W=10（あふれ）の占有率とあふれの質量★（異常②の材料）
  ・W ごとの占有率／MODE ごとの占有率／行動の占有率
"""
import os, sys, json, time, gc
import numpy as np
import world1g as w
from world1g import DECISION, HORIZON
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
D, S = "kernel_full", "solve_D0"


def rss():
    return _proc.memory_info().rss / 1024**3


def main():
    print("D0 の前向き占有率（★実際の方策を使う★）")
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    with open(os.path.join(S, "summary.json"), encoding="utf-8") as f:
        sm = json.load(f)
    print(f"  後ろ向きの V0 = {sm['V_init']:.6f} 歩 ★これを再構成できるかを見る★")
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    n_pairs = man["n_pairs"]
    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob = A["succ"], A["prob"]
    t_prob, t_cause = A["t_prob"], A["t_cause"]
    state_idx = A["state_idx"].astype(np.int64)
    act_id = A["act_id"]
    print(f"  kernel 読み込み  RSS {rss():.2f}GB")

    pair_of = np.full((dx.TOTAL, 4), -1, dtype=np.int64)
    pair_of[state_idx, act_id] = np.arange(n_pairs, dtype=np.int64)
    print(f"  組の索引  RSS {rss():.2f}GB")

    pol = np.memmap(os.path.join(S, "policy.dat"), dtype=np.uint8, mode="r",
                    shape=(HORIZON, dx.STRIDE_M))

    occ = np.zeros(dx.TOTAL, dtype=np.float64)
    occ[dx.to_index(DECISION, w.E_MAX, 0, 12, 3)] = 1.0
    d_starve = d_acute = d_tenju = 0.0
    lifespan = 0.0          # ★Σ_t P(その歩を生きた) = E[T]★
    w_occ = np.zeros(11, dtype=np.float64)     # W ごとの累積占有率
    mode_occ = np.zeros(8, dtype=np.float64)
    act_occ = np.zeros(4, dtype=np.float64)
    times, peak = [], rss()
    t0all = time.perf_counter()
    for t in range(HORIZON):
        t0 = time.perf_counter()
        alive = occ.sum()
        lifespan += alive          # ★その歩を生きた質量★
        # W ごと・MODE ごとの占有率を足す
        occ5 = occ.reshape(8, dx.N_E, dx.N_W, dx.N_S, dx.N_L)
        w_occ += occ5.sum(axis=(0, 1, 3, 4))
        mode_occ += occ5.sum(axis=(1, 2, 3, 4))

        mass = np.zeros(n_pairs, dtype=np.float64)
        forced = state_idx >= dx.STRIDE_M
        mass[forced] = occ[state_idx[forced]]
        sel = pol[t]
        for a in (1, 2, 3):
            idx_dec = np.nonzero(sel == a)[0]
            if len(idx_dec) == 0:
                continue
            pids = pair_of[idx_dec, a]
            good = pids >= 0
            mass[pids[good]] = occ[idx_dec[good]]
            act_occ[a] += occ[idx_dec[good]].sum()
        nxt = np.zeros(dx.TOTAL, dtype=np.float64)
        np.add.at(nxt, succ, np.repeat(mass, np.diff(offs)) * prob)
        tm = np.repeat(mass, np.diff(t_offs)) * t_prob
        d_starve += float(tm[t_cause == 1].sum())
        d_acute += float(tm[t_cause == 2].sum())
        if t + 1 >= HORIZON:
            d_tenju += float(nxt.sum())
            nxt[:] = 0.0
        occ = nxt
        el = time.perf_counter() - t0
        times.append(el)
        peak = max(peak, rss())
        tot = occ.sum() + d_starve + d_acute + d_tenju
        if t < 3 or t % 200 == 0 or t == HORIZON - 1:
            print(f"  t={t:4d} 生存 {occ.sum():.6f} 飢え {d_starve:.6f}"
                  f" 急所 {d_acute:.6f} 天寿 {d_tenju:.6f}"
                  f" 合計 {tot:.12f}  E[T]途中 {lifespan:.4f}"
                  f"  {el:.2f}秒 RSS {rss():.2f}GB", flush=True)
        if abs(tot - 1.0) > 1e-9:
            print(f"  ★質量が保存されていない: {tot}★")
            break
    el_all = time.perf_counter() - t0all
    print()
    print("【結果】")
    print(f"  ★E[T]（前向きで再構成）= {lifespan:.6f} 歩★")
    print(f"  ★後ろ向きの V0        = {sm['V_init']:.6f} 歩★")
    print(f"  ★差 = {abs(lifespan - sm['V_init']):.3e}★")
    print()
    print(f"  天寿   {d_tenju:.6f}")
    print(f"  飢え死 {d_starve:.6f}")
    print(f"  急所死 {d_acute:.6f}")
    print(f"  合計   {d_tenju + d_starve + d_acute:.12f}")
    print()
    print("【W ごとの累積占有率】★異常②の材料★")
    for wi in range(11):
        mark = " ★あふれ★" if wi == 10 else ""
        print(f"  W={wi:2d}: {w_occ[wi]:12.6f}  ({w_occ[wi]/w_occ.sum()*100:6.3f}%)"
              f"{mark}")
    print(f"  ★あふれ（W=10）の質量: {w_occ[10]:.6f}"
          f" = 全体の {w_occ[10]/w_occ.sum()*100:.4f}%★")
    print()
    print("【MODE ごとの累積占有率】")
    for mi, m in enumerate(dx.MODES):
        print(f"  {m:14s} {mode_occ[mi]:12.6f}  ({mode_occ[mi]/mode_occ.sum()*100:6.3f}%)")
    print()
    print("【行動の累積占有率（decision で選ばれた）】")
    for a, nm in ((1, "草"), (2, "小物"), (3, "大物")):
        print(f"  {nm:4s} {act_occ[a]:12.6f}  ({act_occ[a]/act_occ[1:].sum()*100:6.3f}%)")
    print()
    print(f"  {HORIZON} 歩 {el_all:.1f}秒  1歩の中央値 {sorted(times)[len(times)//2]:.3f}秒"
          f"  ピークRSS {peak:.2f}GB")
    with open(os.path.join(S, "forward_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(E_T_forward=lifespan, V0_backward=sm["V_init"],
                       diff=abs(lifespan - sm["V_init"]),
                       tenju=d_tenju, starve=d_starve, acute=d_acute,
                       w_occupancy=w_occ.tolist(),
                       overflow_mass=float(w_occ[10]),
                       overflow_frac=float(w_occ[10] / w_occ.sum()),
                       mode_occupancy=mode_occ.tolist(),
                       act_occupancy=act_occ.tolist(),
                       sec=el_all, peak_rss_gb=peak), f, ensure_ascii=False, indent=2)
    print(f"  保存: {S}/forward_summary.json")
    del pol
    gc.collect()


if __name__ == "__main__":
    main()
