# -*- coding: utf-8 -*-
"""前向き評価（汎用）。forward_D0.py と同じ処理で、読む方策と照合する値だけを変える。
使い方: python forward_eval.py DS   （solve_DS/policy.dat を評価）
        python forward_eval.py D0   （solve_D0/policy.dat を評価）
照合：D0 → 期待寿命 E[T] を後ろ向きの V0 と／DS → 天寿率を後ろ向きの V0 と
★追補（a9defad）：保存された方策の行動が optimal_set の中にあることを毎時刻 assert★
"""
import os, sys, json, time, gc
import numpy as np
import world1g as w
from world1g import DECISION, HORIZON
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
D = "kernel_full"


def rss():
    return _proc.memory_info().rss / 1024**3


def main(obj):
    S = {"D0": "solve_D0", "DS": "solve_DS"}[obj]
    print(f"前向き評価：{obj}（{S}/policy.dat）")
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    with open(os.path.join(S, "summary.json"), encoding="utf-8") as f:
        sm = json.load(f)
    print(f"  後ろ向きの V0 = {sm['V_init']:.12f}")
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    n_pairs = man["n_pairs"]
    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob = A["succ"], A["prob"]
    t_prob, t_cause = A["t_prob"], A["t_cause"]
    state_idx = A["state_idx"].astype(np.int64)
    act_id = A["act_id"]
    pair_of = np.full((dx.TOTAL, 4), -1, dtype=np.int64)
    pair_of[state_idx, act_id] = np.arange(n_pairs, dtype=np.int64)
    forced = state_idx >= dx.STRIDE_M
    rep_nt = np.diff(offs)
    rep_t = np.diff(t_offs)
    print(f"  準備完了  RSS {rss():.2f}GB", flush=True)

    pol = np.memmap(os.path.join(S, "policy.dat"), dtype=np.uint8, mode="r",
                    shape=(HORIZON, dx.STRIDE_M))
    opt = np.memmap(os.path.join(S, "optimal_set.dat"), dtype=np.uint8, mode="r",
                    shape=(HORIZON, dx.STRIDE_M))
    occ = np.zeros(dx.TOTAL, dtype=np.float64)
    occ[dx.to_index(DECISION, w.E_MAX, 0, 12, 3)] = 1.0
    d_starve = d_acute = d_tenju = 0.0
    lifespan = 0.0
    w_occ = np.zeros(11); mode_occ = np.zeros(8); act_occ = np.zeros(4)
    t0all = time.perf_counter(); peak = rss()
    for t in range(HORIZON):
        # ★保存された方策 ∈ optimal_set の確認★
        sel = np.asarray(pol[t]); bits = np.asarray(opt[t])
        v = sel > 0
        ok = ((bits[v] >> (sel[v] - 1)) & 1) == 1
        if not ok.all():
            print(f"  ★停止：t={t} で方策が optimal_set の外（{int((~ok).sum())} 状態）★")
            return
        lifespan += occ.sum()
        occ5 = occ.reshape(8, dx.N_E, dx.N_W, dx.N_S, dx.N_L)
        w_occ += occ5.sum(axis=(0, 1, 3, 4)); mode_occ += occ5.sum(axis=(1, 2, 3, 4))
        mass = np.zeros(n_pairs)
        mass[forced] = occ[state_idx[forced]]
        for a in (1, 2, 3):
            idx_dec = np.nonzero(sel == a)[0]
            pids = pair_of[idx_dec, a]; good = pids >= 0
            mass[pids[good]] = occ[idx_dec[good]]
            act_occ[a] += occ[idx_dec[good]].sum()
        nxt = np.zeros(dx.TOTAL)
        np.add.at(nxt, succ, np.repeat(mass, rep_nt) * prob)
        tm = np.repeat(mass, rep_t) * t_prob
        d_starve += float(tm[t_cause == 1].sum()); d_acute += float(tm[t_cause == 2].sum())
        if t + 1 >= HORIZON:
            d_tenju += float(nxt.sum()); nxt[:] = 0.0
        occ = nxt
        peak = max(peak, rss())
        tot = occ.sum() + d_starve + d_acute + d_tenju
        if t % 200 == 0 or t == HORIZON - 1:
            print(f"  t={t:4d} 生存 {occ.sum():.6f} 飢え {d_starve:.6f} 急所 {d_acute:.6f}"
                  f" 天寿 {d_tenju:.6f} 合計 {tot:.12f}  RSS {rss():.2f}GB", flush=True)
        if abs(tot - 1.0) > 1e-9:
            print(f"  ★停止：質量が保存されていない {tot}★"); return
    el = time.perf_counter() - t0all
    target = d_tenju if obj == "DS" else lifespan
    diff = abs(target - sm["V_init"])
    print()
    print("【結果】")
    print(f"  期待寿命 E[T] = {lifespan:.12f}")
    print(f"  天寿 {d_tenju:.12f} ／ 飢え死 {d_starve:.12f} ／ 急所死 {d_acute:.12f}")
    print(f"  ★照合（{'天寿率' if obj == 'DS' else '期待寿命'}）：前向き {target:.12f}"
          f" ／ 後ろ向き {sm['V_init']:.12f} ／ 差 {diff:.3e}★")
    print(f"  W=10 の累積占有率 {w_occ[10]:.9f}（全体の {w_occ[10]/w_occ.sum()*100:.3e}%）")
    print(f"  行動：草 {act_occ[1]/act_occ[1:].sum()*100:.3f}% ／ 小物 {act_occ[2]/act_occ[1:].sum()*100:.3f}%"
          f" ／ 大物 {act_occ[3]/act_occ[1:].sum()*100:.4f}%")
    print(f"  {el:.1f}秒  ピークRSS {peak:.2f}GB")
    with open(os.path.join(S, "forward_summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(objective=obj, E_T=lifespan, tenju=d_tenju, starve=d_starve,
                       acute=d_acute, V0_backward=sm["V_init"], check_diff=diff,
                       w_occupancy=w_occ.tolist(), mode_occupancy=mode_occ.tolist(),
                       act_occupancy=act_occ.tolist(), sec=el, peak_rss_gb=peak),
                  f, ensure_ascii=False, indent=2)
    del pol, opt; gc.collect()


if __name__ == "__main__":
    main(sys.argv[1])
