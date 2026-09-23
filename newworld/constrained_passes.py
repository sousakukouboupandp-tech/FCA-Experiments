# -*- coding: utf-8 -*-
"""追補（SPEC_追補_同点の幅_凍結、a9defad）：行動を最適集合に制限した後ろ向き。
使い方: python constrained_passes.py D0   （P_max／P_min／P_eval）
        python constrained_passes.py DS   （V_DS／T_max／T_min／T_eval／T_max'）

core() は本番と縮小世界で共通。kernel は辞書で渡す：
  owner(非終端の枝→組) succ prob t_owner t_prob state_idx(組→状態) act_id n_pairs total n_dec
  decision 状態は index < n_dec。opt/pol は index 0..n_dec-1 の uint8（bit0=草 bit1=小物 bit2=大物）
★P_eval／T_eval は保存済みの policy をそのまま評価する（選び直さない）★
★毎層 assert：policy ∈ optimal_set、有効な decision 状態ごとに選ばれる組がちょうど1つ★
"""
import os, sys, json, time, gc
import numpy as np


def core(K, H, get_opt, get_pol, side, init_idx, keep_all=False, log=print):
    owner, succ, prob = K["owner"], K["succ"], K["prob"]
    n_pairs, total, n_dec = K["n_pairs"], K["total"], K["n_dec"]
    state_idx, act_id = K["state_idx"], K["act_id"]
    is_dec = state_idx < n_dec
    forced_p = np.nonzero(~is_dec)[0]
    forced_s = state_idx[forced_p]
    dec_p = np.nonzero(is_dec)[0]
    dec_s = state_idx[dec_p]
    dec_a = act_id[dec_p].astype(np.int64)
    has_pair = np.bincount(dec_s, minlength=n_dec) > 0
    alive = np.bincount(owner, weights=prob, minlength=n_pairs)          # Σ非終端 p
    term = np.bincount(K["t_owner"], weights=K["t_prob"], minlength=n_pairs)  # Σ終端 p

    names = ["P_max", "P_min", "P_eval"] if side == "D0" else \
            ["V_DS", "T_max", "T_min", "T_eval", "T_maxp"]
    nxt = {n: np.zeros(total) for n in names}
    hist = {n: {} for n in names} if keep_all else None
    t0 = time.perf_counter()
    for t in range(H - 1, -1, -1):
        bits = get_opt(t); sel = get_pol(t)
        v = sel > 0
        if not (((bits[v] >> (sel[v] - 1)) & 1) == 1).all():
            raise AssertionError(f"t={t}: policy が optimal_set の外")
        allowed = ((bits[dec_s] >> (dec_a - 1)) & 1).astype(bool)
        chosen = sel[dec_s] == dec_a
        cnt = np.bincount(dec_s[chosen], minlength=n_dec)
        if not (cnt[has_pair] == 1).all():
            raise AssertionError(f"t={t}: 選ばれる組がちょうど1つでない状態がある")

        def pair_val(X, lifespan):
            if t == H - 1:
                return alive + term if lifespan else alive.copy()
            a = np.bincount(owner, weights=prob * X[succ], minlength=n_pairs)
            return a + alive + term if lifespan else a

        def reduce(acc, mode, mask=None):
            out = np.zeros(total)
            out[forced_s] = acc[forced_p]
            if mode == "eval":
                out[dec_s[chosen]] = acc[dec_p][chosen]
                return out
            m = allowed if mask is None else mask
            fill = -np.inf if mode in ("max", "vmax") else np.inf
            best = np.full(n_dec, fill)
            src = np.ones(len(dec_p), bool) if mode == "vmax" else m
            (np.maximum if fill < 0 else np.minimum).at(best, dec_s[src], acc[dec_p][src])
            out[:n_dec] = np.where(np.isfinite(best), best, 0.0)
            return out

        cur = {}
        if side == "D0":
            cur["P_max"] = reduce(pair_val(nxt["P_max"], False), "max")
            cur["P_min"] = reduce(pair_val(nxt["P_min"], False), "min")
            cur["P_eval"] = reduce(pair_val(nxt["P_eval"], False), "eval")
        else:
            cur["V_DS"] = reduce(pair_val(nxt["V_DS"], False), "vmax")   # 制限なしの天寿確率
            cur["T_max"] = reduce(pair_val(nxt["T_max"], True), "max")
            cur["T_min"] = reduce(pair_val(nxt["T_min"], True), "min")
            cur["T_eval"] = reduce(pair_val(nxt["T_eval"], True), "eval")
            # T_max'：★V_DS がちょうど 0.0 の状態でだけ★許容集合の中で寿命最大、それ以外は保存方策
            accp = pair_val(nxt["T_maxp"], True)
            doomed_s = (cur["V_DS"][:n_dec] == 0.0) & has_pair
            doomed_p = doomed_s[dec_s]
            out = reduce(accp, "eval")
            best = np.full(n_dec, -np.inf)
            m = allowed & doomed_p
            np.maximum.at(best, dec_s[m], accp[dec_p][m])
            out[:n_dec] = np.where(doomed_s, best, out[:n_dec])
            cur["T_maxp"] = out
        nxt = cur
        if keep_all:
            for n in names:
                hist[n][t] = cur[n][:n_dec].copy()
        if t % 100 == 0:
            log(f"  t={t:4d}  {time.perf_counter()-t0:.0f}秒")
    res = {n: float(nxt[n][init_idx]) for n in names}
    return res, hist


def main(side):
    import dense_index as dx
    import psutil
    proc = psutil.Process(os.getpid())
    D = "kernel_full"
    S = {"D0": "solve_D0", "DS": "solve_DS"}[side]
    H = 1000
    print(f"追補の制限つき後ろ向き：{side} 側（{S} の optimal_set と policy）")
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    n_pairs = man["n_pairs"]
    K = dict(owner=np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(A["offs"])),
             succ=A["succ"], prob=A["prob"],
             t_owner=np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(A["t_offs"])),
             t_prob=A["t_prob"], state_idx=A["state_idx"].astype(np.int64),
             act_id=A["act_id"], n_pairs=n_pairs, total=dx.TOTAL, n_dec=dx.STRIDE_M)
    pol = np.memmap(os.path.join(S, "policy.dat"), dtype=np.uint8, mode="r", shape=(H, dx.STRIDE_M))
    opt = np.memmap(os.path.join(S, "optimal_set.dat"), dtype=np.uint8, mode="r", shape=(H, dx.STRIDE_M))
    init = dx.decision_index_of(2000, 0, 12, 3)
    print(f"  準備完了  RSS {proc.memory_info().rss/1024**3:.2f}GB", flush=True)
    t0 = time.perf_counter()
    res, _ = core(K, H, lambda t: np.asarray(opt[t]), lambda t: np.asarray(pol[t]), side, init,
                  log=lambda s: print(s + f"  RSS {proc.memory_info().rss/1024**3:.2f}GB", flush=True))
    el = time.perf_counter() - t0
    print()
    print("【初期状態の値】")
    for k, v in res.items():
        print(f"  {k:7s} = {v:.12f}")
    out = dict(side=side, values=res, sec=el, peak_rss_gb=proc.memory_info().rss/1024**3)
    with open(os.path.join(S, f"constrained_{side}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  {el:.1f}秒  保存：{S}/constrained_{side}.json")
    del pol, opt; gc.collect()


if __name__ == "__main__":
    main(sys.argv[1])
