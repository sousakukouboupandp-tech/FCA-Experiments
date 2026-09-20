# -*- coding: utf-8 -*-
"""★D_S-exact（天寿到達確率の最大化）を実寸で解く★
SPEC_D0_DS比較（a4d7304）で凍結した通り。

★報酬の形に落とさず、V_t(s) = その状態から天寿に到達する確率 を直接解く★
  t=999 : その1歩を実行し、飢え死→0／急所死→0／生き残った枝→1（天寿）
          decision なら Q_999(s,a) = Σ_{生存した枝} p
          ★V_1000 は不要★
  t<999 : Q_t(s,a) = Σ_{非終端 s'} p(s'|s,a) · V_{t+1}(s')
          飢え死・急所死の枝は 0
★1000歩目でも急所死・飢え死が天寿より優先する★
★selected は D0 とまったく同じ凍結済みの tie 判定と固定順序（共通関数）★
"""
import os, sys, time, json, gc
import numpy as np
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
D = "kernel_full"
OUT = "solve_DS"
ATOL, RTOL = 1e-9, 1e-12
HORIZON = 1000


def rss():
    return _proc.memory_info().rss / 1024**3


def main():
    print("★★★D_S-exact（天寿到達確率の最大化）を実寸で解く★★★")
    print("凍結仕様 a4d7304 ／ 報酬化せず到達確率を直接解く ／ V_1000 は不要")
    print()
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    print(f"kernel: 組 {man['n_pairs']:,} / 非終端 {man['n_nonterminal']:,}"
          f" / 終端 {man['n_terminal']:,}")
    print(f"        SHA-256 {man['sha256_all_files_full'][:32]}")
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    n_pairs = man["n_pairs"]
    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob = A["succ"], A["prob"]
    t_prob, t_cause = A["t_prob"], A["t_cause"]
    state_idx = A["state_idx"].astype(np.int64)
    act_id = A["act_id"]
    print(f"読み込み完了  RSS {rss():.2f}GB")

    owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(offs))
    t_owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(t_offs))
    is_dec = state_idx < dx.STRIDE_M
    dec_pairs = np.nonzero(is_dec)[0]
    dec_state = state_idx[dec_pairs]
    dec_act = act_id[dec_pairs]
    print(f"decision の組 {len(dec_pairs):,} / 強制MODE の組 {n_pairs-len(dec_pairs):,}")

    os.makedirs(OUT, exist_ok=True)
    pol = np.memmap(os.path.join(OUT, "policy.dat"), dtype=np.uint8, mode="w+",
                    shape=(HORIZON, dx.STRIDE_M))
    opt = np.memmap(os.path.join(OUT, "optimal_set.dat"), dtype=np.uint8, mode="w+",
                    shape=(HORIZON, dx.STRIDE_M))
    print(f"方策ファイルを確保  RSS {rss():.2f}GB")
    print()

    V_next = np.zeros(dx.TOTAL, dtype=np.float64)
    times, peak = [], rss()
    t_start = time.perf_counter()
    for t in range(HORIZON - 1, -1, -1):
        t0 = time.perf_counter()
        if t == HORIZON - 1:
            # ★t=999：生き残った枝が1（天寿）、飢え死・急所死は0★
            acc = np.bincount(owner, weights=prob, minlength=n_pairs)
            # 終端の枝は天寿の判定を solver 側で行う。kernel の終端は
            # 飢え死(1) と急所死(2) のみなので、どちらも 0 → 足さない
        else:
            acc = np.bincount(owner, weights=prob * V_next[succ], minlength=n_pairs)
            # 終端の枝（飢え死・急所死）は 0 なので足さない

        V_cur = np.zeros(dx.TOTAL, dtype=np.float64)
        V_cur[state_idx[~is_dec]] = acc[~is_dec]
        best = np.full(dx.STRIDE_M, -np.inf, dtype=np.float64)
        np.maximum.at(best, dec_state, acc[dec_pairs])
        V_cur[:dx.STRIDE_M] = np.where(np.isfinite(best), best, 0.0)

        thr = ATOL + RTOL * np.maximum(1.0, np.abs(best[dec_state]))
        tie = acc[dec_pairs] >= best[dec_state] - thr
        bits = np.zeros(dx.STRIDE_M, dtype=np.uint8)
        np.bitwise_or.at(bits, dec_state[tie], (1 << (dec_act[tie] - 1)).astype(np.uint8))
        opt[t] = bits
        sel = np.where(bits & 1, 1, np.where(bits & 2, 2, np.where(bits & 4, 3, 0)))
        pol[t] = sel.astype(np.uint8)

        V_next = V_cur
        el = time.perf_counter() - t0
        times.append(el)
        peak = max(peak, rss())
        if t % 100 == 0 or t > HORIZON - 4:
            print(f"  t={t:4d} {el:5.2f}秒 RSS {rss():.2f}GB", flush=True)

    total = time.perf_counter() - t_start
    pol.flush(); opt.flush()
    print()
    print(f"★後ろ向き完了★ {total:.1f}秒（{total/60:.1f}分）")
    print(f"  1層の中央値 {sorted(times)[len(times)//2]:.3f}秒  ピークRSS {peak:.2f}GB")
    print()
    init = dx.decision_index_of(2000, 0, 12, 3)
    print("【初期状態（E=2000, W=0, N_S=12, N_L=3, t=0）】")
    print(f"  ★V0（天寿到達確率）= {V_next[init]:.9f}★")
    print(f"  selected = {int(pol[0][init])}（1=草, 2=小物, 3=大物）")
    print(f"  optimal_set bits = {int(opt[0][init]):03b}")
    print()
    print("  【参考】D0 の方策を使ったときの天寿到達確率 = 0.851102")
    print(f"  ★差 = {V_next[init] - 0.851102:+.9f}★")
    np.save(os.path.join(OUT, "V0.npy"), V_next)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(objective="D_S-exact", horizon=HORIZON,
                       kernel_sha=man["sha256_all_files_full"],
                       V_init=float(V_next[init]),
                       selected_init=int(pol[0][init]),
                       optimal_bits_init=int(opt[0][init]),
                       D0_policy_tenju=0.851102,
                       backward_sec=round(total, 1),
                       layer_median_sec=round(sorted(times)[len(times)//2], 3),
                       peak_rss_gb=round(peak, 2)), f, ensure_ascii=False, indent=2)
    del pol, opt
    gc.collect()
    print(f"  保存先 {OUT}/")


if __name__ == "__main__":
    main()
