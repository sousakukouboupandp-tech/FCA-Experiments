# -*- coding: utf-8 -*-
"""★★★C を開く：D0（期待寿命の最大化）を実寸で解く★★★
凍結仕様：SOLVER_FINAL_実験炉Ver1（955916a）

D0：報酬はその歩を生きたら +1（死んだ歩も1歩として数える）／γ=1
　　合計が寿命に一致する

後ろ向き Bellman（t=999 → 0）
  ・価値は current / next の2層
  ・decision の方策だけを1000時刻ぶん memmap に保存
  ・optimal_set と selected_for_execution を保存
  ・★t=999 の非終端は天寿（V_1000 を参照しない）★
  ・numerical tie: |Qa-Qb| <= 1e-9 + 1e-12*max(1,|Qa|,|Qb|)
"""
import os, sys, time, json, gc
import numpy as np
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
D = "kernel_full"
OUT = "solve_D0"
ATOL, RTOL = 1e-9, 1e-12
HORIZON = 1000


def rss():
    return _proc.memory_info().rss / 1024**3


def main():
    print("★★★D0（期待寿命の最大化）を実寸で解く★★★")
    print("凍結仕様 955916a ／ 報酬=その歩を生きたら+1 ／ γ=1")
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
    t_prob = A["t_prob"]
    state_idx = A["state_idx"].astype(np.int64)
    act_id = A["act_id"]
    print(f"読み込み完了  RSS {rss():.2f}GB")

    owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(offs))
    t_owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(t_offs))
    print(f"owner 配列  RSS {rss():.2f}GB")

    # decision の組だけを取り出す（行動ごと）
    is_dec = state_idx < dx.STRIDE_M
    dec_pairs = np.nonzero(is_dec)[0]
    dec_state = state_idx[dec_pairs]          # decision の index（0..STRIDE_M-1）
    dec_act = act_id[dec_pairs]               # 1=草, 2=小物, 3=大物
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
            # ★t=999：非終端はすべて天寿。V_1000 を参照しない★
            # 報酬はその歩を生きたら +1（天寿でも1歩は生きた）
            acc = np.bincount(owner, weights=prob, minlength=n_pairs)
            acc += np.bincount(t_owner, weights=t_prob, minlength=n_pairs)
        else:
            acc = np.bincount(owner, weights=prob * (1.0 + V_next[succ]),
                              minlength=n_pairs)
            acc += np.bincount(t_owner, weights=t_prob, minlength=n_pairs)

        V_cur = np.zeros(dx.TOTAL, dtype=np.float64)
        # 強制 MODE：そのまま
        V_cur[state_idx[~is_dec]] = acc[~is_dec]
        # decision：行動の最大
        best = np.full(dx.STRIDE_M, -np.inf, dtype=np.float64)
        np.maximum.at(best, dec_state, acc[dec_pairs])
        V_cur[:dx.STRIDE_M] = np.where(np.isfinite(best), best, 0.0)

        # 方策と optimal_set
        thr = ATOL + RTOL * np.maximum(1.0, np.abs(best[dec_state]))
        tie = acc[dec_pairs] >= best[dec_state] - thr
        bits = np.zeros(dx.STRIDE_M, dtype=np.uint8)
        np.bitwise_or.at(bits, dec_state[tie], (1 << (dec_act[tie] - 1)).astype(np.uint8))
        opt[t] = bits
        # selected：固定順序（草=bit0 → 小物=bit1 → 大物=bit2）
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
    # ★初期状態の価値と方策★
    init = dx.decision_index_of(2000, 0, 12, 3)
    print(f"【初期状態（E=2000, W=0, N_S=12, N_L=3, t=0）】")
    print(f"  ★V0（期待寿命）= {V_next[init]:.6f} 歩★")
    print(f"  selected = {int(pol[0][init])}（1=草, 2=小物, 3=大物）")
    print(f"  optimal_set bits = {int(opt[0][init]):03b}（草/小物/大物）")
    np.save(os.path.join(OUT, "V0.npy"), V_next)
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(objective="D0", gamma=1.0, horizon=HORIZON,
                       kernel_sha=man["sha256_all_files_full"],
                       V_init=float(V_next[init]),
                       selected_init=int(pol[0][init]),
                       optimal_bits_init=int(opt[0][init]),
                       backward_sec=round(total, 1),
                       layer_median_sec=round(sorted(times)[len(times)//2], 3),
                       peak_rss_gb=round(peak, 2)), f, ensure_ascii=False, indent=2)
    del pol, opt
    gc.collect()
    print(f"  保存先 {OUT}/")


if __name__ == "__main__":
    main()
