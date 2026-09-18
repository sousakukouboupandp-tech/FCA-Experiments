# -*- coding: utf-8 -*-
"""保存した実寸の遷移表を読み、Bellman を実測する（ChatGPT 返答104の6）。
測るもの：1層目と2層目以降を分ける／中央値・最小・最大／ピーク RSS／枝の処理速度
★ここでは科学的な結果を出さない。実行可能かどうかだけ★
"""
import os, sys, time, json
import numpy as np
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
D = "kernel_full"


def rss():
    return _proc.memory_info().rss / 1024**3


def main(n_layers=10):
    print("実寸 World_1-G の Bellman 実測（保存済みの遷移表を読む）")
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    print(f"  組 {man['n_pairs']:,} / 非終端 {man['n_nonterminal']:,}"
          f" / 終端 {man['n_terminal']:,}")
    print(f"  digest {man['sha256_all_arrays'][:16]}")
    print()
    t0 = time.perf_counter()
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    print(f"読み込み {time.perf_counter()-t0:.1f}秒  RSS {rss():.2f}GB")

    n_pairs = man["n_pairs"]
    # owner 配列（各枝がどの組に属するか）
    t0 = time.perf_counter()
    owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(A["offs"]))
    t_owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(A["t_offs"]))
    print(f"owner 配列 {time.perf_counter()-t0:.1f}秒 "
          f"{(owner.nbytes+t_owner.nbytes)/1024**3:.3f} GB  RSS {rss():.2f}GB")
    print()

    succ, prob = A["succ"], A["prob"]
    t_prob = A["t_prob"]
    state_idx = A["state_idx"].astype(np.int64)
    V = np.zeros(dx.TOTAL, dtype=np.float64)
    reward = 1.0          # D0 相当（その歩を生きたら +1）。速度の測定用
    gamma = 1.0
    times = []
    peak = rss()
    print(f"Bellman を {n_layers} 層")
    for k in range(n_layers):
        t0 = time.perf_counter()
        contrib = prob * (reward + gamma * V[succ])
        acc = np.bincount(owner, weights=contrib, minlength=n_pairs)
        acc += np.bincount(t_owner, weights=t_prob * reward, minlength=n_pairs)
        V2 = np.zeros(dx.TOTAL, dtype=np.float64)
        np.maximum.at(V2, state_idx, acc)
        V = V2
        el = time.perf_counter() - t0
        times.append(el)
        peak = max(peak, rss())
        print(f"  層 {k+1:3d}: {el:6.3f}秒  RSS {rss():.2f}GB", flush=True)
    print()
    ts = sorted(times)
    print(f"1層目 {times[0]:.3f}秒")
    print(f"2層目以降  中央値 {sorted(times[1:])[len(times[1:])//2]:.3f}秒"
          f"  最小 {min(times[1:]):.3f}秒  最大 {max(times[1:]):.3f}秒")
    med = sorted(times[1:])[len(times[1:])//2]
    edges = man["n_nonterminal"] + man["n_terminal"]
    print(f"ピーク RSS {peak:.2f}GB")
    print(f"枝の処理速度 {edges/med/1e6:.1f} M枝/秒")
    print()
    print(f"1000層の見積もり {med*1000/60:.1f} 分")
    print(f"96条件×1000層の見積もり {med*1000*96/3600:.1f} 時間")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
