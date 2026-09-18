# -*- coding: utf-8 -*-
"""実寸 World_1-G の遷移表を作り、Bellman を10層ぶん実測する。
★physics は world1g のまま（probe_kernel_size.branches_with_prob を使う）★
測るもの（ChatGPT 返答102の A〜E）
  A 遷移表の生成時間
  B 遷移表の RAM／容量
  C 1層の Bellman の時間
  D 10〜20層の定常スループット
  E 解いている間のピーク RSS
※ここで C の科学的な結果は出さない。実行可能かどうかだけを見る
"""
import os, sys, time
import numpy as np
import world1g as w
from world1g import State, DECISION, E_MAX
import dense_index as dx
from probe_kernel_size import branches_with_prob
import psutil

_proc = psutil.Process(os.getpid())


def rss():
    return _proc.memory_info().rss / 1024**3


def valid_states():
    """structural-valid な primitive 状態の index を返す（E>=1、大物0なら大物MODEなし）"""
    out = []
    for mi, mode in enumerate(dx.MODES):
        big = mode in (w.SEARCH_LARGE, w.COMBAT)
        for e in range(1, E_MAX + 1):
            for wnd in range(11):
                for ns in range(1, 13):
                    for nl in range(1 if big else 0, 4):
                        out.append(dx.to_index(mode, e, wnd, ns, nl))
    return np.array(out, dtype=np.int64)


def build_kernel(limit_pairs=None):
    """CSR 風の遷移表を作る。
    戻り値：
      pair_key : (state_index, action_id) の配列
      offs     : CSR の offsets
      succ     : 次状態の index（uint32）
      prob     : 確率（float64）
      t_offs, t_prob, t_cause, t_rawe : 終端の枝
    """
    ACT_ID = {None: 0, w.ACT_GRASS: 1, w.ACT_SMALL: 2, w.ACT_LARGE: 3}
    pairs, succ, prob = [], [], []
    t_pairs, t_prob, t_cause, t_rawe = [], [], [], []
    offs, t_offs = [0], [0]
    t0 = time.perf_counter()
    n = 0
    for mi, mode in enumerate(dx.MODES):
        big = mode in (w.SEARCH_LARGE, w.COMBAT)
        for e in range(1, E_MAX + 1):
            for wnd in range(11):
                for ns in range(1, 13):
                    for nl in range(1 if big else 0, 4):
                        i = dx.to_index(mode, e, wnd, ns, nl)
                        acts = [None]
                        if mode == DECISION:
                            s = State(e=e, w=wnd, n_small=ns, n_large=nl,
                                      t=0, mode=DECISION)
                            acts = list(w.legal_actions(s))
                        for a in acts:
                            nt, tm = branches_with_prob(mode, e, wnd, ns, nl, a)
                            pairs.append((i, ACT_ID[a]))
                            for j, p in nt:
                                succ.append(j); prob.append(p)
                            offs.append(len(succ))
                            for p, cause, raw in tm:
                                t_prob.append(p)
                                t_cause.append({None: 0, "starve": 1,
                                                "acute": 2, "tenju": 3}[cause])
                                t_rawe.append(raw)
                            t_offs.append(len(t_prob))
                            n += 1
                            if limit_pairs and n >= limit_pairs:
                                el = time.perf_counter() - t0
                                return (np.array(pairs, dtype=np.int64),
                                        np.array(offs, dtype=np.int64),
                                        np.array(succ, dtype=np.uint32),
                                        np.array(prob, dtype=np.float64),
                                        np.array(t_offs, dtype=np.int64),
                                        np.array(t_prob, dtype=np.float64),
                                        np.array(t_cause, dtype=np.uint8),
                                        np.array(t_rawe, dtype=np.int32), el, True)
    el = time.perf_counter() - t0
    return (np.array(pairs, dtype=np.int64),
            np.array(offs, dtype=np.int64),
            np.array(succ, dtype=np.uint32),
            np.array(prob, dtype=np.float64),
            np.array(t_offs, dtype=np.int64),
            np.array(t_prob, dtype=np.float64),
            np.array(t_cause, dtype=np.uint8),
            np.array(t_rawe, dtype=np.int32), el, False)


def make_owner(offs, n_pairs):
    """各枝がどの (state,action) の組に属するかの配列を作る（reduceat の代わり）"""
    cnt = np.diff(offs)
    return np.repeat(np.arange(n_pairs, dtype=np.int64), cnt)


def bellman_layer(V_next, owner, succ, prob, t_owner, t_prob, gamma,
                  reward_per_edge, n_pairs):
    """1層ぶん。★ここでは D0（報酬=1）で速度だけを測る★
    Σ_j p_j (r + γ V_next[j]) + Σ_terminal p·r
    ★reduceat ではなく bincount を使う（空の区間で落ちないため）★
    """
    contrib = prob * (reward_per_edge + gamma * V_next[succ])
    acc = np.bincount(owner, weights=contrib, minlength=n_pairs)
    if len(t_prob):
        acc += np.bincount(t_owner, weights=t_prob * reward_per_edge,
                           minlength=n_pairs)
    return acc


def main(limit=None, n_layers=10):
    print("実寸 World_1-G の遷移表と Bellman の実測")
    print(f"(state,action) の上限: {limit if limit else '全部'}／測る層数: {n_layers}")
    print()
    print("【A】遷移表の生成...", flush=True)
    (pairs, offs, succ, prob, t_offs, t_prob, t_cause, t_rawe, build_sec, cut) = \
        build_kernel(limit)
    print(f"  生成時間 {build_sec:.1f}秒{'（上限で打ち切り）' if cut else ''}")
    print(f"  (state,action) の組 {len(pairs):,}")
    print(f"  非終端の枝 {len(succ):,}  終端の枝 {len(t_prob):,}")
    print()
    print("【B】遷移表の容量")
    tot = 0
    for nm, arr in (("pairs", pairs), ("offs", offs), ("succ", succ),
                    ("prob", prob), ("t_offs", t_offs), ("t_prob", t_prob),
                    ("t_cause", t_cause), ("t_rawe", t_rawe)):
        print(f"  {nm:8s} {arr.nbytes/1024**3:7.3f} GB  dtype={arr.dtype}")
        tot += arr.nbytes
    print(f"  ★合計 {tot/1024**3:.3f} GB★   RSS {rss():.2f}GB")
    print()
    print(f"【C・D】Bellman を {n_layers} 層（D0 相当、報酬=1、γ=1）")
    n_pairs = len(pairs)
    owner = make_owner(offs, n_pairs)
    t_owner = make_owner(t_offs, n_pairs)
    print(f"  owner 配列 {owner.nbytes/1024**3:.3f} GB + {t_owner.nbytes/1024**3:.3f} GB"
          f"  RSS {rss():.2f}GB", flush=True)
    V = np.zeros(dx.TOTAL, dtype=np.float64)
    times = []
    for k in range(n_layers):
        t0 = time.perf_counter()
        acc = bellman_layer(V, owner, succ, prob, t_owner, t_prob, 1.0, 1.0, n_pairs)
        V2 = np.zeros(dx.TOTAL, dtype=np.float64)
        np.maximum.at(V2, pairs[:, 0], acc)
        V = V2
        el = time.perf_counter() - t0
        times.append(el)
        if k < 3 or k == n_layers - 1:
            print(f"  層 {k+1:3d}: {el:6.3f}秒  RSS {rss():.2f}GB", flush=True)
    med = sorted(times)[len(times)//2]
    print()
    print(f"【E】1層の中央値 {med:.3f}秒  ピークRSS {rss():.2f}GB")
    print(f"  枝の処理速度 {(len(succ)+len(t_prob))/med/1e6:.1f} M枝/秒")
    print(f"  1000層の見積もり {med*1000/60:.1f} 分")
    print(f"  96条件×1000層の見積もり {med*1000*96/3600:.1f} 時間")


if __name__ == "__main__":
    lim = None if len(sys.argv) < 2 or sys.argv[1] == "all" else int(sys.argv[1])
    main(lim, int(sys.argv[2]) if len(sys.argv) > 2 else 10)
