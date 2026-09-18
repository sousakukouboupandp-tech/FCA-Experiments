# -*- coding: utf-8 -*-
"""Gate B：実寸 kernel を正本（world1g）と照合する（ChatGPT 返答106の3）。
★端と普通の両方を意図的に含める（ランダムだけにしない）★
比較するもの：successor / 終端の理由 / 生の体力 / 確率
"""
import os, sys, json, time
import numpy as np
import world1g as w
from world1g import State, DECISION, E_MAX
import dense_index as dx
from probe_kernel_size import branches_with_prob
import psutil

_proc = psutil.Process(os.getpid())
D = "kernel_full"
ACT_ID = {None: 0, w.ACT_GRASS: 1, w.ACT_SMALL: 2, w.ACT_LARGE: 3}
CAUSE_ID = {None: 0, "starve": 1, "acute": 2, "tenju": 3}


def stratified_cases():
    """★端を意図的に含める★"""
    cases = []
    E_LIST = [1, 2, 19, 20, 21, 1979, 1999, 2000]
    W_LIST = [0, 1, 9, 10]
    NS_LIST = [1, 2, 12]
    NL_LIST = [0, 1, 2, 3]
    for mode in dx.MODES:
        big = mode in (w.SEARCH_LARGE, w.COMBAT)
        for e in E_LIST:
            for wnd in W_LIST:
                for ns in NS_LIST:
                    for nl in NL_LIST:
                        if big and nl == 0:
                            continue
                        cases.append((mode, e, wnd, ns, nl))
    return cases


def random_cases(n, seed=20260918):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        mode = dx.MODES[int(rng.integers(8))]
        big = mode in (w.SEARCH_LARGE, w.COMBAT)
        nl = int(rng.integers(1 if big else 0, 4))
        out.append((mode, int(rng.integers(1, E_MAX + 1)),
                    int(rng.integers(0, 11)), int(rng.integers(1, 13)), nl))
    return out


def pair_index_map(state_idx, act_id):
    """(state_index, act_id) → 組の番号"""
    key = state_idx.astype(np.int64) * 4 + act_id
    order = np.argsort(key, kind="stable")
    return key[order], order


def main(n_random=3000):
    print("Gate B：実寸 kernel を正本（world1g）と照合する")
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob = A["succ"], A["prob"]
    t_prob, t_cause, t_rawe = A["t_prob"], A["t_cause"], A["t_rawe"]
    print(f"  読み込み完了  RSS {_proc.memory_info().rss/1024**3:.2f}GB")
    keys, order = pair_index_map(A["state_idx"], A["act_id"])
    print("  索引を作った")
    print()

    def find_pair(i, a):
        k = i * 4 + ACT_ID[a]
        pos = np.searchsorted(keys, k)
        if pos >= len(keys) or keys[pos] != k:
            return None
        return int(order[pos])

    cases = stratified_cases()
    n_strat = len(cases)
    cases += random_cases(n_random)
    print(f"照合する状態: 端から {n_strat:,} 件 + 無作為 {n_random:,} 件")

    n_cmp, ng = 0, []
    max_perr = 0.0
    t0 = time.perf_counter()
    for (mode, e, wnd, ns, nl) in cases:
        acts = [None]
        if mode == DECISION:
            s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=0, mode=DECISION)
            acts = list(w.legal_actions(s))
        i = dx.to_index(mode, e, wnd, ns, nl)
        for a in acts:
            pid = find_pair(i, a)
            if pid is None:
                ng.append(f"組が見つからない {(mode,e,wnd,ns,nl,a)}")
                continue
            # ★正本をその場で呼ぶ★
            nt_ref, tm_ref = branches_with_prob(mode, e, wnd, ns, nl, a)
            ref_nt = {j: p for j, p in nt_ref}
            # kernel 側
            lo, hi = int(offs[pid]), int(offs[pid + 1])
            ker_nt = {int(succ[k]): float(prob[k]) for k in range(lo, hi)}
            if set(ref_nt) != set(ker_nt):
                ng.append(f"successor が違う {(mode,e,wnd,ns,nl,a)}")
            else:
                for j in ref_nt:
                    d = abs(ref_nt[j] - ker_nt[j])
                    max_perr = max(max_perr, d)
                    if d > 1e-15:
                        ng.append(f"確率が違う {(mode,e,wnd,ns,nl,a)} j={j} 差{d:.2e}")
            lo, hi = int(t_offs[pid]), int(t_offs[pid + 1])
            ker_tm = sorted((round(float(t_prob[k]), 18), int(t_cause[k]),
                             int(t_rawe[k])) for k in range(lo, hi))
            ref_tm = sorted((round(p, 18), CAUSE_ID[c], int(raw))
                            for p, c, raw in tm_ref)
            if len(ker_tm) != len(ref_tm):
                ng.append(f"終端の枝の数が違う {(mode,e,wnd,ns,nl,a)}")
            else:
                for (p1, c1, r1), (p2, c2, r2) in zip(ref_tm, ker_tm):
                    if c1 != c2 or r1 != r2 or abs(p1 - p2) > 1e-15:
                        ng.append(f"終端が違う {(mode,e,wnd,ns,nl,a)}")
            n_cmp += 1
    el = time.perf_counter() - t0
    print(f"  照合した (state,action) の組: {n_cmp:,}（{el:.1f}秒）")
    print(f"  確率の最大のずれ: {max_perr:.3e}")
    print()
    if ng:
        print(f"★不一致 {len(ng)} 件★")
        for s in ng[:20]:
            print("  ", s)
    else:
        print("★全件一致（successor / 終端の理由 / 生の体力 / 確率）★")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
