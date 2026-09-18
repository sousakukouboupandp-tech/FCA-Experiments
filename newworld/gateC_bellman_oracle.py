# -*- coding: utf-8 -*-
"""Gate C：実寸 kernel の1層 Bellman を正本と照合する（ChatGPT 返答106の4）。
★V_next=0 だけでは index の誤りを見逃すので、index 依存の値と乱数を使う★
  V0: すべて0
  V1: index の決定的な関数
  V2: 固定 seed の疑似乱数
正本側は world1g.transition を直接呼んで Σ p (r + γ V_next(s')) を逐次計算する。
"""
import os, sys, json, time
import numpy as np
import world1g as w
from world1g import State, DECISION, E_MAX
import dense_index as dx
from probe_kernel_size import branches_with_prob
from gateB_kernel_vs_canonical import (stratified_cases, random_cases,
                                        pair_index_map, ACT_ID, CAUSE_ID)
import psutil

_proc = psutil.Process(os.getpid())
D = "kernel_full"
GAMMA = 0.99
REWARD = 1.0          # D0 相当（その歩を生きたら +1）


def make_V(kind):
    if kind == "V0":
        return np.zeros(dx.TOTAL, dtype=np.float64)
    if kind == "V1":
        # ★index の決定的な関数（index が1つずれたら必ず値が変わる）★
        return (np.arange(dx.TOTAL, dtype=np.float64) % 997) * 0.001
    if kind == "V2":
        rng = np.random.default_rng(20260918)
        return rng.random(dx.TOTAL) * 100.0
    raise ValueError(kind)


def main(n_random=2000):
    print("Gate C：1層 Bellman を正本と照合する")
    print(f"  γ={GAMMA}、報酬={REWARD}（D0 相当）")
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob = A["succ"], A["prob"]
    t_prob = A["t_prob"]
    keys, order = pair_index_map(A["state_idx"], A["act_id"])
    n_pairs = man["n_pairs"]
    print(f"  読み込み完了  RSS {_proc.memory_info().rss/1024**3:.2f}GB")

    def find_pair(i, a):
        k = i * 4 + ACT_ID[a]
        pos = np.searchsorted(keys, k)
        if pos >= len(keys) or keys[pos] != k:
            return None
        return int(order[pos])

    cases = stratified_cases() + random_cases(n_random, seed=20260919)
    print(f"  照合する状態: {len(cases):,} 件")
    print()

    owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(offs))
    t_owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(t_offs))

    print("V_next の種類 | 照合数 | max_abs_diff | max_rel_diff | argmax")
    all_ok = True
    for kind in ("V0", "V1", "V2"):
        V = make_V(kind)
        # ★dense 側：1層ぶんを一括で計算★
        contrib = prob * (REWARD + GAMMA * V[succ])
        acc = np.bincount(owner, weights=contrib, minlength=n_pairs)
        acc += np.bincount(t_owner, weights=t_prob * REWARD, minlength=n_pairs)
        n, mx_abs, mx_rel, arg = 0, 0.0, 0.0, None
        for (mode, e, wnd, ns, nl) in cases:
            acts = [None]
            if mode == DECISION:
                s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=0, mode=DECISION)
                acts = list(w.legal_actions(s))
            i = dx.to_index(mode, e, wnd, ns, nl)
            for a in acts:
                pid = find_pair(i, a)
                if pid is None:
                    continue
                # ★正本側：逐次計算★
                nt, tm = branches_with_prob(mode, e, wnd, ns, nl, a)
                ref = 0.0
                for j, p in nt:
                    ref += p * (REWARD + GAMMA * V[j])
                for p, cause, raw in tm:
                    ref += p * REWARD
                d = abs(ref - acc[pid])
                rel = d / max(1.0, abs(ref), abs(acc[pid]))
                if d > mx_abs:
                    mx_abs, arg = d, (mode, e, wnd, ns, nl, a)
                mx_rel = max(mx_rel, rel)
                n += 1
        ok = mx_abs < 1e-9
        all_ok = all_ok and ok
        print(f"{kind:13s} | {n:6,} | {mx_abs:12.3e} | {mx_rel:12.3e} |"
              f" {str(arg)[:40]:40s} {'OK' if ok else '★NG★'}")
    print()
    print("★全ての V_next で1層 Bellman が正本と一致★" if all_ok else "★不一致あり★")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
