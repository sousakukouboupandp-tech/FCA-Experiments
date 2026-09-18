# -*- coding: utf-8 -*-
"""実寸 World_1-G の solver 用遷移表の大きさを実測する（ChatGPT 返答102）。
★physics は world1g のまま。枝を数えて容量を見積もる★
solver では reachability と違い、次の情報が必要：
  非終端の枝：successor index + probability
  終端の枝  ：probability + 終端の理由 + 終端の生の体力
まず「状態あたりの枝の数」を標本で測り、全体を見積もる。
そのあと、valid な状態全部について実測する（時間がかかるので段階的に）。
"""
import os, sys, time, itertools
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX)
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
HEAL_P = w.HEAL_P
REC_P = w.LARGE_RECOVER_P


def binom_pmf(n, k, p):
    from math import comb
    return comb(n, k) * p**k * (1 - p)**(n - k)


def branches_with_prob(mode, e, wnd, ns, nl, action=None):
    """★確率つきの枝★（solver 用）。world1g.transition をそのまま呼ぶ。
    戻り値: (非終端の枝のリスト[(idx, p)], 終端の枝のリスト[(p, cause, raw_e)])
    """
    base = State(e=e, w=wnd, n_small=ns, n_large=nl, t=0, mode=mode)
    run_mode = w.start_mode(action) if mode == DECISION else mode
    heals = [(binom_pmf(wnd, k, HEAL_P), k) for k in range(wnd + 1)] if wnd else [(1.0, 0)]
    recs = ([(REC_P, True), (1 - REC_P, False)] if nl in (1, 2) else [(1.0, False)])
    if run_mode in (SEARCH_SMALL, w.SEARCH_LARGE):
        pf = (w.small_search_p(ns) if run_mode == SEARCH_SMALL
              else w.large_search_p(nl))
        acts = [(pf, dict(find=True)), (1 - pf, dict(find=False))]
    elif run_mode == CHASE_4:
        acts = [(w.CATCH_P, dict(catch=True)), (1 - w.CATCH_P, dict(catch=False))]
    elif run_mode == COMBAT:
        acts = [(w.COMBAT_KILL, dict(combat="kill")),
                (w.COMBAT_ACUTE, dict(combat="acute")),
                (w.COMBAT_WOUND, dict(combat="wound")),
                (w.COMBAT_MISS, dict(combat="miss"))]
    else:
        acts = [(1.0, dict())]

    nonterm, term = {}, []
    for ph, h in heals:
        for pr, rc in recs:
            for pa, d in acts:
                p = ph * pr * pa
                if p <= 0.0:
                    continue
                nx = transition(base, action, Draws(heal=h, recover=rc,
                                                    find=d.get("find", False),
                                                    catch=d.get("catch", False),
                                                    combat=d.get("combat")))
                if nx.alive:
                    i = dx.to_index(nx.mode, nx.e, nx.w, nx.n_small, nx.n_large)
                    nonterm[i] = nonterm.get(i, 0.0) + p     # ★同じ index は確率を合算★
                else:
                    term.append((p, nx.dead, nx.e))
    return list(nonterm.items()), term


def main(n_sample=3000):
    print("実寸 World_1-G の solver 用遷移表の大きさを実測する")
    print(f"標本 {n_sample:,} 状態（valid なものから無作為）")
    print()
    rng = np.random.default_rng(20260918)
    n_nt, n_tm, n_sa = 0, 0, 0
    max_nt = 0
    t0 = time.perf_counter()
    prob_err = 0.0
    while n_sa < n_sample:
        mi = int(rng.integers(8))
        mode = dx.MODES[mi]
        e = int(rng.integers(1, E_MAX + 1))
        wnd = int(rng.integers(0, 11))
        ns = int(rng.integers(1, 13))
        nl = int(rng.integers(0, 4))
        if mode in (w.SEARCH_LARGE, w.COMBAT) and nl == 0:
            continue
        acts = [None]
        if mode == DECISION:
            s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=0, mode=DECISION)
            acts = list(w.legal_actions(s))
        for a in acts:
            nt, tm = branches_with_prob(mode, e, wnd, ns, nl, a)
            tot = sum(p for _, p in nt) + sum(p for p, _, _ in tm)
            prob_err = max(prob_err, abs(tot - 1.0))
            n_nt += len(nt)
            n_tm += len(tm)
            max_nt = max(max_nt, len(nt))
            n_sa += 1
    el = time.perf_counter() - t0
    print(f"(state, action) の組 {n_sa:,} 件、{el:.1f}秒")
    print(f"  確率の合計の最大のずれ: {prob_err:.2e}")
    print(f"  非終端の枝：平均 {n_nt/n_sa:.2f} 本（最大 {max_nt}）")
    print(f"  終端の枝　：平均 {n_tm/n_sa:.2f} 本")
    print(f"  1組あたりの処理時間: {el/n_sa*1e6:.1f} マイクロ秒")
    print()

    # 実寸の (state, action) の組の数を数える
    valid_per_mode = {}
    for mi, mode in enumerate(dx.MODES):
        if mode in (w.SEARCH_LARGE, w.COMBAT):
            valid_per_mode[mode] = 2000 * 11 * 12 * 3     # E=1..2000, N_L=1..3
        else:
            valid_per_mode[mode] = 2000 * 11 * 12 * 4
    n_states = sum(valid_per_mode.values())
    # decision は行動が2または3
    n_dec_2 = 2000 * 11 * 12 * 1          # N_L=0 → 2行動
    n_dec_3 = 2000 * 11 * 12 * 3          # N_L>=1 → 3行動
    n_pairs = (n_dec_2 * 2 + n_dec_3 * 3
               + sum(v for m2, v in valid_per_mode.items() if m2 != DECISION))
    print(f"structural-valid な primitive 状態: {n_states:,}")
    print(f"(state, action) の組の総数: {n_pairs:,}")
    print()
    nt_avg, tm_avg = n_nt / n_sa, n_tm / n_sa
    tot_nt = n_pairs * nt_avg
    tot_tm = n_pairs * tm_avg
    print("遷移表の見積もり")
    print(f"  非終端の枝の総数: {tot_nt:,.0f}")
    print(f"  終端の枝の総数　: {tot_tm:,.0f}")
    print()
    b_idx, b_prob = 4, 8            # uint32 / float64
    b_term = 8 + 1 + 4              # prob + cause(1byte) + raw_e(int32)
    size = tot_nt * (b_idx + b_prob) + tot_tm * b_term + n_pairs * 8
    print(f"  非終端: {tot_nt*(b_idx+b_prob)/1024**3:.2f} GB（index 4B + 確率 8B）")
    print(f"  終端　: {tot_tm*b_term/1024**3:.2f} GB（確率 8B + 理由 1B + 生の体力 4B）")
    print(f"  CSR の offsets: {n_pairs*8/1024**3:.2f} GB")
    print(f"  ★合計 {size/1024**3:.2f} GB★")
    print()
    print(f"  確率を float32 にすると: {(tot_nt*(4+4) + tot_tm*(4+1+4) + n_pairs*8)/1024**3:.2f} GB")
    print()
    print(f"  遷移表の生成時間の見積もり: {n_pairs * el/n_sa/3600:.2f} 時間")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
