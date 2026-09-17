# -*- coding: utf-8 -*-
"""到達可能集合の support 列挙（continuation memoization 版）と、
旧版（reach_probe2 と同じ辿り方）との厳密一致の検証。

計器の定義（ChatGPT 返答58の9で整理）
  raw_internal   : 内部1歩の展開回数（step_support の呼び出し回数）
  raw_landing    : decision へ着地した事象の数（重複込み）
  uniq_landing   : その層で着地した一意な (e,w,ns,nl,τ) の数
  uniq_successor : 次の decision 状態として一意なもの（層に入った数）

digest: 各 S_t を正順に並べて固定書式で連結し SHA-256。
        Python の hash は使わない（プロセスごとに変わるため）
"""
import os, sys, time, itertools, hashlib
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX, HORIZON)
import psutil

K_WOUND = 10
TAU_MAX = 100
_proc = psutil.Process(os.getpid())
STAT = dict(raw_internal=0, raw_landing=0, cont_hit=0, cont_miss=0,
            step_hit=0, step_miss=0)


def cap_w(x):
    return min(x, K_WOUND)


def digest_of(states):
    """S_t の決定論的ダイジェスト。正順に並べて固定書式で連結する"""
    h = hashlib.sha256()
    for rec in sorted(states):
        h.update(("%d,%d,%d,%d;" % rec).encode("ascii"))
    return h.hexdigest()[:16]


_step = {}


def step_support(x):
    """内部状態 x=(mode,e,w,ns,nl) の1歩。(続く枝, decisionへ戻る枝)"""
    hit = _step.get(x)
    if hit is not None:
        STAT["step_hit"] += 1
        return hit
    STAT["step_miss"] += 1
    mode, e, wnd, ns, nl = x
    base = State(e=e, w=wnd, n_small=ns, n_large=nl, t=0, mode=mode)
    heals = range(0, wnd + 1)
    recs = (False, True) if nl in (1, 2) else (False,)
    if mode in (SEARCH_SMALL, w.SEARCH_LARGE):
        finds, catches, combats = (False, True), (False,), (None,)
    elif mode == CHASE_4:
        finds, catches, combats = (False,), (False, True), (None,)
    elif mode == COMBAT:
        finds, catches, combats = (False,), (False,), ("kill", "acute", "wound", "miss")
    else:
        finds, catches, combats = (False,), (False,), (None,)
    cont, back = [], []
    for h, rc, fd, ct, cb in itertools.product(heals, recs, finds, catches, combats):
        STAT["raw_internal"] += 1
        nxt = transition(base, None, Draws(heal=h, recover=rc, find=fd,
                                           catch=ct, combat=cb))
        if not nxt.alive:
            continue
        rec = (nxt.e, cap_w(nxt.w), nxt.n_small, nxt.n_large)
        if nxt.mode == DECISION:
            back.append(rec)
        else:
            cont.append((nxt.mode,) + rec)
    res = (tuple(sorted(set(cont))), tuple(sorted(set(back))))
    _step[x] = res
    return res


_cont = {}


def continuation(x, room):
    """内部状態 x から、room 歩以内に decision へ戻る全ての (τ, 状態) の support。
    room は min(TAU_MAX, horizonまでの残り)。
    World_1-G では行動の途中で体力が増えないので、内部は体力について DAG になる。
    """
    key = (x, room)
    hit = _cont.get(key)
    if hit is not None:
        STAT["cont_hit"] += 1
        return hit
    STAT["cont_miss"] += 1
    if room <= 0:
        _cont[key] = ()
        return ()
    cont, back = step_support(x)
    out = set()
    for rec in back:
        out.add((1, rec))          # 1歩で decision に戻る
        STAT["raw_landing"] += 1
    if room > 1:
        for y in cont:
            for (dt, rec) in continuation(y, room - 1):
                out.add((dt + 1, rec))
    res = tuple(sorted(out))
    _cont[key] = res
    return res


_macro = {}


def macro_support(e, wnd, ns, nl, action, room):
    """decision 状態 + 行動 → {τ: 着地状態の集合}。continuation を使う"""
    key = (e, wnd, ns, nl, action, room)
    hit = _macro.get(key)
    if hit is not None:
        return hit
    start = (w.start_mode(action), e, cap_w(wnd), ns, nl)
    out = {}
    for (tau, rec) in continuation(start, room):
        out.setdefault(tau, set()).add(rec)
    res = {k: frozenset(v) for k, v in out.items()}
    _macro[key] = res
    return res


def run(max_t, verbose=True):
    BUF = TAU_MAX + 1
    buf = [set() for _ in range(BUF)]
    buf[0].add((E_MAX, 0, 12, 3))
    rows = []
    t0 = time.perf_counter()
    for t in range(max_t + 1):
        cur = buf[t % BUF]
        b0 = dict(STAT)
        ts = time.perf_counter()
        uniq_landing = set()
        for (e, wnd, ns, nl) in cur:
            s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=DECISION)
            room = min(TAU_MAX, HORIZON - t)
            for a in w.legal_actions(s):
                for tau, landed in macro_support(e, wnd, ns, nl, a, room).items():
                    tgt = t + tau
                    if tgt >= HORIZON:
                        continue
                    buf[tgt % BUF] |= landed
                    for rec in landed:
                        uniq_landing.add((tau,) + rec)
        dt = time.perf_counter() - ts
        rows.append(dict(t=t, n=len(cur), sec=dt, digest=digest_of(cur),
                         raw_internal=STAT["raw_internal"] - b0["raw_internal"],
                         raw_landing=STAT["raw_landing"] - b0["raw_landing"],
                         uniq_landing=len(uniq_landing),
                         rss=_proc.memory_info().rss / 1024**3))
        buf[t % BUF] = set()
    return rows, time.perf_counter() - t0


def main(max_t=9):
    print("到達可能集合の support 列挙（continuation memoization 版）")
    print("世界 386bf76 ／ solver仕様 a66e9e4 ／ K =", K_WOUND, "／ τ_max =", TAU_MAX)
    print(f"範囲 t=0..{max_t}  ※cProfile なし（性能比較用）")
    print()
    rows, total = run(max_t)
    print("  t | |S_t| |   層の秒 | digest(SHA-256先頭16) | 内部展開 | 着地(重複込) | 一意な着地 | RSS")
    for r in rows:
        print(f"{r['t']:3d} | {r['n']:5,} | {r['sec']:8.3f} | {r['digest']} |"
              f" {r['raw_internal']:8,} | {r['raw_landing']:12,} |"
              f" {r['uniq_landing']:10,} | {r['rss']:.2f}GB")
    print()
    print("累計:", {k: f"{v:,}" for k, v in STAT.items()})
    print("continuation キャッシュ:", f"{len(_cont):,}",
          " step キャッシュ:", f"{len(_step):,}",
          " macro キャッシュ:", f"{len(_macro):,}")
    print("全体の秒: %.2f  RSS %.2fGB"
          % (total, _proc.memory_info().rss / 1024**3))


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
