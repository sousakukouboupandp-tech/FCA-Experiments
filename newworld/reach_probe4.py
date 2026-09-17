# -*- coding: utf-8 -*-
"""到達可能集合の support 列挙（primitive-time global propagation 版）
ChatGPT 返答60の第三案。macro にまとめず、正本の1歩 MDP を時刻方向に一度だけ流す。
同じ時刻の同じ primitive 状態は集合で1つに潰れるので、重複が構造的に消える。
exactness は変えない（確率の打ち切り・標本化・粗視化・集約なし）。

計器（返答58の8で整理）
  primitive_frontier : F_t の大きさ（MODE 込み）
  decision_states    : F_t のうち decision のもの = S_t
  raw_step_outcomes  : その時刻に生成した枝の数
  unique_next        : F_{t+1} の大きさ
  dedup_ratio        : raw_step_outcomes ÷ unique_next
digest: S_t を正順に並べて固定書式で SHA-256（Python の hash は使わない）
"""
import os, sys, time, itertools, hashlib
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX, HORIZON)
import psutil

K_WOUND = 10
LIMIT_STATES = 20_000_000
LIMIT_SECONDS = 30 * 60
LIMIT_RSS_GB = 8.0          # 実験コード側の停止条件（watchdog とは別に持つ）
_proc = psutil.Process(os.getpid())


def cap_w(x):
    return min(x, K_WOUND)


def digest_of(states):
    h = hashlib.sha256()
    for rec in sorted(states):
        h.update(("%d,%d,%d,%d;" % rec).encode("ascii"))
    return h.hexdigest()[:16]


_step = {}


def step_outcomes(x, action=None):
    """primitive 状態 x=(mode,e,w,ns,nl) の1歩。生存した次の状態のタプルを返す。
    decision のときは action を指定する（その行動の1歩目を同じ歩で実行）。"""
    key = (x, action)
    hit = _step.get(key)
    if hit is not None:
        return hit
    mode, e, wnd, ns, nl = x
    base = State(e=e, w=wnd, n_small=ns, n_large=nl, t=0, mode=mode)
    run_mode = w.start_mode(action) if mode == DECISION else mode
    heals = range(0, wnd + 1)
    recs = (False, True) if nl in (1, 2) else (False,)
    if run_mode in (SEARCH_SMALL, w.SEARCH_LARGE):
        finds, catches, combats = (False, True), (False,), (None,)
    elif run_mode == CHASE_4:
        finds, catches, combats = (False,), (False, True), (None,)
    elif run_mode == COMBAT:
        finds, catches, combats = (False,), (False,), ("kill", "acute", "wound", "miss")
    else:
        finds, catches, combats = (False,), (False,), (None,)
    out = set()
    n_raw = 0
    for h, rc, fd, ct, cb in itertools.product(heals, recs, finds, catches, combats):
        n_raw += 1
        nxt = transition(base, action, Draws(heal=h, recover=rc, find=fd,
                                             catch=ct, combat=cb))
        if not nxt.alive:
            continue          # 終端は次の frontier に入れない
        out.add((nxt.mode, nxt.e, cap_w(nxt.w), nxt.n_small, nxt.n_large))
    res = (tuple(sorted(out)), n_raw)
    _step[key] = res
    return res


def run(max_t, quiet=False):
    """時刻方向に1歩ずつ前進する。F_t（MODE 込み）→ F_{t+1}"""
    frontier = {(DECISION, E_MAX, 0, 12, 3)}
    rows = []
    total_decision = 0
    t0 = time.perf_counter()
    stop = None
    for t in range(max_t + 1):
        dec = {x[1:] for x in frontier if x[0] == DECISION}
        total_decision += len(dec)
        raw = 0
        nxt = set()
        for x in frontier:
            if x[0] == DECISION:
                s = State(e=x[1], w=x[2], n_small=x[3], n_large=x[4],
                          t=t, mode=DECISION)
                for a in w.legal_actions(s):
                    outs, n_raw = step_outcomes(x, a)
                    raw += n_raw
                    nxt.update(outs)
            else:
                outs, n_raw = step_outcomes(x, None)
                raw += n_raw
                nxt.update(outs)
        el = time.perf_counter() - t0
        rss = _proc.memory_info().rss / 1024**3
        rows.append(dict(t=t, frontier=len(frontier), dec=len(dec),
                         digest=digest_of(dec), raw=raw, uniq=len(nxt),
                         sec=el, rss=rss))
        if not quiet and (t < 20 or t % 25 == 0 or t == max_t):
            r = rows[-1]
            print(f"t={t:4d} |F_t|={r['frontier']:>9,} |S_t|={r['dec']:>8,}"
                  f" {r['digest']} 枝{r['raw']:>10,} 次{r['uniq']:>9,}"
                  f" 重複比{(r['raw']/max(1,r['uniq'])):6.1f}"
                  f" RSS {rss:.2f}GB {el:.1f}秒", flush=True)
        if t + 1 >= HORIZON:
            stop = "horizon に到達"
            break
        if total_decision > LIMIT_STATES:
            stop = f"decision 状態の累計が {LIMIT_STATES:,} を超えた（t={t}）"
            break
        if el > LIMIT_SECONDS:
            stop = f"実行時間が {LIMIT_SECONDS} 秒を超えた（t={t}）"
            break
        if rss > LIMIT_RSS_GB:
            stop = f"自プロセスの RSS が {LIMIT_RSS_GB}GB を超えた（t={t}）"
            break
        frontier = nxt        # 前の層は捨てる（2層だけ持つ）
    return rows, stop, time.perf_counter() - t0, total_decision


def main(max_t):
    print("到達可能集合の support 列挙（primitive-time global propagation 版）")
    print("世界 386bf76 ／ solver仕様 a66e9e4 ／ K =", K_WOUND)
    print("打ち切り: decision累計", f"{LIMIT_STATES:,}", "・",
          LIMIT_SECONDS // 60, "分 ・ 自プロセスRSS", LIMIT_RSS_GB, "GB")
    print(f"範囲 t=0..{max_t}")
    print()
    rows, stop, total, total_dec = run(max_t)
    print()
    print("停止理由:", stop if stop else f"t={max_t} まで完了")
    print("decision 状態の累計:", f"{total_dec:,}")
    print("step キャッシュ:", f"{len(_step):,}")
    print("全体の秒: %.2f  RSS %.2fGB"
          % (total, _proc.memory_info().rss / 1024**3))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)
