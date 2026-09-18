# -*- coding: utf-8 -*-
"""dense 表現による到達可能集合の support 列挙（prototype）。
ChatGPT 返答74の指示：
  ★表現だけを変える。遷移は正本（world1g.py 経由の step_outcomes）をそのまま使う★
  ★SHA-256 は dense のバイト列ではなく、正準形 (e,w,ns,nl) に戻してから取る★
  ★watchdog は常時使う★
"""
import os, sys, time, hashlib
import numpy as np
import world1g as w
from world1g import DECISION, E_MAX, HORIZON
import dense_index as dx
from reach_probe4 import step_outcomes      # ★遷移は正本を使う（新規実装しない）★
import psutil

LIMIT_STATES = 20_000_000
LIMIT_SECONDS = 30 * 60
LIMIT_RSS_GB = 8.0
_proc = psutil.Process(os.getpid())


def digest_decision(mask):
    """decision 状態だけを正準形 (e,w,ns,nl) に戻し、正順に並べて SHA-256。
    ★dense のバイト列は hash しない（旧版と比較するため）★"""
    idx = np.nonzero(mask[:dx.STRIDE_M])[0]     # m=0 の範囲が decision
    h = hashlib.sha256()
    for i in idx:
        _, e, wnd, ns, nl = dx.from_index(int(i))
        h.update(("%d,%d,%d,%d;" % (e, wnd, ns, nl)).encode("ascii"))
    return h.hexdigest()[:16], len(idx)


def run(max_t, quiet=False):
    cur = np.zeros(dx.TOTAL, dtype=bool)
    cur[dx.to_index(DECISION, E_MAX, 0, 12, 3)] = True
    rows = []
    t0 = time.perf_counter()
    stop = None
    total_dec = 0
    for t in range(max_t + 1):
        dg, n_dec = digest_decision(cur)
        total_dec += n_dec
        nxt = np.zeros(dx.TOTAL, dtype=bool)
        raw = 0
        for i in np.nonzero(cur)[0]:
            st = dx.from_index(int(i))
            mode, e, wnd, ns, nl = st
            if mode == DECISION:
                s = w.State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=DECISION)
                for a in w.legal_actions(s):
                    outs, n_raw = step_outcomes((mode, e, wnd, ns, nl), a)
                    raw += n_raw
                    for o in outs:
                        nxt[dx.to_index(*o)] = True
            else:
                outs, n_raw = step_outcomes((mode, e, wnd, ns, nl), None)
                raw += n_raw
                for o in outs:
                    nxt[dx.to_index(*o)] = True
        el = time.perf_counter() - t0
        rss = _proc.memory_info().rss / 1024**3
        cpu = _proc.cpu_times()
        rows.append(dict(t=t, frontier=int(cur.sum()), dec=n_dec, digest=dg,
                         raw=raw, uniq=int(nxt.sum()), sec=el, rss=rss,
                         cpu=cpu.user + cpu.system))
        if not quiet and (t < 15 or t % 25 == 0 or t == max_t):
            r = rows[-1]
            print(f"t={t:4d} |F_t|={r['frontier']:>9,} |S_t|={r['dec']:>8,}"
                  f" {r['digest']} 枝{r['raw']:>10,} 次{r['uniq']:>9,}"
                  f" 重複比{(r['raw']/max(1,r['uniq'])):6.1f}"
                  f" RSS {rss:.2f}GB {el:.1f}秒", flush=True)
        if t + 1 >= HORIZON:
            stop = "horizon に到達"
            break
        if total_dec > LIMIT_STATES:
            stop = f"decision 累計が {LIMIT_STATES:,} を超えた（t={t}）"
            break
        if el > LIMIT_SECONDS:
            stop = f"実行時間が {LIMIT_SECONDS} 秒を超えた（t={t}）"
            break
        if rss > LIMIT_RSS_GB:
            stop = f"自プロセスの RSS が {LIMIT_RSS_GB}GB を超えた（t={t}）"
            break
        cur = nxt
    return rows, stop, time.perf_counter() - t0, total_dec


def main(max_t):
    print("dense 表現による support 列挙（prototype）")
    print("世界 386bf76 ／ 遷移は正本（world1g.py）をそのまま使用 ／ 表現のみ dense")
    print(f"TOTAL index = {dx.TOTAL:,}  bool 1層 = {dx.TOTAL/1024**2:.1f} MB")
    print(f"範囲 t=0..{max_t}")
    print()
    rows, stop, total, total_dec = run(max_t)
    print()
    print("停止理由:", stop if stop else f"t={max_t} まで完了")
    print("decision 累計:", f"{total_dec:,}")
    print("全体の秒: %.2f  RSS %.2fGB" % (total, _proc.memory_info().rss / 1024**3))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
