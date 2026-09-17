# -*- coding: utf-8 -*-
"""到達可能な decision 状態の support を「1歩時間 t ごとの層」で列挙する。
仕様：SOLVER_SPEC_World1G（f6c1ad0 → 503d2f4 → a66e9e4）
  ・層は1歩時間 t ごと。decision 回数を層番号にしない
  ・τ_max = 100（物理から導出）。rolling buffer 101本
  ・support は確率の大きさを見ない（0より大きければ入れる）
  ・horizon 越えと終端は未来の bucket に入れない
  ・打ち切り：状態総数2000万／30分／自プロセスRSS 8GB
world1g.py は変更しない。
"""
import os, sys, time, itertools
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL,
                     CHASE_4, COMBAT, E_MAX, HORIZON)
import psutil

K_WOUND = 10
LIMIT_STATES = 20_000_000
LIMIT_SECONDS = 30 * 60
LIMIT_RSS_GB = 8.0
TAU_MAX = 100

_proc = psutil.Process(os.getpid())
def rss_gb():
    return _proc.memory_info().rss / (1024 ** 3)


def cap_w(x):
    return min(x, K_WOUND)


# 内部の1歩の展開（時刻を含めない相対表現）。同じ組は使い回す
_step_cache = {}


def step_support(mode, e, wnd, ns, nl):
    """内部 MODE の1歩を、確率>0 の全分岐で展開する。
    戻り値: (生存して続く枝の集合, decision に戻る枝の集合)
      続く枝    : (mode, e, w, ns, nl)   ← 相対。時刻は呼び側が足す
      戻る枝    : (e, w, ns, nl)
    死んだ枝は捨てる（終端は未来の bucket に入れない）
    """
    key = (mode, e, wnd, ns, nl)
    hit = _step_cache.get(key)
    if hit is not None:
        return hit
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

    cont, back = set(), set()
    for h, rc, fd, ct, cb in itertools.product(heals, recs, finds, catches, combats):
        nxt = transition(base, None, Draws(heal=h, recover=rc, find=fd,
                                           catch=ct, combat=cb))
        if not nxt.alive:
            continue
        rec = (nxt.e, cap_w(nxt.w), nxt.n_small, nxt.n_large)
        if nxt.mode == DECISION:
            back.add(rec)
        else:
            cont.add((nxt.mode,) + rec)
    res = (frozenset(cont), frozenset(back))
    _step_cache[key] = res
    return res


def macro_support(e, wnd, ns, nl, action, t_room):
    """decision 状態から1つの行動を取ったとき、
    「τ歩後に decision へ戻る状態」の集合を τ ごとに返す。
    t_room: horizon までの残り歩数（これを超える τ は捨てる）
    戻り値: {τ: set((e,w,ns,nl))}
    """
    out = {}
    cur = {(w.start_mode(action), e, cap_w(wnd), ns, nl)}
    for tau in range(1, min(TAU_MAX, t_room) + 1):
        if not cur:
            break
        nxt = set()
        landed = set()
        for (mode, ee, ww, nn, ll) in cur:
            cont, back = step_support(mode, ee, ww, nn, ll)
            nxt |= cont
            landed |= back
        if landed:
            out[tau] = landed
        cur = nxt
    return out


def main():
    t0 = time.perf_counter()
    print("到達可能な decision 状態の support 列挙（1歩時間 t ごとの層）")
    print("世界 386bf76 ／ 実装 1cde347 ／ solver仕様 a66e9e4")
    print("K =", K_WOUND, "／ τ_max =", TAU_MAX,
          "／ 打ち切り: 状態", f"{LIMIT_STATES:,}", "・",
          LIMIT_SECONDS // 60, "分 ・ RSS", LIMIT_RSS_GB, "GB")
    print("理論上限（直積）: 1時刻あたり", f"{(E_MAX+1)*(K_WOUND+1)*12*4:,}")
    print()

    # rolling buffer：101本。index は t % 101
    BUF = TAU_MAX + 1
    buf = [set() for _ in range(BUF)]
    buf[0].add((E_MAX, 0, 12, 3))
    total = 1
    stop = None
    peak_rss = 0.0

    for t in range(HORIZON):
        cur = buf[t % BUF]
        if cur:
            t_room = HORIZON - t
            for (e, wnd, ns, nl) in cur:
                s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=DECISION)
                for a in w.legal_actions(s):
                    for tau, landed in macro_support(e, wnd, ns, nl, a, t_room).items():
                        tgt = t + tau
                        if tgt >= HORIZON:      # horizon 越えは終端。未来の bucket に入れない
                            continue
                        b = buf[tgt % BUF]
                        before = len(b)
                        b |= landed
                        total += len(b) - before
        n_cur = len(cur)
        buf[t % BUF] = set()        # 使い終わった層は捨てる
        rss = rss_gb()
        peak_rss = max(peak_rss, rss)
        el = time.perf_counter() - t0

        if t < 15 or t % 25 == 0 or t == HORIZON - 1:
            if n_cur:
                es = [x[0] for x in cur]; ws = [x[1] for x in cur]
                nss = [x[2] for x in cur]; nls = [x[3] for x in cur]
                rng = (f"E[{min(es)}-{max(es)}] W[{min(ws)}-{max(ws)}]"
                       f" NS[{min(nss)}-{max(nss)}] NL[{min(nls)}-{max(nls)}]")
            else:
                rng = "（空）"
            print(f"t={t:4d}  |S_t|={n_cur:>9,}  累計 {total:>12,}  {rng}"
                  f"  RSS {rss:.2f}GB  {el:.1f}秒", flush=True)

        if total > LIMIT_STATES:
            stop = f"状態総数が {LIMIT_STATES:,} を超えた（累計 {total:,}、t={t}）"
            break
        if el > LIMIT_SECONDS:
            stop = f"実行時間が {LIMIT_SECONDS} 秒を超えた（t={t}）"
            break
        if rss > LIMIT_RSS_GB:
            stop = f"自プロセスの RSS が {LIMIT_RSS_GB}GB を超えた（{rss:.2f}GB、t={t}）"
            break

    print()
    print("停止理由:", stop if stop else "horizon まで到達")
    print("累計の一意な状態数（層ごとの合計）:", f"{total:,}")
    print("内部1歩のキャッシュ件数:", f"{len(_step_cache):,}")
    print("ピークRSS: %.2f GB" % peak_rss)
    print("実行時間: %.1f 秒" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
