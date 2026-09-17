# -*- coding: utf-8 -*-
"""到達可能な decision 状態の support を層ごとに列挙し、規模を実測する。
仕様：SOLVER_SPEC_World1G（f6c1ad0 + 訂正 503d2f4）
  ・support は確率の大きさを見ない（0より大きければ入れる）
  ・層ごとに処理し、前の層は捨てる
  ・打ち切り：状態総数2000万／30分／自プロセスRSS 8GB
world1g.py は変更しない。ここは solver 側の道具。
"""
import os, sys, time, itertools
import world1g as w
from world1g import (State, Draws, transition, DECISION, GRASS, SEARCH_SMALL,
                     CHASE_1, CHASE_2, CHASE_3, CHASE_4, SEARCH_LARGE, COMBAT,
                     ACT_GRASS, ACT_SMALL, ACT_LARGE, E_MAX, HORIZON)

K_WOUND = 10                 # 傷のあふれ（solver の近似。世界の物理ではない）
LIMIT_STATES = 20_000_000
LIMIT_SECONDS = 30 * 60
LIMIT_RSS_GB = 8.0

try:
    import psutil
    _proc = psutil.Process(os.getpid())
    def rss_gb():
        return _proc.memory_info().rss / (1024 ** 3)
except Exception:
    def rss_gb():
        return 0.0      # psutil が無い場合は 0 を返す（その旨を報告する）


def cap_w(x):
    return min(x, K_WOUND)


def next_decision_states(s):
    """decision 状態 s から、各行動について「次の decision 境界の状態」の support を返す。
    確率は見ない（0より大きい枝すべて）。終端に入る枝は含めない（別に数える）。
    戻り値: {行動: set(状態タプル)}, 終端に入りうるか
    """
    out = {}
    for a in w.legal_actions(s):
        acc = set()
        # 内部の強制 MODE を、確率0でない全分岐で展開する。
        # 各要素は (mode, e, wnd, ns, nl, t)。t は horizon で打ち切る。
        frontier = {(w.start_mode(a), s.e, s.w, s.n_small, s.n_large, s.t)}
        seen = set()
        while frontier:
            cur = frontier.pop()
            if cur in seen:
                continue
            seen.add(cur)
            mode, e, wnd, ns, nl, t = cur
            base = State(e=e, w=wnd, n_small=ns, n_large=nl, t=t, mode=mode)
            # その歩に起こりうる抽選の組み合わせ
            heals = range(0, wnd + 1)                      # 0..W（二項なのでどれも確率>0）
            recs = (False, True) if nl in (1, 2) else (False,)
            if mode == SEARCH_SMALL or mode == SEARCH_LARGE:
                finds = (False, True)
                catches = (False,)
                combats = (None,)
            elif mode == CHASE_4:
                finds = (False,)
                catches = (False, True)
                combats = (None,)
            elif mode == COMBAT:
                finds = (False,)
                catches = (False,)
                combats = ("kill", "acute", "wound", "miss")
            else:
                finds = (False,)
                catches = (False,)
                combats = (None,)
            for h, rc, fd, ct, cb in itertools.product(heals, recs, finds, catches, combats):
                # base.mode は decision ではないので行動は渡さない（v02：decision は境界）
                nxt = transition(base, None, Draws(heal=h, recover=rc, find=fd,
                                                   catch=ct, combat=cb))
                if not nxt.alive:
                    continue                                 # 終端は次の層に入らない
                key = (nxt.mode, nxt.e, cap_w(nxt.w), nxt.n_small, nxt.n_large, nxt.t)
                if nxt.t >= HORIZON:
                    continue                                 # 天寿は終端
                if nxt.mode == DECISION:
                    acc.add((nxt.e, cap_w(nxt.w), nxt.n_small, nxt.n_large, nxt.t))
                else:
                    frontier.add(key)
        out[a] = acc
    return out


def main():
    t_start = time.perf_counter()
    print("到達可能な decision 状態の support 列挙（層ごと）")
    print("世界 386bf76 ／ 実装 1cde347 ／ solver仕様 f6c1ad0 + 503d2f4")
    print("K（傷のあふれ）=", K_WOUND, " 打ち切り: 状態総数", f"{LIMIT_STATES:,}",
          "／", LIMIT_SECONDS // 60, "分 ／ 自プロセスRSS", LIMIT_RSS_GB, "GB")
    if rss_gb() == 0.0:
        print("※ psutil が無いため RSS を測れない。RSS の停止条件は働かない")
    print("理論上限（直積）: 1時刻あたり", f"{(E_MAX+1)*(K_WOUND+1)*12*4:,}")
    print()

    init = (E_MAX, 0, 12, 3, 0)
    layer = {init}
    total = 1
    stop = None
    rows = []

    for t in range(HORIZON):
        if not layer:
            stop = "到達可能な状態が尽きた（全滅）"
            break
        nxt = set()
        for st in layer:
            e, wnd, ns, nl, tt = st
            s = State(e=e, w=wnd, n_small=ns, n_large=nl, t=tt, mode=DECISION)
            for a, acc in next_decision_states(s).items():
                nxt |= acc
        total += len(nxt)
        peak = rss_gb()
        el = time.perf_counter() - t_start

        if t < 12 or t % 25 == 0:
            es = [x[0] for x in nxt] or [0]
            ws = [x[1] for x in nxt] or [0]
            nss = [x[2] for x in nxt] or [0]
            nls = [x[3] for x in nxt] or [0]
            ts = sorted({x[4] for x in nxt})
            print(f"層 t={t:4d} → 次層 {len(nxt):>9,} 状態  累計 {total:>12,}"
                  f"  E[{min(es)}-{max(es)}] W[{min(ws)}-{max(ws)}]"
                  f" NS[{min(nss)}-{max(nss)}] NL[{min(nls)}-{max(nls)}]"
                  f" 時刻幅 {len(ts)}  RSS {peak:.2f}GB  {el:.1f}秒", flush=True)
            rows.append((t, len(nxt), total, peak, el))

        if total > LIMIT_STATES:
            stop = f"状態総数が {LIMIT_STATES:,} を超えた（累計 {total:,}）"
            break
        if el > LIMIT_SECONDS:
            stop = f"実行時間が {LIMIT_SECONDS} 秒を超えた"
            break
        if peak > LIMIT_RSS_GB:
            stop = f"自プロセスの RSS が {LIMIT_RSS_GB}GB を超えた（{peak:.2f}GB）"
            break

        layer = nxt      # 前の層は捨てる

    print()
    print("停止理由:", stop if stop else "horizon まで到達")
    print("累計の生成数:", f"{total:,}")
    print("最終RSS: %.2f GB" % rss_gb())
    print("実行時間: %.1f 秒" % (time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
