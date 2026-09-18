# -*- coding: utf-8 -*-
"""D0 の解析(1)：方策地図と行動領域の数（凍結した解析の順番の第1項目）。
★結果を見てから測り方を変えない★

出すもの（表3 で凍結した項目のうち、D0 単体で出るもの）
  ・行動の占有率（到達可能な decision 状態のうち、各行動が選ばれる割合）
  ・行動エントロピー（★記述統計としてのみ。高い方が良いとはしない★）
  ・時刻ごとの行動の分布（領域がいくつに分かれるか）
  ・体力・小物・大物・傷 ごとの行動の切り替わり
  ・前向きに流して死因の分布と期待寿命を確認（V0 と一致するか）
"""
import os, sys, json, time, gc
import numpy as np
import world1g as w
from world1g import DECISION, HORIZON
import dense_index as dx
from gateB_kernel_vs_canonical import pair_index_map, ACT_ID
import psutil

_proc = psutil.Process(os.getpid())
D, S = "kernel_full", "solve_D0"
ACT_NAME = {0: "（無効）", 1: "草", 2: "小物", 3: "大物"}


def rss():
    return _proc.memory_info().rss / 1024**3


def main():
    print("D0 の解析(1)：方策地図と行動領域の数")
    with open(os.path.join(S, "summary.json"), encoding="utf-8") as f:
        sm = json.load(f)
    print(f"  V0（期待寿命）= {sm['V_init']:.6f} 歩 ／ 後ろ向き {sm['backward_sec']}秒")
    print()
    pol = np.memmap(os.path.join(S, "policy.dat"), dtype=np.uint8, mode="r",
                    shape=(HORIZON, dx.STRIDE_M))
    opt = np.memmap(os.path.join(S, "optimal_set.dat"), dtype=np.uint8, mode="r",
                    shape=(HORIZON, dx.STRIDE_M))

    # --- 全時刻・全 decision 状態の行動の分布（structural-valid のみ）
    print("【全時刻の行動の分布】（structural-valid な decision 状態 = E>=1）")
    cnt = np.zeros(4, dtype=np.int64)
    tie_cnt = np.zeros(8, dtype=np.int64)     # optimal_set のビットの組み合わせ
    for t in range(HORIZON):
        p = np.asarray(pol[t])
        cnt += np.bincount(p, minlength=4)
        tie_cnt += np.bincount(np.asarray(opt[t]), minlength=8)
    tot_valid = cnt[1:].sum()
    print(f"  有効な (状態,時刻): {tot_valid:,}")
    for a in (1, 2, 3):
        print(f"    {ACT_NAME[a]:4s} {cnt[a]:>12,}  {cnt[a]/tot_valid*100:6.2f}%")
    ent = -sum((cnt[a]/tot_valid) * np.log(cnt[a]/tot_valid)
               for a in (1, 2, 3) if cnt[a] > 0)
    print(f"  行動エントロピー {ent:.4f}（★記述統計のみ。高い方が良いとはしない★）")
    print()
    print("【optimal_set の分布】（同点がどれだけあるか）")
    names = {0: "なし", 1: "草", 2: "小物", 3: "草+小物", 4: "大物",
             5: "草+大物", 6: "小物+大物", 7: "草+小物+大物"}
    for b in range(8):
        if tie_cnt[b]:
            print(f"    {names[b]:14s} {tie_cnt[b]:>12,}  {tie_cnt[b]/tot_valid*100:6.2f}%")
    print()

    # --- 時刻ごとの行動の分布（領域の数）
    print("【時刻ごとの行動の分布】")
    print("    t |        草 |      小物 |      大物 | 最多")
    for t in (0, 1, 10, 50, 100, 200, 400, 600, 800, 900, 950, 990, 999):
        p = np.asarray(pol[t])
        c = np.bincount(p, minlength=4)
        tv = c[1:].sum()
        best = int(np.argmax(c[1:])) + 1
        print(f" {t:4d} | {c[1]/tv*100:8.2f}% | {c[2]/tv*100:8.2f}%"
              f" | {c[3]/tv*100:8.2f}% | {ACT_NAME[best]}")
    print()

    # --- 体力ごとの行動（t=500 で切る）
    print("【t=500 の体力ごとの行動】（W=0, N_S=12, N_L=3 に固定）")
    print("   体力 | 行動")
    prev = None
    for e in range(1, 2001):
        i = dx.decision_index_of(e, 0, 12, 3)
        a = int(pol[500][i])
        if a != prev:
            print(f"  {e:5d} | {ACT_NAME[a]}  ← ★切り替わり★")
            prev = a
    print()

    # --- 小物の残り具合ごとの行動
    print("【t=500, E=1000, W=0, N_L=3 の小物ごとの行動】")
    for ns in range(1, 13):
        i = dx.decision_index_of(1000, 0, ns, 3)
        print(f"  N_S={ns:2d} → {ACT_NAME[int(pol[500][i])]}")
    print()
    print("【t=500, E=1000, W=0, N_S=12 の大物ごとの行動】")
    for nl in range(0, 4):
        i = dx.decision_index_of(1000, 0, 12, nl)
        print(f"  N_L={nl} → {ACT_NAME[int(pol[500][i])]}")
    print()
    print("【t=500, E=1000, N_S=12, N_L=3 の傷ごとの行動】")
    for wnd in range(0, 11):
        i = dx.decision_index_of(1000, wnd, 12, 3)
        print(f"  W={wnd:2d} → {ACT_NAME[int(pol[500][i])]}")
    del pol, opt
    gc.collect()
    print()
    print(f"ピークRSS {rss():.2f}GB")


if __name__ == "__main__":
    main()
