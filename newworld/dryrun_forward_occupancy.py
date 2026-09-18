# -*- coding: utf-8 -*-
"""dummy policy を作り、前向きに占有率を流す dry-run（ChatGPT 返答108の5）。
★科学的な目的関数は一切使わない。データ経路が動くかだけを見る★
dummy policy: (state_index + t) % 合法行動の数

確認するもの
  ・約1GB の方策ファイルを memmap で作れるか、書き込み速度
  ・時刻順に読めるか
  ・current / next の2層だけで前向きに流せるか
  ・強制 MODE は方策を参照せず強制遷移、decision だけ参照
  ・★各歩で P(生存) + P(不利な死) + P(天寿) = 1 が保たれるか★
  ・t=999 の horizon 処理
  ・ピーク RSS、1000歩の所要時間、一時ファイルの削除
"""
import os, sys, json, time
import numpy as np
import world1g as w
from world1g import DECISION, HORIZON
import dense_index as dx
from gateB_kernel_vs_canonical import pair_index_map, ACT_ID
import psutil

_proc = psutil.Process(os.getpid())
D = "kernel_full"
POLICY_PATH = "_dummy_policy.dat"


def rss():
    return _proc.memory_info().rss / 1024**3


def main(n_steps=None):
    n_steps = n_steps or HORIZON
    print("dummy policy と前向き占有率の dry-run（科学的な結果は出さない）")
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob = A["succ"], A["prob"]
    t_prob, t_cause = A["t_prob"], A["t_cause"]
    state_idx, act_id = A["state_idx"].astype(np.int64), A["act_id"]
    n_pairs = man["n_pairs"]
    print(f"  kernel 読み込み  RSS {rss():.2f}GB")

    # 組の索引：state_index → その状態の組の番号（行動ごと）
    t0 = time.perf_counter()
    pair_of = np.full((dx.TOTAL, 4), -1, dtype=np.int64)
    pair_of[state_idx, act_id] = np.arange(n_pairs, dtype=np.int64)
    print(f"  組の索引を作った {time.perf_counter()-t0:.1f}秒"
          f"  {pair_of.nbytes/1024**3:.3f}GB  RSS {rss():.2f}GB")

    # 各 decision 状態の合法行動の数（1=草, 2=小物, 3=大物）
    dec_valid = pair_of[:dx.STRIDE_M, 1] >= 0
    n_acts = (pair_of[:dx.STRIDE_M, 1] >= 0).astype(np.int8) \
           + (pair_of[:dx.STRIDE_M, 2] >= 0).astype(np.int8) \
           + (pair_of[:dx.STRIDE_M, 3] >= 0).astype(np.int8)
    print(f"  decision の合法行動の数の分布: "
          f"{ {int(k): int(v) for k, v in zip(*np.unique(n_acts, return_counts=True))} }")

    # --- 方策ファイル（memmap）を作る
    print()
    print(f"【方策ファイル】{dx.STRIDE_M:,} × {n_steps} × 1バイト"
          f" = {dx.STRIDE_M*n_steps/1024**3:.3f} GB")
    t0 = time.perf_counter()
    pol = np.memmap(POLICY_PATH, dtype=np.uint8, mode="w+",
                    shape=(n_steps, dx.STRIDE_M))
    base = np.arange(dx.STRIDE_M, dtype=np.int64)
    for t in range(n_steps):
        # ★dummy: (state_index + t) % 合法行動の数 → 1..3 に写す★
        sel = np.where(n_acts > 0, (base + t) % np.maximum(n_acts, 1) + 1, 0)
        pol[t] = sel.astype(np.uint8)
    pol.flush()
    el_w = time.perf_counter() - t0
    size_gb = os.path.getsize(POLICY_PATH) / 1024**3
    print(f"  書き込み {el_w:.1f}秒  {size_gb:.3f} GB"
          f"  {size_gb/el_w:.2f} GB/s  RSS {rss():.2f}GB")
    del pol

    # --- 前向きに流す
    print()
    print("【前向き占有率】current / next の2層だけ")
    pol = np.memmap(POLICY_PATH, dtype=np.uint8, mode="r",
                    shape=(n_steps, dx.STRIDE_M))
    occ = np.zeros(dx.TOTAL, dtype=np.float64)
    occ[dx.to_index(DECISION, w.E_MAX, 0, 12, 3)] = 1.0
    dead_starve = dead_acute = dead_tenju = 0.0
    times = []
    peak = rss()
    t_start = time.perf_counter()
    for t in range(n_steps):
        t0 = time.perf_counter()
        nxt = np.zeros(dx.TOTAL, dtype=np.float64)
        # 組ごとの入ってくる質量を作る
        mass = np.zeros(n_pairs, dtype=np.float64)
        # 強制 MODE（act_id=0）：方策を参照しない
        forced = state_idx >= dx.STRIDE_M
        mass[forced] = occ[state_idx[forced]]
        # decision（act_id=1..3）：★方策が選んだ行動だけ流す★
        sel = pol[t]                                  # 0 なら無効な状態
        for a in (1, 2, 3):
            idx_dec = np.nonzero(sel == a)[0]
            if len(idx_dec) == 0:
                continue
            pids = pair_of[idx_dec, a]
            good = pids >= 0
            mass[pids[good]] = occ[idx_dec[good]]
        # 枝に沿って流す
        owner_mass = np.repeat(mass, np.diff(offs))
        np.add.at(nxt, succ, owner_mass * prob)
        tm = np.repeat(mass, np.diff(t_offs)) * t_prob
        dead_starve += float(tm[t_cause == 1].sum())
        dead_acute += float(tm[t_cause == 2].sum())
        # ★t=999 の非終端は天寿★
        if t + 1 >= HORIZON:
            dead_tenju += float(nxt.sum())
            nxt[:] = 0.0
        occ = nxt
        el = time.perf_counter() - t0
        times.append(el)
        peak = max(peak, rss())
        total = occ.sum() + dead_starve + dead_acute + dead_tenju
        if t < 3 or t % 200 == 0 or t == n_steps - 1:
            print(f"  t={t:4d} 生存 {occ.sum():.6f} 飢え {dead_starve:.6f}"
                  f" 急所 {dead_acute:.6f} 天寿 {dead_tenju:.6f}"
                  f" ★合計 {total:.12f}★  {el:.2f}秒 RSS {rss():.2f}GB", flush=True)
        if abs(total - 1.0) > 1e-9:
            print(f"  ★質量が保存されていない: {total}★")
            break
    el_all = time.perf_counter() - t_start
    print()
    print(f"【結果】{n_steps} 歩 {el_all:.1f}秒（1歩 中央値 {sorted(times)[len(times)//2]:.3f}秒）")
    print(f"  ピーク RSS {peak:.2f}GB")
    print(f"  最終の質量の合計 {occ.sum() + dead_starve + dead_acute + dead_tenju:.12f}")
    del pol
    import gc
    gc.collect()          # ★memmap を確実に閉じてから削除する★
    for _ in range(10):
        try:
            os.remove(POLICY_PATH)
            break
        except PermissionError:
            time.sleep(0.5)
    print(f"  一時ファイルを削除した（存在: {os.path.exists(POLICY_PATH)}）")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
