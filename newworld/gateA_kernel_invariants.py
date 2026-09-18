# -*- coding: utf-8 -*-
"""Gate A：実寸 kernel の全 9,768,000 組について機械的な不変条件を確認する。
（ChatGPT 返答106の2）
  ・確率の合計 ≈ 1
  ・確率 > 0
  ・非終端の successor index が範囲内
  ・successor が structural-valid
  ・終端の cause が定義済みの範囲
  ・offsets が単調
  ・最終 offset = 正確な枝の数
"""
import os, sys, time, json
import numpy as np
import world1g as w
import dense_index as dx
import psutil

_proc = psutil.Process(os.getpid())
D = "kernel_full"


def rss():
    return _proc.memory_info().rss / 1024**3


def structural_valid_mask():
    """structural-valid な primitive index のマスク（E>=1、大物MODEは N_L>=1）"""
    mask = np.zeros(dx.TOTAL, dtype=bool)
    for mi, mode in enumerate(dx.MODES):
        big = mode in (w.SEARCH_LARGE, w.COMBAT)
        base = mi * dx.STRIDE_M
        for e in range(1, dx.N_E):
            off = base + e * dx.STRIDE_E
            for wnd in range(dx.N_W):
                off2 = off + wnd * dx.STRIDE_W
                for ns in range(dx.N_S):
                    off3 = off2 + ns * dx.STRIDE_S
                    lo = 1 if big else 0
                    mask[off3 + lo: off3 + dx.N_L] = True
    return mask


ok, ng = 0, []


def check(name, cond, detail=""):
    global ok
    if cond:
        ok += 1
        print(f"  OK   {name}  {detail}")
    else:
        ng.append(name)
        print(f"  ★NG★ {name}  {detail}")


def main():
    print("Gate A：実寸 kernel の全組の不変条件")
    with open(os.path.join(D, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    A = {nm: np.load(os.path.join(D, nm + ".npy")) for nm in man["arrays"]}
    n_pairs = man["n_pairs"]
    print(f"  組 {n_pairs:,} / 非終端 {man['n_nonterminal']:,}"
          f" / 終端 {man['n_terminal']:,}  RSS {rss():.2f}GB")
    print()

    offs, t_offs = A["offs"], A["t_offs"]
    succ, prob = A["succ"], A["prob"]
    t_prob, t_cause, t_rawe = A["t_prob"], A["t_cause"], A["t_rawe"]
    state_idx, act_id = A["state_idx"], A["act_id"]

    # offsets
    check("offs が単調非減少", bool(np.all(np.diff(offs) >= 0)))
    check("t_offs が単調非減少", bool(np.all(np.diff(t_offs) >= 0)))
    check("offs の末尾 = 非終端の枝の数", int(offs[-1]) == man["n_nonterminal"],
          f"{int(offs[-1]):,}")
    check("t_offs の末尾 = 終端の枝の数", int(t_offs[-1]) == man["n_terminal"],
          f"{int(t_offs[-1]):,}")
    check("offs の長さ = 組の数+1", len(offs) == n_pairs + 1)

    # 確率
    check("非終端の確率が全て > 0", bool(np.all(prob > 0.0)),
          f"最小 {prob.min():.3e}")
    check("終端の確率が全て > 0", bool(np.all(t_prob > 0.0)),
          f"最小 {t_prob.min():.3e}")
    check("非終端の確率が全て <= 1", bool(np.all(prob <= 1.0)),
          f"最大 {prob.max():.6f}")

    # 確率の合計（全組）
    t0 = time.perf_counter()
    owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(offs))
    t_owner = np.repeat(np.arange(n_pairs, dtype=np.int32), np.diff(t_offs))
    tot = np.bincount(owner, weights=prob, minlength=n_pairs)
    tot += np.bincount(t_owner, weights=t_prob, minlength=n_pairs)
    err = np.abs(tot - 1.0)
    check("★全組で確率の合計が 1（誤差 1e-12 未満）★", bool(err.max() < 1e-12),
          f"最大のずれ {err.max():.3e}（{time.perf_counter()-t0:.1f}秒）")
    print(f"       ずれが 1e-15 を超えた組の数: {int((err > 1e-15).sum()):,}")

    # successor の範囲と valid
    check("successor index が範囲内", bool(succ.max() < dx.TOTAL),
          f"最大 {int(succ.max()):,} < {dx.TOTAL:,}")
    t0 = time.perf_counter()
    vmask = structural_valid_mask()
    print(f"       structural-valid マスクを作った（{vmask.sum():,} 件、"
          f"{time.perf_counter()-t0:.1f}秒）")
    check("structural-valid の数が 7,920,000", int(vmask.sum()) == 7_920_000,
          f"{int(vmask.sum()):,}")
    t0 = time.perf_counter()
    bad = int((~vmask[succ]).sum())
    check("★全ての successor が structural-valid★", bad == 0,
          f"valid でない successor {bad:,} 件（{time.perf_counter()-t0:.1f}秒）")

    # 終端の cause
    check("終端の cause が 1〜3 の範囲", bool(np.all((t_cause >= 1) & (t_cause <= 3))),
          f"一意な値 {sorted(set(t_cause.tolist()[:100000]))}")
    for cid, nm in ((1, "飢え死"), (2, "急所死"), (3, "天寿")):
        n = int((t_cause == cid).sum())
        print(f"       {nm}: {n:,} 件")

    # 終端の生の体力
    check("★飢え死の終端の体力が 0 以下★",
          bool(np.all(t_rawe[t_cause == 1] <= 0)),
          f"最大 {int(t_rawe[t_cause==1].max())}, 最小 {int(t_rawe[t_cause==1].min())}")
    acute = t_rawe[t_cause == 2]
    check("急所死の終端の体力に正の値がある（0 clip されていない）",
          bool(acute.max() > 0), f"最大 {int(acute.max())}, 最小 {int(acute.min())}")

    # 組の状態
    check("state_idx が structural-valid", bool(np.all(vmask[state_idx.astype(np.int64)])))
    check("act_id が 0〜3", bool(np.all(act_id <= 3)),
          f"一意な値 {sorted(set(act_id.tolist()[:100000]))}")

    print()
    print(f"Gate A: 合格 {ok} 件 / NG {len(ng)} 件  ピークRSS {rss():.2f}GB")
    for n in ng:
        print("  NG:", n)


if __name__ == "__main__":
    main()
