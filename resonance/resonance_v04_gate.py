# -*- coding: utf-8 -*-
"""
共鳴実験 v04 土俵ゲート(§7-8/§8-1/§8-4/§8-6) —— 本実行前の必須煙試験
合格基準(事前宣言・数値固定):
 [G1] Dbig+でIsolated単独5シード(§8-6)、末尾100epの最短路(mid)使用率 >= 90%
 [G2] 同、末尾100epの成功率 >= 0.99 (§8-1で数値固定)
 [G3] ロードQA(§8-4): FullCopy転送の受信件数 == A生涯アーカイブ件数(全5シード)
不合格なら凍結解除・地形再設計。勝手に進めず著者に報告する。
"""
import os, sys
BASE = os.path.dirname(os.path.abspath(__file__))
g = {"__name__": "gate_import", "__file__": os.path.join(BASE, "resonance_v04_full.py")}
exec(open(os.path.join(BASE, 'resonance_v04_full.py'), encoding='utf-8').read(), g)

np = g["np"]
GATE_SEEDS = 5
results = []
print("=== 土俵ゲート: Dbig+ Isolated単独 %dシード ===" % GATE_SEEDS)
for seed in range(GATE_SEEDS):
    # (1) A学習(ロードQA用)
    A = g["Agent"](seed)
    envA = g["ResonanceEnv"](g["PATROL_WORLD_A"])
    sA = g["new_stats"]()
    g["run_episodes"](A, envA, g["N_EP_A"], sA)
    n_arch = len(A.archive)
    # (2) ロードQA(§8-4): FullCopy転送で受信==アーカイブ件数
    Bqa = g["Agent"](seed + 1000)
    received, _, _ = g["transfer"](dict(A.archive), Bqa, gated=False)
    qa_ok = (received == n_arch)
    # (3) Isolated単独をDbig+で1000ep
    B = g["Agent"](seed + 1000)
    envB = g["ResonanceEnv"](g["PATROL_B_DBIGPLUS"])
    st = g["new_stats"]()
    g["run_episodes"](B, envB, g["N_EP_B"], st)
    routes = st["route"][-100:]
    mid_rate = routes.count("mid") / 100.0
    succ_rate = float(np.mean(st["succ"][-100:]))
    results.append((seed, mid_rate, succ_rate, qa_ok, n_arch, received,
                    st["collisions"]))
    print("  seed%d: mid使用率%.2f 成功率%.2f 衝突ep%d | QA: アーカイブ%d件 受信%d件 %s" % (
        seed, mid_rate, succ_rate, st["collisions"],
        n_arch, received, "OK" if qa_ok else "NG"))

print()
g1 = all(r[1] >= 0.90 for r in results)
g2 = all(r[2] >= 0.99 for r in results)
g3 = all(r[3] for r in results)
print("[G1] 最短路使用率>=90%% (全シード): %s" % ("合格" if g1 else "不合格"))
print("[G2] 成功率>=0.99 (全シード): %s" % ("合格" if g2 else "不合格"))
print("[G3] ロードQA 受信==アーカイブ件数: %s" % ("合格" if g3 else "不合格"))
print()
if g1 and g2 and g3:
    print("*** 土俵ゲート: 全項目合格。本実行(resonance_v04_full.py)へ進んでよい ***")
else:
    print("*** 土俵ゲート: 不合格。凍結解除・地形再設計。著者に報告すること ***")
