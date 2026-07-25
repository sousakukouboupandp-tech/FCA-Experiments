# -*- coding: utf-8 -*-
"""
新世界 予行計算 v0 — 葛藤の存在証明の机上版（数字はすべて仮置き）
問い: この世界は「賭けずには生きられない」か。楽園でも地獄でもない帯はどこか。
"""
import numpy as np

# ---- 仮パラメータ ----
LIFESPAN   = 1000    # 天寿（ステップ）
E_MAX      = 100.0   # 体力上限
UPKEEP     = 1.0     # 1ステップの基礎消耗（移動込みの平均）

# 逃げない餌（草）: 実効摂取レート = 栄養 / 到達間隔
PLANT_NUTRI    = 6.0
PLANT_INTERVAL = 10.0   # 何ステップに1個食えるか（密度と再生で決まる）

# 逃げる餌（小動物）
CHASE_STEPS = 10      # 追跡に要する歩数（消耗=歩数×UPKEEP）
CATCH_P     = 0.7     # 捕獲成功率
SMALL_NUTRI = 40.0

# 反撃する餌（大物）
KILL_P      = 0.4     # 1交戦ラウンドでの仕留め成功率
INJURY_P    = 0.3     # 1ラウンドで反撃を食らう確率
VITAL_P     = 0.10    # 反撃が急所に入る確率（即死相当）
BIG_NUTRI   = 120.0

print("="*60)
print("【1】安全だけ戦略（草オンリー）")
rate = PLANT_NUTRI / PLANT_INTERVAL
net  = rate - UPKEEP
print(f"  摂取レート {rate:.2f}/歩  −  消耗 {UPKEEP:.2f}/歩  =  純収支 {net:+.2f}/歩")
if net < 0:
    t_starve = E_MAX / (-net)
    print(f"  → 満タンから飢死まで {t_starve:.0f} 歩（天寿{LIFESPAN}の {100*t_starve/LIFESPAN:.0f}%）")
    print("  → 安全だけでは生きられない：成立")
else:
    print("  → 楽園化（賭け不要）。PLANT側を締める必要あり")

print()
print("【2】小物狩り戦略（逃げる餌）")
# 1回の狩り: CHASE_STEPS歩の消耗、確率CATCH_Pで栄養獲得
cost = CHASE_STEPS * UPKEEP
ev   = CATCH_P * SMALL_NUTRI - cost
per_step = ev / CHASE_STEPS
print(f"  1狩りの期待値 = {CATCH_P}×{SMALL_NUTRI} − {cost} = {ev:+.1f}（{per_step:+.2f}/歩）")
# 空振り連続で死ぬ確率: 満タンから、失敗のたび-10。破産までの失敗許容回数
fail_budget = int(E_MAX // cost)
p_ruin = (1-CATCH_P) ** fail_budget
print(f"  満タンから連続空振り破産まで {fail_budget} 回。その確率 {(p_ruin*100):.3f}%")
print(f"  → 働けば食えるが、失敗はちゃんと痛い：{'成立' if ev>0 else '地獄化'}")

print()
print("【3】大物狩り戦略（反撃する餌）")
# 1交戦ラウンドの遷移: 仕留めKILL_P / 負傷INJURY_P（うち急所VITAL_P）/ 継続
# 1回の狩りで死ぬ確率（仕留めるまでラウンド反復、簡略化: 独立試行）
p_die_round = INJURY_P * VITAL_P
# 決着（仕留め or 死）までの吸収確率
p_die_per_hunt = p_die_round / (p_die_round + KILL_P)
hunts_to_death = 1.0 / p_die_per_hunt
rounds_per_hunt = 1.0 / (p_die_round + KILL_P)
print(f"  1ラウンド即死率 = {INJURY_P}×{VITAL_P} = {p_die_round:.3f}")
print(f"  1狩りあたり死亡率 ≈ {p_die_per_hunt*100:.1f}% / 期待{rounds_per_hunt:.1f}ラウンド")
print(f"  期待 {hunts_to_death:.1f} 回目の狩りで死ぬ")
# 天寿まで大物だけで食うなら何回狩る必要があるか
need_energy = LIFESPAN * UPKEEP
hunts_needed = need_energy / BIG_NUTRI
p_survive_life = (1 - p_die_per_hunt) ** hunts_needed
print(f"  天寿まで大物だけなら {hunts_needed:.1f} 回必要 → 生き残る確率 {p_survive_life*100:.1f}%")

print()
print("【4】帯の走査：草の実効レートを振って世界の顔を見る")
print(f"  {'草レート':>8} {'純収支/歩':>9}  世界の顔")
for interval in [4, 6, 8, 10, 15, 25, 60]:
    r = PLANT_NUTRI / interval
    n = r - UPKEEP
    if n >= 0:
        face = "楽園（賭け不要・前世界の再来）"
    elif n > -0.3:
        face = "微圧（狩りは時々でよい）"
    elif n > -0.7:
        face = "★帯内（狩らねば死ぬが、狩れば生きられる）"
    else:
        face = "地獄寄り（草がほぼ無意味）"
    print(f"  {r:8.2f} {n:+9.2f}  {face}")

print()
print("【5】結論（仮数字での判定）")
print("  草だけ→確実に飢死 / 小物→期待値プラスだが空振りが痛い /")
print("  大物→一発は旨いが通いつめると死ぬ。三すくみは数字の上で成立。")
print("  葛藤の存在証明: 「安全＝死、賭け＝生存の可能性」が机上で示せた。")
