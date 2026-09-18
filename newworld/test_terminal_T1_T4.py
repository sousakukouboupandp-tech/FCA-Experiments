# -*- coding: utf-8 -*-
"""T1〜T4：終端の扱いを固定するテスト（ChatGPT 返答96・98の指示）。
★実装前に登録した検証項目（post-diagnostic ではない）★
  T1 飢え死ちょうど0    : D1a と D1b が同じになる
  T2 飢え死の行き過ぎ    : E<0。D1a は生の値、D1b は 0 → ★分岐する★
  T3 急所死で体力が正    : D1a は生の正の値、D1b は 0
  T4 天寿               : D1b の置換を発動させない
"""
import mini_world as m
from mini_solver_ref import Objective, drive

ok, ng = 0, []


def check(name, cond, detail=""):
    global ok
    if cond:
        ok += 1
        print(f"  OK   {name}  {detail}")
    else:
        ng.append(name)
        print(f"  ★NG★ {name}  {detail}")


E_SET, Q = 140, 2.0
d1a = Objective("D1a", e_set=E_SET, q=Q, gamma=0.99)
d1b = Objective("D1b", e_set=E_SET, q=Q, gamma=0.99)

print(f"T1〜T4（E_set={E_SET}, q={Q}）")
print()

# ---- T1：飢え死ちょうど0 -------------------------------------------------
# E=20, W=0, 消耗20 → E=0 → 飢え死
print("T1 飢え死ちょうど0（E=20 → E=0）")
br = [b for b in m.branches(m.SEARCH_SMALL, 20, 0, 3, 2, 5, None)
      if b[2] == m.STARVE]
check("T1 飢え死の枝が存在する", len(br) > 0, f"枝 {len(br)} 本")
b = br[0]
check("T1 終端の生の体力が0", b[3] == 0, f"raw_e={b[3]}")
r_a = d1a.step_value(20, b)[0]
r_b = d1b.step_value(20, b)[0]
exp = drive(20, E_SET, Q) - drive(0, E_SET, Q)
check("T1 D1a の報酬", abs(r_a - exp) < 1e-9, f"{r_a:.4f}（期待 {exp:.4f}）")
check("T1 D1b の報酬", abs(r_b - exp) < 1e-9, f"{r_b:.4f}")
check("T1 D1a と D1b が同じ", abs(r_a - r_b) < 1e-12, "E=0 なので一致するのが正しい")
print()

# ---- T2：飢え死の行き過ぎ -----------------------------------------------
# E=2, W=0, 消耗20 → E=-18 → 飢え死
print("T2 飢え死の行き過ぎ（E=2 → E=-18）")
br = [b for b in m.branches(m.SEARCH_SMALL, 2, 0, 3, 2, 5, None)
      if b[2] == m.STARVE]
b = br[0]
check("T2 終端の生の体力が-18（0に切らない）", b[3] == -18, f"raw_e={b[3]}")
r_a = d1a.step_value(2, b)[0]
r_b = d1b.step_value(2, b)[0]
exp_a = drive(2, E_SET, Q) - drive(-18, E_SET, Q)
exp_b = drive(2, E_SET, Q) - drive(0, E_SET, Q)
check("T2 D1a は生の値を使う", abs(r_a - exp_a) < 1e-9, f"{r_a:.4f}（期待 {exp_a:.4f}）")
check("T2 D1b は0として評価", abs(r_b - exp_b) < 1e-9, f"{r_b:.4f}（期待 {exp_b:.4f}）")
check("T2 D1a と D1b が★分岐する★", abs(r_a - r_b) > 1.0,
      f"差 {r_a - r_b:.4f}（D1a のほうが {'厳しい' if r_a < r_b else '甘い'}）")
print()

# ---- T3：急所死で体力が正 -----------------------------------------------
print("T3 急所死で体力が正（E=200 → 急所死）")
br = [b for b in m.branches(m.COMBAT, 200, 0, 3, 2, 5, None) if b[2] == m.ACUTE]
check("T3 急所死の枝が存在する", len(br) > 0, f"枝 {len(br)} 本")
b = br[0]
check("T3 終端の生の体力が正", b[3] > 0, f"raw_e={b[3]}")
r_a = d1a.step_value(200, b)[0]
r_b = d1b.step_value(200, b)[0]
exp_a = drive(200, E_SET, Q) - drive(b[3], E_SET, Q)
exp_b = drive(200, E_SET, Q) - drive(0, E_SET, Q)
check("T3 D1a は生の正の値を使う", abs(r_a - exp_a) < 1e-9, f"{r_a:.4f}")
check("T3 D1b は0として評価", abs(r_b - exp_b) < 1e-9, f"{r_b:.4f}")
check("T3 D1a と D1b が分岐する", abs(r_a - r_b) > 1.0, f"差 {r_a - r_b:.4f}")
print()

# ---- T4：天寿 ------------------------------------------------------------
print(f"T4 天寿（t={m.HORIZON-1} を生き抜く）")
br = [b for b in m.branches(m.GRASS, 200, 0, 3, 2, m.HORIZON - 1, None)
      if b[2] == m.TENJU]
check("T4 天寿の枝が存在する", len(br) > 0, f"枝 {len(br)} 本")
b = br[0]
check("T4 終端の生の体力が正", b[3] > 0, f"raw_e={b[3]}")
r_a = d1a.step_value(200, b)[0]
r_b = d1b.step_value(200, b)[0]
check("T4 ★天寿では D1b の置換が発動しない★", abs(r_a - r_b) < 1e-12,
      f"D1a={r_a:.4f}, D1b={r_b:.4f}")
exp = drive(200, E_SET, Q) - drive(b[3], E_SET, Q)
check("T4 両方とも生の体力を使う", abs(r_a - exp) < 1e-9 and abs(r_b - exp) < 1e-9,
      f"期待 {exp:.4f}")
print()

print(f"T1〜T4: 合格 {ok} 件 / NG {len(ng)} 件")
for n in ng:
    print("  NG:", n)
