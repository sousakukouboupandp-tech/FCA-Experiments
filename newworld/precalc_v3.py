# -*- coding: utf-8 -*-
r"""予行計算 v3 — 語りから数字を絞り出す【裁定：語りが構造体を構造化させる】
語り→数字の対応:
  草   : 一日籠っても一割足りない      → 実効レート 0.7 (消耗1.0に対し)
  小物 : 一匹=三十歩分の命            → 栄養30 / 初期12匹・狩るほど探す足が伸びる
  大物 : 小物三匹分=九十歩分 / 野に三頭 → 栄養90 / 残3
  抗い : 七回に一回は帰らぬ            → 1狩り死亡率 13.8%【始祖裁定で採用】
  借金 : 傷は残りの一生で分割払い      → 傷1つ=消耗+0.25/歩 × 塞がるまで150歩(残寿命で頭打ち)
実行: C:\Python312\python.exe -X utf8 precalc_v3.py
"""
UPKEEP=1.0; E_MAX=100.0; LIFESPAN=1000; K=24.0
PLANT_R=0.70
SMALL_N=30.0; CATCH=0.7; CHASE=4; NS0=12
BIG_N=90.0; NB=3; KILL=0.25; INJ=0.40; VIT=0.10
W_UP=0.25; HEAL=150

def small(n):
    st=K/n+CHASE; ev=CATCH*SMALL_N-st*UPKEEP; return ev/st, st, ev

def big(rl):
    search=K/NB
    pdr=INJ*VIT
    pdie=pdr/(pdr+KILL)
    rounds=1.0/(pdr+KILL)
    st=search+rounds
    wounds=rounds*INJ*(1-VIT)
    debt=wounds*W_UP*min(HEAL,rl)
    ev=(1-pdie)*BIG_N-st*UPKEEP-debt
    return ev/st, st, ev, pdie, wounds, debt

print("="*66)
print(f"【草】 純収支 {PLANT_R-UPKEEP:+.2f}/歩 → 満タンから{E_MAX/(UPKEEP-PLANT_R):.0f}歩で静かに死ぬ")
r,st,ev,pd,wd,db = big(rl=800)
print(f"【大物】探索{K/NB:.0f}歩+抗い{st-K/NB:.1f}歩 / 1狩り死亡率{pd*100:.1f}% 期待負傷{wd:.2f}")
print(f"        借金{db:.1f}点を差引 → {r:+.2f}/歩 (若い個体・残寿命800)")
print("="*66)
print(f"{'小物残':>5} {'/歩':>7}   若い大物{r:+.2f}との差   世界の顔")
print("-"*66)
for n in [12,10,8,7,6,5,4,3,2]:
    sr=small(n)[0]; d=sr-r
    if   d> 0.4: face="小物の季節"
    elif d>-0.4: face="★迷いの季節（葛藤の帯）"
    else:        face="賭けの季節（大物へ）"
    print(f"{n:>5} {sr:+7.2f}   {d:+6.2f}            {face}")
print("-"*66)
print("【老いの効き】残り寿命が短いほど借金が軽い＝賭けが安くなる")
for rl in [800,400,200,100,50]:
    rr=big(rl)[0]
    print(f"  残寿命{rl:>4}歩 → 大物 {rr:+.2f}/歩")
print("-"*66)
print("【判定】草=確実な緩死 / 小物=序盤の正解だが枯れて足が伸びる /")
print("大物=中盤から迷いに入り、老いるほど賭けが安い。正解が時期と個体で動く。")
