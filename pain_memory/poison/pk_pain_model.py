# -*- coding: utf-8 -*-
"""
薬物動態モデル（線形1-コンパートメント経口投与モデル）を痛みの強度に転用する。
血中濃度 C(t) = Dose * ka / (Vd*(ka-kel)) * (exp(-kel*t) - exp(-ka*t))
これを「痛みの強さ」pain(t)として使う。ka=吸収速度定数、kel=消失速度定数。
"""
import numpy as np

def pk_pain_curve(t_array, ka, kel, dose_scale=1.0):
    """薬物動態の古典的Bateman関数。t=0で0、Tmaxでピーク、その後減衰。"""
    if abs(ka - kel) < 1e-6:
        kel = ka * 1.0001  # 特異点回避
    c = dose_scale * (ka / (ka - kel)) * (np.exp(-kel * t_array) - np.exp(-ka * t_array))
    return -np.abs(c)  # 痛みなのでマイナス

def tmax(ka, kel):
    return np.log(ka / kel) / (ka - kel)

def normalize_to_total(ka, kel, target_total=-10.0, max_t=30):
    """総ダメージ(離散和)がtarget_totalに一致するようdose_scaleを逆算する"""
    t_array = np.arange(0, max_t)
    raw = pk_pain_curve(t_array, ka, kel, dose_scale=1.0)
    raw_total = np.sum(raw)
    if raw_total == 0:
        return 0.0
    scale = target_total / raw_total
    return scale

# 3水準のパラメータ設計
# やかん型: 吸収が非常に速い(ka大)、消失も速い(kel大) -> 鋭く一瞬で終わる山
# 中間型: 中程度
# 毒型: 吸収が遅い(ka小)、消失も遅い(kel小) -> なだらかで長引く山
PARAM_SETS = {
    "やかん型": {"ka": 3.0, "kel": 2.5},
    "中間型":   {"ka": 1.0, "kel": 0.4},
    "毒型":     {"ka": 0.4, "kel": 0.15},
}

print("=== 薬物動態モデルによる3水準の設計 ===")
for name, p in PARAM_SETS.items():
    ka, kel = p["ka"], p["kel"]
    scale = normalize_to_total(ka, kel, target_total=-10.0, max_t=30)
    t_array = np.arange(0, 15)
    curve = pk_pain_curve(t_array, ka, kel, dose_scale=scale)
    tp = tmax(ka, kel)
    total = np.sum(pk_pain_curve(np.arange(0,30), ka, kel, dose_scale=scale))
    peak = np.min(curve)
    peak_t = int(np.argmin(curve))
    print(f"\n{name}: ka={ka}, kel={kel}, 正規化scale={scale:.3f}")
    print(f"  理論Tmax(ピーク到達時刻)={tp:.2f}, 実測ピーク時刻={peak_t}, ピーク強度={peak:.2f}")
    print(f"  総ダメージ(30ステップ和)={total:.3f}")
    print(f"  t=0~9の推移: {[round(x,2) for x in curve[:10]]}")
