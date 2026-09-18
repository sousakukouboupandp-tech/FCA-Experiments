# -*- coding: utf-8 -*-
"""World_1-G の dense 表現（固定 index）と、到達可能集合の support 列挙。
仕様：SOLVER_SPEC（f6c1ad0→503d2f4→a66e9e4）＋ dense 追補（4ba05ee→e989be9→38d5a61→57dc591）

index（凍結済み）
  i = ((((m * 2001 + e) * 11 + w) * 12 + (n_small - 1)) * 4 + n_large)
  m : decision=0, search_small=1, chase_1=2, chase_2=3, chase_3=4,
      chase_4=5, search_large=6, combat=7
  e : 0..2000 ／ w : 0..10（index 10 は「W >= 10 のあふれ」）
  n_small : 1..12 → 0..11 ／ n_large : 0..3
  合計 8 * 2001 * 11 * 12 * 4 = 8,452,224

★世界のルールは world1g.py（正本）のまま。ここは表現の変換だけ。★
"""
import numpy as np
import world1g as w
from world1g import (DECISION, SEARCH_SMALL, CHASE_1, CHASE_2, CHASE_3,
                     CHASE_4, SEARCH_LARGE, COMBAT, E_MAX)

K_WOUND = 10
N_E = E_MAX + 1          # 2001
N_W = K_WOUND + 1        # 11（最後の枠は「あふれ」）
N_S = 12                 # n_small 1..12
N_L = 4                  # n_large 0..3

MODES = (DECISION, SEARCH_SMALL, CHASE_1, CHASE_2, CHASE_3,
         CHASE_4, SEARCH_LARGE, COMBAT)
MODE_ID = {m: i for i, m in enumerate(MODES)}
N_M = len(MODES)         # 8

TOTAL = N_M * N_E * N_W * N_S * N_L
STRIDE_M = N_E * N_W * N_S * N_L
STRIDE_E = N_W * N_S * N_L
STRIDE_W = N_S * N_L
STRIDE_S = N_L


def to_index(mode, e, wnd, n_small, n_large):
    """状態 → 固定 index"""
    m = MODE_ID[mode]
    return (((m * N_E + e) * N_W + min(wnd, K_WOUND)) * N_S
            + (n_small - 1)) * N_L + n_large


def from_index(i):
    """固定 index → 状態（往復の検証用）"""
    n_large = i % N_L
    i //= N_L
    n_small = i % N_S + 1
    i //= N_S
    wnd = i % N_W
    i //= N_W
    e = i % N_E
    i //= N_E
    return (MODES[i], e, wnd, n_small, n_large)


def to_index_arr(m_arr, e_arr, w_arr, s_arr, l_arr):
    """配列版。numpy で一括変換する"""
    return (((m_arr * N_E + e_arr) * N_W + w_arr) * N_S
            + (s_arr - 1)) * N_L + l_arr


def decision_index_of(e, wnd, n_small, n_large):
    """decision 状態（m=0）の index"""
    return ((e * N_W + min(wnd, K_WOUND)) * N_S + (n_small - 1)) * N_L + n_large
