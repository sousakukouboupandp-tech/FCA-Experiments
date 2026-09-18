# -*- coding: utf-8 -*-
"""tuple/set 版（reach_probe4）と dense 版（reach_dense）の一致検証。
比較するのは表現ではなく、到達した decision 状態の集合そのもの。
両方とも正準形 (e,w,ns,nl) に戻して正順に並べ、SHA-256 を取る。
"""
import time, sys
import reach_probe4 as tup
import reach_dense as den

MAXT = int(sys.argv[1]) if len(sys.argv) > 1 else 11

print(f"tuple/set 版を走らせる（t=0..{MAXT}）", flush=True)
t0 = time.perf_counter()
rows_t, _, _, _ = tup.run(MAXT, quiet=True)
sec_t = time.perf_counter() - t0
print("  %.2f 秒" % sec_t, flush=True)

print("dense 版を走らせる", flush=True)
t0 = time.perf_counter()
rows_d, _, _, _ = den.run(MAXT, quiet=True)
sec_d = time.perf_counter() - t0
print("  %.2f 秒" % sec_d, flush=True)

print()
print("  t | tuple |S_t| | dense |S_t| | tuple digest     | dense digest     | 一致")
ok = True
for a, b in zip(rows_t, rows_d):
    same = (a["dec"] == b["dec"] and a["digest"] == b["digest"])
    ok = ok and same
    print("%3d | %11d | %11d | %s | %s | %s"
          % (a["t"], a["dec"], b["dec"], a["digest"], b["digest"],
             "OK" if same else "★不一致★"))
print()
print("frontier（MODE 込み）の一致も確認")
f_ok = all(a["frontier"] == b["frontier"] for a, b in zip(rows_t, rows_d))
print("  frontier count:", "全時刻一致" if f_ok else "★不一致★")
print()
print("★全時刻で count と digest が一致★" if ok and f_ok else "★不一致あり★")
print("速度: tuple/set %.2f 秒 ／ dense %.2f 秒" % (sec_t, sec_d))
