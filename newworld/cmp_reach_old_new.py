# -*- coding: utf-8 -*-
"""旧版(reach_probe2 の辿り方)と新版(reach_probe4)の S_t を digest で突き合わせる。
高速化で support が変わっていないことの検証。
"""
import hashlib, time
import reach_probe2 as old
import reach_probe4 as new
import world1g as w

MAXT = 11


def dg(states):
    h = hashlib.sha256()
    for rec in sorted(states):
        h.update(("%d,%d,%d,%d;" % rec).encode("ascii"))
    return h.hexdigest()[:16]


print("旧版を走らせる（t=0..%d）" % MAXT, flush=True)
t0 = time.perf_counter()
BUF = old.TAU_MAX + 1
buf = [set() for _ in range(BUF)]
buf[0].add((w.E_MAX, 0, 12, 3))
old_rows = []
for t in range(MAXT + 1):
    cur = buf[t % BUF]
    old_rows.append((t, len(cur), dg(cur)))
    for (e, wn, ns, nl) in cur:
        s = old.State(e=e, w=wn, n_small=ns, n_large=nl, t=t, mode=old.DECISION)
        for a in w.legal_actions(s):
            for tau, landed in old.macro_support(e, wn, ns, nl, a, w.HORIZON - t).items():
                if t + tau < w.HORIZON:
                    buf[(t + tau) % BUF] |= landed
    buf[t % BUF] = set()
old_sec = time.perf_counter() - t0
print("旧版 %.1f 秒" % old_sec, flush=True)

print("新版を走らせる", flush=True)
t0 = time.perf_counter()
new_rows, stop, total, _ = new.run(MAXT, quiet=True)
new_sec = time.perf_counter() - t0
print("新版 %.2f 秒" % new_sec, flush=True)

print()
print("  t |  旧 |S_t| |  新 |S_t| | 旧 digest        | 新 digest        | 一致")
ok = True
for (t, n_old, d_old), r in zip(old_rows, new_rows):
    same = (n_old == r["dec"] and d_old == r["digest"])
    ok = ok and same
    print("%3d | %9d | %9d | %s | %s | %s"
          % (t, n_old, r["dec"], d_old, r["digest"], "OK" if same else "不一致"))
print()
print("全時刻で count と digest が一致" if ok else "★不一致あり★")
print("速度比: %.0f 倍" % (old_sec / max(new_sec, 1e-9)))
