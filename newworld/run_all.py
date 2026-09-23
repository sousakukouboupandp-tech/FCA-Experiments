# -*- coding: utf-8 -*-
"""②→④→⑤を順番に自動で走らせる（SPEC_追補_同点の幅_凍結 §9）。
各段は watchdog（RSS 8GB・二重起動ロック）経由。1段が失敗しても次へ進むが、結果は解釈しない。
"""
import subprocess, sys, os, time

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
steps = [("log_step2_forward_DS.txt", ["forward_eval.py", "DS"]),
         ("log_step4_constrained_D0.txt", ["constrained_passes.py", "D0"]),
         ("log_step5_constrained_DS.txt", ["constrained_passes.py", "DS"])]
for log, args in steps:
    t0 = time.time()
    with open("log_run_all.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 開始 {' '.join(args)}\n")
    r = subprocess.run([sys.executable, "-X", "utf8", "watchdog_run.py", "8.0", log] + args)
    with open("log_run_all.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 終了 {' '.join(args)}"
                f" code={r.returncode} {time.time()-t0:.0f}秒\n")
with open("log_run_all.txt", "a", encoding="utf-8") as f:
    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} ★全段終了★\n")
