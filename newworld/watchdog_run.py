# -*- coding: utf-8 -*-
"""RSS watchdog。子プロセスとして実験を走らせ、親が RSS を監視して止める。
使い方: python watchdog_run.py <上限GB> <出力ファイル> <スクリプト> [引数...]
2026/9/18 にメモリ事故が2回起きたため、停止条件を実験コードの外側に置く。
"""
import subprocess, sys, time, os
import psutil

def main():
    limit_gb = float(sys.argv[1])
    out_path = sys.argv[2]
    cmd = [sys.executable, "-X", "utf8"] + sys.argv[3:]
    script = sys.argv[3]

    # ★★二重起動の防止★★（2026/9/20 に同じ計算を2回起動した事故のため）
    # 同じスクリプトが既に走っていたら、起動を拒否する
    running = []
    me = os.getpid()
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            if p.info["pid"] in (me, os.getppid()):
                continue
            cl = p.info.get("cmdline") or []
            if any("python" in str(c).lower() for c in cl[:1]) and \
               any(script in str(c) for c in cl):
                running.append((p.info["pid"], p.info["create_time"], " ".join(cl)))
        except Exception:
            continue
    if running:
        print("*" * 70)
        print(f"★★起動を拒否した：{script} が既に走っている★★")
        for (pid, ct, cl) in running:
            print(f"  PID {pid}  起動 {time.strftime('%H:%M:%S', time.localtime(ct))}")
            print(f"    {cl[:110]}")
        print()
        print("  → 走っているものの出力ファイルを読むか、")
        print("     止めたいなら PID を指定して終了させてから、もう一度実行する")
        print("*" * 70, flush=True)
        sys.exit(3)

    print(f"watchdog: 上限 {limit_gb}GB / 出力 {out_path}")
    print("実行:", " ".join(cmd), flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        proc = psutil.Process(p.pid)
        peak = 0.0
        t0 = time.perf_counter()
        while p.poll() is None:
            try:
                rss = proc.memory_info().rss / 1024**3
            except Exception:
                break
            peak = max(peak, rss)
            if rss > limit_gb:
                p.kill()
                el = time.perf_counter() - t0
                msg = (f"\n*** watchdog が停止させた: RSS {rss:.2f}GB > 上限 {limit_gb}GB"
                       f"（経過 {el:.1f}秒）***\n")
                print(msg, flush=True)
                with open(out_path, "a", encoding="utf-8") as g:
                    g.write(msg)
                sys.exit(2)
            time.sleep(0.25)
        el = time.perf_counter() - t0
        print(f"watchdog: 子プロセス終了 code={p.returncode} "
              f"ピークRSS {peak:.2f}GB 経過 {el:.1f}秒", flush=True)


if __name__ == "__main__":
    main()
