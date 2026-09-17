# -*- coding: utf-8 -*-
"""A-1 / A-1b / A-2 / B の実行。
仕様：TEST_SPEC_World1G_v02.md（結果前コミット 1cde347）
世界：DESIGN_新世界_v02_World1G_表1と再現仕様.md（386bf76）
不都合な値も省略せず全部出す。実測を見てから合格線を変えない。
"""
import math, time, sys
import numpy as np
import world1g as w
from world1g import (State, Draws, transition, DECISION, SEARCH_SMALL, CHASE_4,
                     SEARCH_LARGE, COMBAT, ACT_GRASS, ACT_SMALL, ACT_LARGE, E_MAX)

SEEDS = [20260918, 20260919, 20260920, 20260921, 20260922]
BIG_E = E_MAX            # 部品テスト用。体力は毎歩作り直すので上限のままでよい
results = []


def report(name, theo_mean, theo_var, per_seed_sum, per_seed_n, unit=""):
    """pooled を主判定、seed 単独は |z|>5 で要調査"""
    tot_sum = sum(per_seed_sum)
    tot_n = sum(per_seed_n)
    mean = tot_sum / tot_n
    se = math.sqrt(theo_var / tot_n)
    z = (mean - theo_mean) / se if se > 0 else 0.0
    seed_rows = []
    flag = False
    for s, n, sd in zip(SEEDS, per_seed_n, per_seed_sum):
        m = sd / n
        zs = (m - theo_mean) / math.sqrt(theo_var / n)
        seed_rows.append((s, m, zs))
        if abs(zs) > 5:
            flag = True
    verdict = "PASS" if abs(z) <= 4 else "**不合格**"
    if flag:
        verdict += "（seed単独 |z|>5 あり：要調査）"
    results.append((name, theo_mean, mean, z, verdict, seed_rows, unit))
    return verdict


def show():
    print("=" * 78)
    for name, theo, mean, z, verdict, rows, unit in results:
        print(f"{name}")
        print(f"  理論 {theo:.6f}{unit}   実測 {mean:.6f}{unit}   pooled z = {z:+.3f}   {verdict}")
        print("  seed別: " + " / ".join(f"{s}:{m:.5f}(z{zs:+.2f})" for s, m, zs in rows))
    print("=" * 78)


# ---------------- A-1：エンジン統合試験 -----------------------------------
def run_combat(rng, n):
    """戦闘単体。寿命・飢え死・治癒・回復なし。死亡/ラウンド数/負傷発生数"""
    deaths = rounds = wounds = 0
    for _ in range(n):
        s = State(e=BIG_E, w=0, n_large=3, mode=COMBAT)
        r = 0
        while True:
            u = rng.random()
            if u < 0.25:
                out = "kill"
            elif u < 0.29:
                out = "acute"
            elif u < 0.65:
                out = "wound"
            else:
                out = "miss"
            d = Draws(combat=out)
            s_next = transition(s, None, d)
            r += 1
            if d.combat == "wound":
                wounds += 1
            if d.combat in ("acute", "kill"):
                if d.combat == "acute":
                    deaths += 1
                break
            # 傷の消耗を切って戦闘だけ測る（部品テストの条件）
            s = State(e=BIG_E, w=0, n_small=s.n_small, n_large=s.n_large,
                      t=s_next.t, mode=COMBAT)
        rounds += r
    return deaths, rounds, wounds


def run_small(rng, n, n_small):
    """小物：decision から次の decision まで。探索歩数・捕獲率・合計歩数を分けて記録"""
    search_steps = 0
    catches = 0
    total_steps = 0
    for _ in range(n):
        s = State(e=BIG_E, w=0, n_small=n_small, mode=DECISION)
        a = ACT_SMALL
        t0 = s.t
        while True:
            # その歩で実際に走る MODE（decision のときは選んだ行動の1歩目）。
            # sample_draws と同じ解釈に揃える。v02：decision は時間を使わない境界
            run_mode = w.start_mode(a) if s.mode == DECISION else s.mode
            d = w.sample_draws(s, a, rng)
            s_next = transition(s, a, d)
            if run_mode == SEARCH_SMALL and d.find:
                search_steps += s_next.t - t0
            if run_mode == CHASE_4 and d.catch:
                catches += 1
            a = None
            s = s_next
            if s.mode == DECISION:
                break
            s = State(e=BIG_E, w=0, n_small=n_small, n_large=s.n_large,
                      t=s.t, mode=s.mode)   # 体力と資源を固定して部品だけ測る
        total_steps += s.t - t0
    return search_steps, catches, total_steps


def run_large_search(rng, n, n_large):
    """大物の探索歩数だけ（遭遇した歩を含む）。回復なし。
    体力は毎歩作り直す（部品テストの条件：飢え死なし）"""
    steps = 0
    pf = w.large_search_p(n_large)
    for _ in range(n):
        s = State(e=E_MAX, w=0, n_large=n_large, mode=DECISION)
        a = ACT_LARGE
        t0 = s.t
        while s.mode != COMBAT:
            d = Draws(find=bool(rng.random() < pf))
            s = transition(s, a, d)
            a = None
            s = State(e=E_MAX, w=0, n_small=s.n_small, n_large=n_large,
                      t=s.t, mode=s.mode)
        steps += s.t - t0
    return steps


def run_heal(rng, n):
    """傷1つが治るまでの歩数"""
    steps = 0
    for _ in range(n):
        s = State(e=BIG_E, w=1, n_large=3, mode=w.GRASS)
        t0 = s.t
        while s.w > 0:
            d = w.Draws(heal=int(rng.binomial(s.w, w.HEAL_P)))
            s = transition(s, None, d)
            s = State(e=BIG_E, w=s.w, n_small=s.n_small, n_large=s.n_large,
                      t=s.t, mode=w.GRASS)
        steps += s.t - t0
    return steps


def run_recover(rng, n):
    """大物が1頭戻るまでの歩数（N_L=2 から 3 へ）"""
    steps = 0
    for _ in range(n):
        s = State(e=BIG_E, w=0, n_large=2, mode=w.GRASS)
        t0 = s.t
        while s.n_large == 2:
            d = w.Draws(recover=bool(rng.random() < w.LARGE_RECOVER_P))
            s = transition(s, None, d)
            s = State(e=BIG_E, w=0, n_small=s.n_small, n_large=s.n_large,
                      t=s.t, mode=w.GRASS)
        steps += s.t - t0
    return steps


# ---------------- A-1b：分布サンプラー（ベクトル化）------------------------
def sampler_geom(rng, n, p):
    """1歩ずつの抽選を幾何過程として生成（numpy の geometric は使わず、
    本番と同じ『毎歩 p で当たる』を逆変換で作る）。戻り値：初回当たりまでの歩数"""
    u = rng.random(n)
    return np.floor(np.log1p(-u) / np.log1p(-p)).astype(np.int64) + 1


def sampler_binom(rng, n, w_count, p):
    return rng.binomial(w_count, p, size=n)


# ---------------- A-2：性質テスト -----------------------------------------
def run_property(seed, n_steps):
    rng = np.random.default_rng(seed)
    s = State()
    prev = s
    violations = []
    episode = 0
    for i in range(n_steps):
        a = None
        if s.at_decision:
            acts = w.legal_actions(s)
            a = acts[int(rng.integers(len(acts)))]
        d = w.sample_draws(s, a, rng)
        nxt = transition(s, a, d)
        bad = []
        if not (1 <= nxt.n_small <= 12):
            bad.append("n_small 範囲外")
        if not (0 <= nxt.n_large <= 3):
            bad.append("n_large 範囲外")
        if nxt.e > E_MAX:
            bad.append("体力が上限超過")
        if nxt.alive and nxt.e <= 0:
            bad.append("生存なのに体力0以下")
        if s.n_large == 0 and nxt.n_large != 0:
            bad.append("N_L=0 から復活")
        if nxt.t != s.t + 1:
            bad.append("時刻が+1でない")
        if s.mode != DECISION and a is not None:
            bad.append("decision 以外で行動")
        if s.mode == CHASE_4 and not d.catch and nxt.n_small != s.n_small:
            bad.append("捕獲失敗で小物が減った")
        if d.combat == "acute" and nxt.alive:
            bad.append("急所死なのに生存")
        if d.combat == "wound" and nxt.w != s.w - d.heal + 1:
            bad.append("新しい傷が同じ歩の治癒に入った")
        if bad:
            violations.append((seed, episode, s.t, s, a, d, nxt, bad))
        prev, s = s, nxt
        if not s.alive:
            episode += 1
            s = State()
    return violations


# ---------------- B：宣言した変更の指紋 -----------------------------------
def trunc_geom_theory(p, R):
    q = 1 - p
    m = sum(q ** (k - 1) for k in range(1, R + 1))
    m2 = sum((2 * k - 1) * q ** (k - 1) for k in range(1, R + 1))
    return m, m2 - m * m


# ---------------- 実行 -----------------------------------------------------
def main():
    t_start = time.perf_counter()
    print("World_1-G テスト実行")
    print("世界仕様 386bf76 ／ 実装・テスト仕様 1cde347")
    print("Python", sys.version.split()[0], "／ numpy", np.__version__, "／ PCG64")
    print("seeds", SEEDS)
    print()

    # ---- A-1 戦闘
    print("[A-1] 戦闘（各 seed 100万回）...", flush=True)
    ds, rs, ws_, ns = [], [], [], []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        d_, r_, w_ = run_combat(rng, 1_000_000)
        ds.append(d_); rs.append(r_); ws_.append(w_); ns.append(1_000_000)
    p = w.COMBAT_ACUTE / (w.COMBAT_ACUTE + w.COMBAT_KILL)
    report("A-1 戦闘の死亡率", p, p * (1 - p), ds, ns)
    report("A-1 戦闘のラウンド数", 1 / 0.29, 0.71 / 0.29 ** 2, rs, ns)
    q = 0.29 / 0.65
    report("A-1 戦闘の負傷発生回数", (1 - q) / q, (1 - q) / q ** 2, ws_, ns)

    # ---- A-1 小物
    for n_small, trials in ((12, 1_000_000), (1, 100_000)):
        print(f"[A-1] 小物 N_S={n_small}（各 seed {trials:,}回）...", flush=True)
        ss, cs, ts, nn = [], [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            a, b, c = run_small(rng, trials, n_small)
            ss.append(a); cs.append(b); ts.append(c); nn.append(trials)
        pf = n_small / 24.0
        report(f"A-1 小物N_S={n_small} 探索歩数", 1 / pf, (1 - pf) / pf ** 2, ss, nn)
        report(f"A-1 小物N_S={n_small} 捕獲率", 0.7, 0.21, cs, nn)
        report(f"A-1 小物N_S={n_small} 合計歩数", 1 / pf + 4, (1 - pf) / pf ** 2, ts, nn)

    # ---- A-1 大物の探索
    for n_large, trials in ((3, 1_000_000), (1, 50_000)):
        print(f"[A-1] 大物 N_L={n_large}（各 seed {trials:,}回）...", flush=True)
        ss, nn = [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            ss.append(run_large_search(rng, trials, n_large)); nn.append(trials)
        pf = w.large_search_p(n_large)
        report(f"A-1 大物N_L={n_large} 探索歩数", 1 / pf, (1 - pf) / pf ** 2, ss, nn)

    # ---- A-1 傷の治癒・大物の回復
    print("[A-1] 傷の治癒（各 seed 2万回）...", flush=True)
    ss, nn = [], []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ss.append(run_heal(rng, 20_000)); nn.append(20_000)
    report("A-1 傷1つの治癒歩数", 150.0, 149 * 150.0, ss, nn)

    print("[A-1] 大物の回復（各 seed 1万回）...", flush=True)
    ss, nn = [], []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        ss.append(run_recover(rng, 10_000)); nn.append(10_000)
    report("A-1 大物1頭の回復歩数", 250.0, 249 * 250.0, ss, nn)

    show()

    # ---- A-1b 分布サンプラー
    print("\n[A-1b] 分布サンプラー（各 seed 100万）")
    cases = [("小物 N_S=1", 1 / 24, 24), ("小物 N_S=12", 12 / 24, 3),
             ("大物 N_L=3", 1 / 8, 8), ("大物 N_L=1", w.large_search_p(1), 40),
             ("傷の治癒", w.HEAL_P, 150), ("大物の回復", w.LARGE_RECOVER_P, 250)]
    for name, pf, kk in cases:
        means, p1s, tails, nn = [], [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            x = sampler_geom(rng, 1_000_000, pf)
            means.append(x.sum()); p1s.append(int((x == 1).sum()))
            tails.append(int((x > kk).sum())); nn.append(1_000_000)
        report(f"A-1b {name} 平均", 1 / pf, (1 - pf) / pf ** 2, means, nn)
        report(f"A-1b {name} P(T=1)", pf, pf * (1 - pf), p1s, nn)
        tail = (1 - pf) ** kk
        report(f"A-1b {name} P(T>{kk})", tail, tail * (1 - tail), tails, nn)
    # 複数傷の独立治癒
    sums, nn = [], []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        sums.append(int(sampler_binom(rng, 1_000_000, 5, w.HEAL_P).sum())); nn.append(1_000_000)
    report("A-1b 複数傷(W=5)の1歩の治癒数", 5 * w.HEAL_P, 5 * w.HEAL_P * (1 - w.HEAL_P), sums, nn)
    show()

    # ---- A-2 性質テスト
    print("\n[A-2] 性質テスト（各 seed 100万歩）...", flush=True)
    total_v = 0
    for sd in SEEDS:
        v = run_property(sd, 1_000_000)
        total_v += len(v)
        print(f"  seed {sd}: 違反 {len(v)} 件")
        if v:
            seed_, ep, t_, s_, a_, d_, n_, bad = v[0]
            print("   最初の違反:", bad)
            print("   前:", s_, "\n   行動:", a_, "\n   抽選:", d_, "\n   次:", n_)
    print(f"A-2 合計違反 {total_v} 件")

    # ---- B 指紋
    print("\n[B] 傷の打ち切り支払い歩数 min(T,R)（各 seed 20万）")
    for R in (800, 400, 200, 150, 100, 50):
        th_m, th_v = trunc_geom_theory(w.HEAL_P, R)
        sums, nn = [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            x = np.minimum(sampler_geom(rng, 200_000, w.HEAL_P), R)
            sums.append(int(x.sum())); nn.append(200_000)
        report(f"B R={R} の E[min(T,R)]", th_m, th_v, sums, nn)
    show()

    print("実行時間 %.1f 秒" % (time.perf_counter() - t_start))


if __name__ == "__main__":
    main()
