# -*- coding: utf-8 -*-
"""
嘘の伝播と語りの減衰 v07 本実行（A→B→C三世代・思想注入第5弾）
事前登録: DESIGN_嘘の伝播_v02_凍結.md（2026年7月16日凍結・2R監査・3AI同意・エラッタ1適用）
著者の方向仮説:「話は伝わらない。伝わるのは、中継者の傷になった部分だけである。」
方式: resonance_v05_sweep.py の関数群を全流用(exec)。Agentは継承拡張（TrackedAgent）で
      出自タグ・昇格追跡を外付けし、v05のロジック・乱数消費は一切変更しない。
主要検定: P1 = R_chain(T1000,E1) < R_firsthand ／ P2 = R_chain(T150,E1) > R_chain(T1000,E1)
使い方: python lie_prop.py diag | full
"""
import sys, os, json, time, copy
_BASE7 = os.path.dirname(os.path.abspath(__file__))
_v05 = open(os.path.join(_BASE7, 'resonance_v05_sweep.py'), encoding='utf-8').read()
exec(compile(_v05.split('if __name__')[0], 'resonance_v05_sweep.py', 'exec'), globals())

SEEDS7 = list(range(30))
CHILDHOOD = 100
T_SNAP = (150, 1000)

class TrackedAgent(Agent):
    """v05 Agentの継承。tags/ep_countは観測専用メタデータ（凍結§3・UT-Bで無影響を実証）。
    行動決定・S計算・減衰のいかなる経路もtagsを参照しない。"""
    def __init__(self, seed, name):
        super().__init__(seed)
        self.tags = {}
        self.name = name
        self.ep_count = 0

    def record_pain(self, state, intended):
        k = (state, intended)
        tg = self.tags.get(k)
        if tg is not None:
            if tg["origin"] == "A" and not tg["promoted"]:
                tg["promoted"] = True
                tg["promotion_ep"] = self.ep_count + 1  # 初回衝突epで固定（凍結§4）
        else:
            self.tags[k] = {"origin": self.name, "promoted": False, "promotion_ep": None}
        super().record_pain(state, intended)

    def decay(self):
        super().decay()
        self.ep_count += 1

def transfer_tagged(tale, tale_tags, rec, gated, E):
    """凍結§3: S計算・ゲート判定はベース強度（E非依存）。書き込み強度にのみEを乗算。
    tale: dict key->intensity。戻り値に受理キー集合を含む。"""
    own = list(rec.archive.keys())
    n_received = 0
    S_presented, S_accepted, acc_keys = [], [], []
    for k, inten in tale.items():
        if gated:
            S = max([sim_entry(k, kb) for kb in own], default=0.0)
            I_base = inten * ETA_RES * S
        else:
            S = 1.0
            I_base = inten
        S_presented.append(S)
        if I_base > 0.05:
            I = I_base * E
            if k in rec.pm:
                rec.pm[k][0] = max(rec.pm[k][0], I)
                rec.pm[k][2] = max(rec.pm[k][2], S)
            else:
                rec.pm[k] = [I, False, S]
            n_received += 1
            S_accepted.append(S)
            acc_keys.append(k)
            src = tale_tags.get(k)
            if src is None:
                rec.tags[k] = {"origin": "A", "promoted": False, "promotion_ep": None}
            else:
                rec.tags[k] = {"origin": src["origin"], "promoted": bool(src["promoted"]),
                               "promotion_ep": src["promotion_ep"]}
    return n_received, S_presented, S_accepted, acc_keys

def snap_tale(agent):
    """語り時点のpmスナップショット（深いコピー・凍結§3）"""
    tale = {k: float(v[0]) for k, v in agent.pm.items()}
    tags = {}
    for k in tale:
        tg = agent.tags.get(k)
        tags[k] = dict(tg) if tg else {"origin": agent.name, "promoted": False, "promotion_ep": None}
    return tale, tags

def run_B(seed, A_tale, E):
    """B: 幼年期100epで受信(誇張E)→ep150/1000でスナップショット。ラン1本・乱数追加消費なし。"""
    B = TrackedAgent(seed + 1000, "B")
    env = ResonanceEnv(PATROL_B_D0)
    st = new_stats()
    run_episodes(B, env, CHILDHOOD, st)
    a_tags = {k: {"origin": "A", "promoted": False, "promotion_ep": None} for k in A_tale}
    recv, S_pre, S_acc, _ = transfer_tagged(A_tale, a_tags, B, gated=True, E=E)
    run_episodes(B, env, T_SNAP[0] - CHILDHOOD, st)
    tale150, tags150 = snap_tale(B)
    h150 = repr(sorted(tale150.items()))
    run_episodes(B, env, N_EP_B - T_SNAP[0], st)
    tale1000, tags1000 = snap_tale(B)
    assert repr(sorted(tale150.items())) == h150, "UT-D不合格: snap150が汚染された"
    def a_split(tags, tale):
        aA = [k for k in tale if tags[k]["origin"] == "A"]
        return len(aA), sum(1 for k in aA if tags[k]["promoted"])
    n150, p150 = a_split(tags150, tale150)
    n1000, p1000 = a_split(tags1000, tale1000)
    return {"tales": {150: (tale150, tags150), 1000: (tale1000, tags1000)},
            "S_pre": [float(s) for s in S_pre], "received": recv,
            "coll_flags": st["coll_flags"],
            "b_stats": {"A_alive_150": n150, "A_promoted_150": p150,
                        "A_alive_1000": n1000, "A_promoted_1000": p1000}}

def run_C(seed, arm, tale=None, tags=None, nA=None, iso_pobj=None):
    """C: 5アーム共通。isolated以外は幼年期100ep後に受信。"""
    C = TrackedAgent(seed + 2000, "C")
    env = ResonanceEnv(PATROL_B_D0)
    st = new_stats()
    if arm == "isolated":
        run_episodes(C, env, N_EP_B, st)
        return {"arm": arm, "coll_flags": st["coll_flags"], "coll_pos": st["coll_pos"],
                "R": None, "M": None, "delta": None, "n_presented": 0, "received": 0}
    run_episodes(C, env, CHILDHOOD, st)
    recv, S_pre, S_acc, acc_keys = transfer_tagged(tale, tags, C, gated=True, E=1.0)
    a_keys = [k for k in acc_keys if C.tags[k]["origin"] == "A"]
    promoted = [k for k in a_keys if C.tags[k]["promoted"]]
    R = len(a_keys) / float(nA)
    M = (len(promoted) / float(len(a_keys))) if a_keys else None
    # Δ（記述・エラッタ1適用: 49セル・ID=x*7+y）
    delta = None
    if iso_pobj is not None:
        subj = np.zeros(49)
        for k, v in C.pm.items():
            (pos, _), _ = k
            subj[pos[0] * 7 + pos[1]] += v[0]
        if subj.sum() > 0 and iso_pobj.sum() > 0:
            delta = float(0.5 * np.abs(subj / subj.sum() - iso_pobj / iso_pobj.sum()).sum())
    run_episodes(C, env, N_EP_B - CHILDHOOD, st)
    return {"arm": arm, "coll_flags": st["coll_flags"],
            "R": R, "M": M, "delta": delta,
            "n_A_arrived": len(a_keys), "n_A_promoted_arrived": len(promoted),
            "n_presented": len(S_pre), "received": recv}

def pobj_from(coll_pos):
    v = np.zeros(49)
    for pos in coll_pos:
        v[pos[0] * 7 + pos[1]] += 1.0
    return v

def run_seed(seed, A_tale):
    nA = len(A_tale)
    B1 = run_B(seed, A_tale, E=1.0)
    B3 = run_B(seed, A_tale, E=3.0)
    # UT-C: S計算はE非依存（凍結§3の処理順序の実証）
    assert B1["S_pre"] == B3["S_pre"], "UT-C不合格: EがS計算に混入"
    iso = run_C(seed, "isolated")
    pobj = pobj_from(iso["coll_pos"])
    a_tags = {k: {"origin": "A", "promoted": False, "promotion_ep": None} for k in A_tale}
    arms = {
        "isolated": iso,
        "firsthand": run_C(seed, "firsthand", A_tale, a_tags, nA, pobj),
        "chain_T1000_E1": run_C(seed, "chain", *B1["tales"][1000], nA, pobj),
        "chain_T150_E1": run_C(seed, "chain", *B1["tales"][150], nA, pobj),
        "chain_T1000_E3": run_C(seed, "chain", *B3["tales"][1000], nA, pobj),
    }
    # UT-A: FirstHand提示件数==A_cache件数
    assert arms["firsthand"]["n_presented"] == nA, "UT-A不合格: FirstHand提示件数不一致"
    return {"nA": nA, "B_E1": B1["b_stats"], "B_E3": B3["b_stats"], "arms": arms}

def ut_B_noninterference(A_tale):
    """UT-B: タグ機構が挙動に無影響（TrackedAgent vs 素のAgentで衝突flags完全一致）"""
    Bt = TrackedAgent(1000, "B")
    envt = ResonanceEnv(PATROL_B_D0)
    stt = new_stats()
    run_episodes(Bt, envt, CHILDHOOD, stt)
    a_tags = {k: {"origin": "A", "promoted": False, "promotion_ep": None} for k in A_tale}
    transfer_tagged(A_tale, a_tags, Bt, gated=True, E=1.0)
    run_episodes(Bt, envt, N_EP_B - CHILDHOOD, stt)
    Bp = Agent(1000)
    envp = ResonanceEnv(PATROL_B_D0)
    stp = new_stats()
    run_episodes(Bp, envp, CHILDHOOD, stp)
    transfer(A_tale, Bp, gated=True)
    run_episodes(Bp, envp, N_EP_B - CHILDHOOD, stp)
    return stt["coll_flags"] == stp["coll_flags"]

def diag7():
    print("=== 診断モード（seed0・構造チェックのみ・統計なし） ===")
    A_tale = learn_A(0)
    print("Aアーカイブ件数:", len(A_tale))
    ok = ut_B_noninterference(A_tale)
    print("UT-B(タグ無影響):", "合格" if ok else "不合格→停止")
    if not ok: sys.exit(1)
    r = run_seed(0, A_tale)   # UT-A/C/Dは内部assertで検証
    print("UT-A/C/D: 内部assert全通過")
    b = r["B_E1"]
    print("B(E=1): A由来生存 ep150=%d(昇格%d) / ep1000=%d(昇格%d)" % (
        b["A_alive_150"], b["A_promoted_150"], b["A_alive_1000"], b["A_promoted_1000"]))
    if b["A_alive_1000"] > b["A_promoted_1000"]:
        print("!!! λ検算と矛盾: ep1000に未昇格の又聞きが生存 → 実装確認要"); sys.exit(1)
    for arm in ("firsthand", "chain_T150_E1", "chain_T1000_E1", "chain_T1000_E3"):
        a = r["arms"][arm]
        print("%-16s R=%.3f 受理A由来=%d(昇格%d) M=%s Δ=%s" % (
            arm, a["R"], a["n_A_arrived"], a["n_A_promoted_arrived"],
            "NA" if a["M"] is None else "%.2f" % a["M"],
            "NA" if a["delta"] is None else "%.3f" % a["delta"]))
    for arm, a in r["arms"].items():
        assert len(a["coll_flags"]) == 1000, "flags長異常: " + arm
    print("=== 診断: 全チェック合格 ===")

def full7():
    t0 = time.time()
    out = {"seeds": {}}
    for seed in SEEDS7:
        A_tale = learn_A(seed)
        out["seeds"][str(seed)] = run_seed(seed, A_tale)
        if (seed + 1) % 5 == 0:
            print("  seed%d まで完了 (%.0f秒)" % (seed, time.time() - t0))
    return out, t0

def analyze7(out):
    def col(arm, field):
        return [out["seeds"][str(s)]["arms"][arm][field] for s in SEEDS7]
    r_first = col("firsthand", "R")
    r_1000 = col("chain_T1000_E1", "R")
    r_150 = col("chain_T150_E1", "R")
    P1 = paired_wilcoxon_pratt(r_1000, r_first)   # mean_diff<0 = 伝言で減衰（著者仮説方向）
    P2 = paired_wilcoxon_pratt(r_150, r_1000)     # mean_diff>0 = 早い語りほど届く
    if P1 is not None and P2 is not None:
        h1, h2 = holm_2(P1["p"], P2["p"]); P1["p_holm"] = h1; P2["p_holm"] = h2
    # 探索枠（検定判定に使わない）: 嘘E=3
    c_e3 = [coll_interval(f, 101, 1000) for f in col("chain_T1000_E3", "coll_flags")]
    c_e1 = [coll_interval(f, 101, 1000) for f in col("chain_T1000_E1", "coll_flags")]
    expl_lie = paired_wilcoxon_pratt(c_e3, c_e1)
    return {"P1": P1, "P2": P2,
            "data": {"r_first": r_first, "r_1000": r_1000, "r_150": r_150},
            "expl_lie_coll": expl_lie,
            "expl_lie_data": {"e3": c_e3, "e1": c_e1}}

def report7(out, tests):
    print()
    print("--- 記述（30シード） ---")
    for arm in ("firsthand", "chain_T150_E1", "chain_T1000_E1", "chain_T1000_E3"):
        rs = [out["seeds"][str(s)]["arms"][arm] for s in SEEDS7]
        Rv = [r["R"] for r in rs]
        Ms = [r["M"] for r in rs if r["M"] is not None]
        na = sum(1 for r in rs if r["M"] is None)
        ds = [r["delta"] for r in rs if r["delta"] is not None]
        print("%-16s R平均=%.4f R中央=%.4f M平均=%s(NA=%d) Δ平均=%s" % (
            arm, np.mean(Rv), np.median(Rv),
            "%.3f" % np.mean(Ms) if Ms else "全NA", na,
            "%.3f" % np.mean(ds) if ds else "NA"))
    b1 = [out["seeds"][str(s)]["B_E1"] for s in SEEDS7]
    print("B(E=1)平均: A由来生存 ep150=%.1f(昇格%.1f) / ep1000=%.1f(昇格%.1f)" % (
        np.mean([b["A_alive_150"] for b in b1]), np.mean([b["A_promoted_150"] for b in b1]),
        np.mean([b["A_alive_1000"] for b in b1]), np.mean([b["A_promoted_1000"] for b in b1])))
    print()
    print("=== 主要検定（凍結§5・2本のみ・判定は§5/§6基準のみ） ===")
    P1 = tests["P1"]
    print("[P1] R: 連鎖(T1000,E1) vs 直接(FirstHand) — 著者仮説の方向=連鎖が低い")
    print("     連鎖: %s / 直接: %s" % (iqr_str(tests["data"]["r_1000"]), iqr_str(tests["data"]["r_first"])))
    if P1 is None:
        print("     全ペア差分ゼロ: 検定不能")
    else:
        print("     Wilcoxon(pratt) W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.4f (参考t p=%.5f)" % (
            P1["W"], P1["p"], P1["p_holm"], P1["rank_biserial"], P1["mean_diff"], P1["t_p_ref"]))
    P2 = tests["P2"]
    print("[P2] R: 連鎖(T150) vs 連鎖(T1000) — 早い語りほど届くか")
    print("     T150: %s / T1000: %s" % (iqr_str(tests["data"]["r_150"]), iqr_str(tests["data"]["r_1000"])))
    if P2 is None:
        print("     全ペア差分ゼロ: 検定不能")
    else:
        print("     Wilcoxon(pratt) W=%.1f p=%.5f p_holm=%.5f rb=%.3f 平均差=%+.4f (参考t p=%.5f)" % (
            P2["W"], P2["p"], P2["p_holm"], P2["rank_biserial"], P2["mean_diff"], P2["t_p_ref"]))
    print()
    e = tests["expl_lie_coll"]
    print("--- 探索枠（検定しない・凍結§5）: 嘘E=3の連鎖CへのC衝突[101,1000]の影響 ---")
    if e is None:
        print("  差分全ゼロ")
    else:
        print("  E3: %s / E1: %s / p=%.4f rb=%.3f 平均差=%+.2f（方向と効果量の記述のみ）" % (
            iqr_str(tests["expl_lie_data"]["e3"]), iqr_str(tests["expl_lie_data"]["e1"]),
            e["p"], e["rank_biserial"], e["mean_diff"]))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if mode == "diag":
        diag7()
    else:
        out, t0 = full7()
        tests = analyze7(out)
        # JSONにはcoll_flagsを含む全生データを保存（タプルキーは文字列化不能のためarmsの数値のみ）
        slim = {"_tests": tests, "seeds": {}}
        for s, sd in out["seeds"].items():
            slim["seeds"][s] = {"nA": sd["nA"], "B_E1": sd["B_E1"], "B_E3": sd["B_E3"],
                "arms": {a: {kk: vv for kk, vv in r.items() if kk != "coll_pos"}
                         for a, r in sd["arms"].items()}}
        json.dump(slim, open(os.path.join(_BASE7, 'lie_prop_results_30seed.json'), 'w'),
                  ensure_ascii=False)
        print("JSON保存完了 (%.0f秒)" % (time.time() - t0))
        report7(out, tests)
        print()
        print("判定はDESIGN_嘘の伝播_v02_凍結.mdの事前登録基準のみで行うこと。")
