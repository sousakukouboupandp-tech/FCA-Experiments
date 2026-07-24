# -*- coding: utf-8 -*-
"""
再構成実験 v01 実装（身体・急所・死・幻肢痛つき自己同一性測定）
事前登録: 再構成実験_設計書v08（凍結直前版・監査4AI条件充足済み）
問い: 身体を持ち死ねる個体において、自己はどの記録成分（Q表/傷跡帳/Tabu）に宿るか
条件: CONT / FULL / Q-only / SCAR-only / NONE（+探索枠 SCAR-rand）
主要指標: N_hist（チャンネルL=喪失接近 + チャンネルP=幻肢痛、部位上限1.0対称）
対抗指標: D_behav（共通生存期間の訪問分布JS距離）
仮説: H1: N_hist SCAR-only > Q-only / H2: D_behav Q-only < SCAR-only

【実装時修正記録・2026年7月24日】設計書§2.2「各危険マスに部位を固定割当
（6部位均等）」はD0世界の危険セルが3つのため実装不可能と実装時に判明。
修正: シードから生成した6部位均等巡回列（6部位×N回シャッフル）を環境が保持し、
衝突イベントごとに列が進む方式に変更。頻度は完全均等・決定論的・乱数消費ゼロ・
シード内全条件で同一列。凍結版v09に反映する。

使い方: python recon_v01.py ut / prelim / full
"""
import sys, json, os, time, copy
import numpy as np
from scipy import stats as spstats
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
BASE = os.path.dirname(os.path.abspath(__file__))
RESO = os.path.join(os.path.dirname(BASE), 'resonance')
sys.path.insert(0, RESO)
exec(open(os.path.join(RESO, 'resonance_v01_env.py'), encoding='utf-8').read())
from resonance_v07_zerolr import (sim_entry, argmax_rt,
    paired_wilcoxon_pratt, holm_2, iqr_str)

# ============ 身体系定数（付録B候補・予備実験で較正して凍結） ============
PARTS = ["brain", "neck", "heart", "organs", "arm", "leg"]
VITALS = {"brain", "neck", "heart", "organs"}   # 急所: 到達=個体死
LIMBS = {"arm", "leg"}                            # 末端: 到達=部位破壊
LIMIT = {"brain": 25.0, "neck": 25.0, "heart": 25.0, "organs": 25.0,
         "arm": 12.0, "leg": 12.0}  # 較正2回目2026-07-24: マス固定割当下で死亡率30%
ARM_STEP_COST_MULT = 2.0    # 腕破壊: STEP_COST倍率（行動精度ペナルティの実装）
LEG_MAX_STEPS_MULT = 0.75   # 脚破壊: MAX_STEPS縮小（移動ペナルティの実装）
HIT_INTENSITY = 1.0         # 1衝突のダメージ強度（既存record_painと同一）

T_TOTAL = 1000
T_SPLIT = 500
SEEDS_FULLRUN = list(range(100, 130))   # 本番30シード（v09で保険数を確定）
SEEDS_PRELIM = list(range(200, 210))    # 予備実験10シード（本番と非重複）
EXCLUDE_FRAC = 0.20   # 共通後半生が後半生長の20%未満のペアは主要検定から除外

LAMBDA_OWN = 0.97
LAMBDA_VITAL = 0.97     # 裁定⑧較正2026-07-24（3回目・確定）: greedy衝突の実測で
                        # 「代替経路の確立」は1回の衝突で完了（同一キーでの2回目の
                        # greedy衝突はゼロ）。現行λ=0.97は115ep保持し要件を大幅に
                        # 満たすため、急所の延長は不要と実測が示した。機構は残す。
TABOO_TH = 0.3
PM_DEL_TH = 0.05

def make_part_seq(seed, n_rep=40):
    """【実装時修正記録2・2026年7月24日】巡回列方式は衝突順序依存の部位ノイズを生み
    予備実験でG1不成立（FULL一致0.34）の原因となった。設計原義「マス固定割当」へ復帰:
    衝突可能位置（パトロールセル）→部位の固定写像。同じ場所の傷は必ず同じ部位。
    シードごとに6部位から3部位（急所≥1・末端≥1）を選ぶ。決定論・乱数消費ゼロ。"""
    g = np.random.RandomState(seed * 7919 + 13)
    cells = sorted(set(PATROL_B_D0))
    vit = sorted(VITALS); lim = sorted(LIMBS)
    picks = [vit[g.randint(len(vit))], lim[g.randint(len(lim))]]
    rest = [p for p in PARTS if p not in picks]
    picks.append(rest[g.randint(len(rest))])
    g.shuffle(picks)
    return {cells[i]: picks[i % len(picks)] for i in range(len(cells))}

class Body:
    """6部位の独立ステータス管理（グラフ力学なし・設計書§2.1）。
    ダメージは部位独立。急所いずれか到達=即死。末端到達=部位破壊+機能ペナルティ永続。"""
    def __init__(self):
        self.dmg = {p: 0.0 for p in PARTS}
        self.broken = set()
        self.dead = False
        self.death_part = None
    def apply_hit(self, part, intensity=HIT_INTENSITY):
        """ダメージ適用。返り値: チャンネルL算入分（限界までの残量按分・付録C規則）。"""
        if self.dead:
            return 0.0
        pre = self.dmg[part]
        lim = LIMIT[part]
        contrib = 0.0
        if pre < lim:
            contrib = min(intensity, lim - pre)   # 到達イベントは残量のみ算入
        self.dmg[part] = pre + intensity
        if self.dmg[part] >= lim:
            if part in VITALS:
                self.dead = True
                self.death_part = part
            else:
                self.broken.add(part)
        return contrib
    def step_cost(self):
        return STEP_COST * (ARM_STEP_COST_MULT if "arm" in self.broken else 1.0)
    def max_steps(self):
        return int(MAX_STEPS * (LEG_MAX_STEPS_MULT if "leg" in self.broken else 1.0))

class ReconAgent:
    """Q表・傷跡帳(pm)・Tabu(pm経由)・身体を持つ個体。resonance系Agentを身体つきに拡張。
    pm: {(state, intended): [強度, is_own, S]} 既存仕様と同一。"""
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        self.Q = {}
        self.pm = {}
        self.archive = {}
        self.body = Body()
        self.part_map = {}   # run_lifeが自動設定（裁定⑧の部位別減衰に使用）
    def get_Q(self, s):
        if s not in self.Q: self.Q[s] = np.zeros(N_ACTIONS)
        return self.Q[s]
    def is_tabooed(self, state, ai):
        nxt = move(state[0], ACTIONS[ai])
        if nxt == state[0]: return False
        cand = (state, nxt)
        v = self.pm.get(cand)
        if v is not None and v[1] and v[0] > TABOO_TH:
            return True
        for k, v in self.pm.items():
            if v[1]: continue
            if v[0] * sim_entry(cand, k) > TABOO_TH:
                return True
        return False
    def record_pain(self, state, intended):
        k = (state, intended)
        if k in self.pm:
            self.pm[k][0] += 1.0; self.pm[k][1] = True; self.pm[k][2] = 1.0
        else:
            self.pm[k] = [1.0, True, 1.0]
        self.archive[k] = max(self.archive.get(k, 0.0), self.pm[k][0])
        return k
    def decay(self):
        """裁定⑧: 記憶の寿命は傷の深さと場所で決まる。急所の記憶は代替経路が
        確立するまで（実測934ep）を跨ぐようλを機械導出。末端は従来通り薄れる。
        部位はpmキー(state, intended)のintended=衝突セルからpart_mapで一意に決まる。"""
        for k in list(self.pm.keys()):
            part = self.part_map.get(k[1])
            lam = LAMBDA_VITAL if part in VITALS else LAMBDA_OWN
            self.pm[k][0] *= lam
            if self.pm[k][0] < PM_DEL_TH: del self.pm[k]

def run_life(agent, env, part_seq, part_idx, ep_start, ep_end, events, visits):
    """人生区間 [ep_start, ep_end) を走らせる。
    events: 被ダメージイベントログ (ep, part, path_key, intensity, L_contrib) を追記
    visits: 訪問pos頻度 {pos: count} を追記（生きたエピソードのみ）
    返り値: (最終part_idx, 死亡ep or None, 衝突ep数, 成功ep数)
    死亡時はその場で打ち切り（死は本物・記録終了）。"""
    coll_eps, succ_eps = 0, 0
    death_ep = None
    agent.part_map = part_seq   # 裁定⑧: 部位別減衰のため写像を保持
    for ep in range(ep_start, ep_end):
        if agent.body.dead:
            death_ep = death_ep if death_ep is not None else ep
            break
        state = env.reset(ep)
        ep_coll, ep_succ = False, False
        eff_max = agent.body.max_steps()
        ev_vis = visits.setdefault(ep, {})
        for t in range(eff_max):
            s = state
            ev_vis[s[0]] = ev_vis.get(s[0], 0) + 1
            qv = agent.get_Q(s).copy()
            tab = [i for i in range(N_ACTIONS) if agent.is_tabooed(s, i)]
            if len(tab) < N_ACTIONS:
                for i in tab: qv[i] = -np.inf
            if agent.rng.rand() < EPSILON:
                ch = [i for i in range(N_ACTIONS) if i not in tab] or list(range(N_ACTIONS))
                a = agent.rng.choice(ch)
                explored = True
            else:
                a = argmax_rt(qv, agent.rng)
                explored = False
            ns, r, done, info = env.step(a)
            r = r - STEP_COST + agent.body.step_cost()   # 腕ペナルティ反映
            best = np.max(agent.get_Q(ns)) if not done else 0.0
            agent.get_Q(s)[a] += LR*(r + GAMMA*best - agent.get_Q(s)[a])
            if info["collided"]:
                ep_coll = True
                intended = move(s[0], ACTIONS[a])
                pk = agent.record_pain(s, intended)
                part = part_seq[info["collided_at"]]   # マス固定割当（修正記録2）
                lc = agent.body.apply_hit(part, HIT_INTENSITY)
                events.append({"ep": ep, "part": part, "pk": repr(pk),
                               "intensity": HIT_INTENSITY, "L_contrib": lc,
                               "explored": explored,
                               "broken_at_hit": part in agent.body.broken and lc == 0.0})
                if agent.body.dead:
                    break
            if info["success"]: ep_succ = True
            state = ns
            if done: break
        agent.decay()
        if ep_coll: coll_eps += 1
        if ep_succ: succ_eps += 1
        if agent.body.dead and death_ep is None:
            death_ep = ep + 1   # このepまで生きた（死亡epは記録に含む）
    return part_idx, death_ep, coll_eps, succ_eps

def snapshot(agent, part_idx):
    """T_split時点の完全記録（付録A: pkはrepr文字列で一意化した(state,intended)経路キー）。"""
    return {"Q": copy.deepcopy(agent.Q),
            "pm": copy.deepcopy(agent.pm),
            "body": copy.deepcopy(agent.body),
            "part_idx": part_idx}

def reconstruct(cond, snap, new_seed):
    """成分選択的継承（設計書§3）。身体は全条件継承。CONTはこの関数を通らない。"""
    ag = ReconAgent(new_seed)
    ag.body = copy.deepcopy(snap["body"])
    if cond in ("FULL",):
        ag.Q = copy.deepcopy(snap["Q"])
        ag.pm = copy.deepcopy(snap["pm"])
    elif cond == "Q-only":
        ag.Q = copy.deepcopy(snap["Q"])
        # pm空 = 傷跡帳もTabuも喪失
    elif cond in ("SCAR-only", "SCAR-rand"):
        ag.pm = copy.deepcopy(snap["pm"])
        if cond == "SCAR-rand":   # 探索枠: 乱数初期化Q（本番は白紙）
            g = np.random.RandomState(new_seed + 77)
            # 既知状態にのみ小乱数を播く（未知状態はget_Qでゼロ生成）
            for s in snap["Q"]:
                ag.Q[s] = g.uniform(-0.01, 0.01, N_ACTIONS)
    elif cond == "NONE":
        pass
    return ag

def narrative_weights(events, pm_final, common_end):
    """語りの重みベクトル構築（設計書§4.1）。
    キー: (part, pk, channel)。共通生存期間 [T_SPLIT, common_end) のイベントのみ。
    L: 生存部位への傷=L_contrib/limit（按分済・部位Σ<=1は按分則が保証）
    P: 破壊済み末端への受傷=比較時点のpm残存強度/limit、部位ごと比例圧縮でΣ<=1"""
    wL, p_keys = {}, {}
    for e in events:
        if e["ep"] >= common_end: continue
        part, pk = e["part"], e["pk"]
        if e["L_contrib"] > 0:
            k = (part, pk, "L")
            wL[k] = wL.get(k, 0.0) + e["L_contrib"] / LIMIT[part]
        elif part in LIMBS:   # 破壊済み末端への受傷 → 幻肢痛候補
            p_keys.setdefault(part, set()).add(pk)
    wP = {}
    for part, pks in p_keys.items():
        raw = {}
        for pk in pks:
            # 比較時点(共通期間末)のpm残存強度。pmキーはevalで復元せずrepr照合。
            inten = 0.0
            for mk, mv in pm_final.items():
                if repr(mk) == pk: inten = mv[0]; break
            if inten > 0: raw[pk] = inten / LIMIT[part]
        tot = sum(raw.values())
        scale = min(1.0, 1.0/tot) if tot > 0 else 1.0   # 部位上限1.0比例圧縮
        for pk, v in raw.items():
            wP[(part, pk, "P")] = v * scale
    w = dict(wL); w.update(wP)
    return w, wL, wP

def ruzicka(wa, wb):
    """Weighted Jaccard（Ruzicka類似度）。両者空なら定義不能としてNone。"""
    keys = set(wa) | set(wb)
    if not keys: return None
    num = sum(min(wa.get(k, 0.0), wb.get(k, 0.0)) for k in keys)
    den = sum(max(wa.get(k, 0.0), wb.get(k, 0.0)) for k in keys)
    return num/den if den > 0 else None

def js_distance(va, vb):
    """訪問pos分布のJensen-Shannon距離（底2・0〜1）。"""
    keys = sorted(set(va) | set(vb))
    if not keys: return None
    pa = np.array([va.get(k, 0) for k in keys], dtype=float)
    pb = np.array([vb.get(k, 0) for k in keys], dtype=float)
    if pa.sum() == 0 or pb.sum() == 0: return None
    pa /= pa.sum(); pb /= pb.sum()
    m = 0.5*(pa+pb)
    def kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask]*np.log2(p[mask]/q[mask])))
    return float(np.sqrt(0.5*kl(pa, m) + 0.5*kl(pb, m)))

def sum_w(w):
    return float(sum(w.values()))

CONDS_MAIN = ["FULL", "Q-only", "SCAR-only", "NONE"]

def run_seed(seed, include_rand=False):
    """1シードの全条件実行。前半生→snapshot→CONT継続＋各条件後半生→指標。"""
    # --- 前半生（共通の生い立ち） ---
    A = ReconAgent(seed)
    env = ResonanceEnv(PATROL_B_D0)
    part_seq = make_part_seq(seed)
    evA, visA = [], {}
    pidx, dA, cA, sA = run_life(A, env, part_seq, 0, 0, T_SPLIT, evA, visA)
    if A.body.dead:
        return {"seed": seed, "skip": "died_in_first_half", "death_ep": dA}
    snap = snapshot(A, pidx)
    # --- CONT: 乱数列も連続のまま後半生 ---
    evC, visC = list(evA), {}
    evC_post = []
    pidxC, dC, cC, sC = run_life(A, env, part_seq, pidx, T_SPLIT, T_TOTAL, evC_post, visC)
    contD = dC if dC is not None else T_TOTAL
    # --- 各再構成条件（新乱数=seed+5000で全条件共通） ---
    conds = CONDS_MAIN + (["SCAR-rand"] if include_rand else [])
    out = {"seed": seed, "skip": None, "cont_death": contD,
           "cont_events_post": evC_post, "results": {}}
    for cond in conds:
        B = reconstruct(cond, snap, seed + 5000)
        envB = ResonanceEnv(PATROL_B_D0)
        evB, visB = [], {}
        _, dB, cB, sB = run_life(B, envB, part_seq, snap["part_idx"],
                                 T_SPLIT, T_TOTAL, evB, visB)
        lifeB = dB if dB is not None else T_TOTAL
        common_end = min(contD, lifeB)
        common_len = common_end - T_SPLIT
        valid = common_len >= EXCLUDE_FRAC * (T_TOTAL - T_SPLIT)
        wB, wBL, wBP = narrative_weights(evB, B.pm, common_end)
        wC, wCL, wCP = narrative_weights(evC_post, A.pm, common_end)
        def agg(vis):
            d = {}
            for ep, m in vis.items():
                if ep >= common_end: continue
                for pos, c in m.items(): d[pos] = d.get(pos, 0) + c
            return d
        visBc, visCc = agg(visB), agg(visC)
        out["results"][cond] = {
            "life": lifeB, "death_part": B.body.death_part,
            "common_len": common_len, "valid_pair": bool(valid),
            "N_hist": ruzicka(wB, wC),
            "N_hist_L": ruzicka(wBL, wCL),
            "N_hist_P": ruzicka(wBP, wCP),
            "D_behav": js_distance(visBc, visCc),
            "sum_w": sum_w(wB), "sum_wC": sum_w(wC),
            "sum_wL": sum_w(wBL), "sum_wP": sum_w(wBP),
            "n_events": len([e for e in evB if e["ep"] < common_end]),
            "coll_eps": cB, "succ_eps": sB,
            "broken": sorted(B.body.broken),
            "phantom_hits": len([e for e in evB
                if e["ep"] < common_end and e["L_contrib"] == 0.0
                and e["part"] in LIMBS]),
        }
    out["cont_death_part"] = A.body.death_part
    out["cont_broken"] = sorted(A.body.broken)
    return out

# 実装ノート（付録候補）: チャンネルPの残存強度は「各個体の記録終了時点のpm」を使用。
# 共通期間末との厳密一致は減衰の決定論により巻き戻し可能だが削除済みキーは復元不能
# のため近似とし、予備実験(h)で影響を確認する。

def ut():
    print("=== UT-R1: 身体系（蓄積・按分・破壊・死・キャップ） ===")
    b = Body()
    la = LIMIT["arm"]
    assert b.apply_hit("arm", la - 1.0) == la - 1.0 and "arm" not in b.broken
    assert b.apply_hit("arm", 3.0) == 1.0 and "arm" in b.broken  # 残量1.0のみ算入
    assert b.apply_hit("arm", 5.0) == 0.0                        # 破壊後は算入ゼロ
    assert not b.dead
    n_h = int(LIMIT["heart"] / HIT_INTENSITY)
    for _ in range(n_h): b.apply_hit("heart", HIT_INTENSITY)
    assert b.dead and b.death_part == "heart"
    assert b.apply_hit("leg", 1.0) == 0.0    # 死後は全て無効
    print("合格")
    print("=== UT-R2: ペナルティ（腕=コスト倍・脚=歩数減・乱数非消費） ===")
    b2 = Body()
    assert b2.max_steps() == MAX_STEPS and abs(b2.step_cost() - STEP_COST) < 1e-12
    b2.broken.add("leg"); assert b2.max_steps() == int(MAX_STEPS*LEG_MAX_STEPS_MULT)
    b2.broken.add("arm"); assert abs(b2.step_cost() - STEP_COST*2.0) < 1e-12
    print("合格")
    print("=== UT-R3: 部位写像（マス固定・急所末端保証・決定論） ===")
    s1, s2 = make_part_seq(100), make_part_seq(100)
    assert s1 == s2, "決定論性の破れ"
    for sd in range(100, 140):
        m = make_part_seq(sd)
        assert len(m) == len(set(PATROL_B_D0)), "セル網羅の破れ"
        ps = set(m.values())
        assert ps & VITALS and ps & LIMBS, "急所/末端の最低保証の破れ seed%d" % sd
    print("合格: 全セル固定割当・急所≥1・末端≥1（40シード検証）")
    print("=== UT-R4: 数学的同一の較正（FULL+同一乱数=CONT・設計§1.1の確認） ===")
    A = ReconAgent(300); env = ResonanceEnv(PATROL_B_D0)
    seq = make_part_seq(300); ev, vis = [], {}
    pidx, d, c, s = run_life(A, env, seq, 0, 0, 100, ev, vis)
    if not A.body.dead:
        snap = snapshot(A, pidx)
        # CONT側: 乱数状態を保存して継続
        rng_state = A.rng.get_state()
        evC, visC = [], {}
        run_life(A, env, seq, pidx, 100, 200, evC, visC)
        # FULL+同一乱数: スナップショットから再構成し、乱数状態も複製
        Bf = reconstruct("FULL", snap, 999)
        Bf.rng.set_state(rng_state)
        envF = ResonanceEnv(PATROL_B_D0)
        evF, visF = [], {}
        run_life(Bf, envF, seq, snap["part_idx"], 100, 200, evF, visF)
        same_ev = [ (e["ep"],e["part"],e["pk"]) for e in evC ] == \
                  [ (e["ep"],e["part"],e["pk"]) for e in evF ]
        assert same_ev, "!!! 完全継承+同一乱数がCONTと分岐 = 実装に隠れ状態あり"
        print("合格: イベント列完全一致（%d件）" % len(evC))
    else:
        print("スキップ（seed300が前半死・別シードで要再確認）")
    print("=== UT-R5: Ruzicka/JSの基本性質 ===")
    assert ruzicka({"a":1.0},{"a":1.0}) == 1.0
    assert ruzicka({"a":1.0},{"b":1.0}) == 0.0
    assert ruzicka({},{}) is None
    d0 = js_distance({(0,0):10},{(0,0):10}); assert d0 is not None and d0 < 1e-9
    print("合格")
    print("=== 全UT合格 ===")

def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None

def prelim():
    """予備実験10シード（設計§7-2・較正であり検定ではない）。"""
    t0 = time.time()
    rows = []
    for seed in SEEDS_PRELIM:
        r = run_seed(seed, include_rand=True)
        rows.append(r)
        tag = "前半死" if r.get("skip") else ("CONT死%d" % r["cont_death"])
        print("seed%d 完了 (%s, %.0fs)" % (seed, tag, time.time()-t0))
    with open(os.path.join(BASE, "recon_prelim_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    ok = [r for r in rows if not r.get("skip")]
    print("\n=== 予備実験サマリー（%d/%dシード有効） ===" % (len(ok), len(rows)))
    cont_d = [r["cont_death"] for r in ok]
    print("(e)死亡率: CONT死亡 %d/%d, 死亡ep中央値 %s" % (
        sum(1 for d in cont_d if d < T_TOTAL), len(ok), med(cont_d)))
    for cond in CONDS_MAIN + ["SCAR-rand"]:
        rs = [r["results"][cond] for r in ok if cond in r["results"]]
        if not rs: continue
        print("%-10s N_hist=%s L=%s P=%s D_behav=%s 有効ペア%d/%d 死%d phantom中央%s" % (
            cond, med([x["N_hist"] for x in rs]), med([x["N_hist_L"] for x in rs]),
            med([x["N_hist_P"] for x in rs]), med([x["D_behav"] for x in rs]),
            sum(1 for x in rs if x["valid_pair"]), len(rs),
            sum(1 for x in rs if x["life"] < T_TOTAL),
            med([x["phantom_hits"] for x in rs])))
    # (d) G1予行: FULL vs NONE + NONE分布95パーセンタイル
    fu = [r["results"]["FULL"]["N_hist"] for r in ok
          if r["results"]["FULL"]["N_hist"] is not None]
    no = [r["results"]["NONE"]["N_hist"] for r in ok
          if r["results"]["NONE"]["N_hist"] is not None]
    if fu and no:
        p95 = float(np.percentile(no, 95))
        print("(d)G1予行: FULL中央値=%.4f NONE中央値=%.4f NONE95pct=%.4f → FULL>95pct: %s"
              % (np.median(fu), np.median(no), p95, np.median(fu) > p95))
    # (b) デッドロック確認（新規座標訪問数 CONT比30%以上）— 集約訪問から
    print("(b)(f)(g)の詳細はJSONを参照。分解能: N_hist分布の要約を出力済み")
    print("経過 %.0f 秒" % (time.time()-t0))

def full():
    """本番実行（凍結版v09の確定後にのみ実行すること）。"""
    t0 = time.time()
    rows = []
    for seed in SEEDS_FULLRUN:
        r = run_seed(seed, include_rand=False)
        rows.append(r)
        print("seed%d 完了 (%.0fs)" % (seed, time.time()-t0))
    with open(os.path.join(BASE, "recon_full_results_30seed.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    ok = [r for r in rows if not r.get("skip")]
    # H1: N_hist SCAR-only > Q-only / H2: D_behav Q-only < SCAR-only（対応あり・有効ペアのみ）
    pairs_h1, pairs_h2 = [], []
    for r in ok:
        sc, qo = r["results"]["SCAR-only"], r["results"]["Q-only"]
        if sc["valid_pair"] and qo["valid_pair"] and None not in (
                sc["N_hist"], qo["N_hist"], sc["D_behav"], qo["D_behav"]):
            pairs_h1.append((sc["N_hist"], qo["N_hist"]))
            pairs_h2.append((qo["D_behav"], sc["D_behav"]))
    print("\n=== 本番結果（有効ペア %d） ===" % len(pairs_h1))
    if len(pairs_h1) < 20:
        print("有効ペアN<20 → 事前登録フォールバック発動: 検定せず記述統計のみ")
    else:
        a1 = [x for x, _ in pairs_h1]; b1 = [y for _, y in pairs_h1]
        a2 = [x for x, _ in pairs_h2]; b2 = [y for _, y in pairs_h2]
        r1 = paired_wilcoxon_pratt(a1, b1)   # H1: SCAR > Q-only（N_hist）
        r2 = paired_wilcoxon_pratt(a2, b2)   # H2: Q-only < SCAR（D_behav: Q側が小）
        ph = holm_2(r1["p"], r2["p"])
        print("H1(語りの座) SCAR中央%.4f vs Qonly中央%.4f p=%.5f holm=%.5f rb=%.3f" % (
            np.median(a1), np.median(b1), r1["p"], ph[0], r1["rb"]))
        print("H2(行動の座) Qonly中央%.4f vs SCAR中央%.4f p=%.5f holm=%.5f rb=%.3f" % (
            np.median(a2), np.median(b2), r2["p"], ph[1], r2["rb"]))
    # 分離分析（記述・事前登録済み）
    for tag in ["N_hist_L", "N_hist_P"]:
        sc = med([r["results"]["SCAR-only"][tag] for r in ok])
        qo = med([r["results"]["Q-only"][tag] for r in ok])
        print("分離分析 %s: SCAR=%s Q-only=%s" % (tag, sc, qo))
    print("経過 %.0f 秒" % (time.time()-t0))

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ut"
    if mode == "ut": ut()
    elif mode == "prelim": prelim()
    elif mode == "full": full()
    else: print("usage: python recon_v01.py ut|prelim|full")
