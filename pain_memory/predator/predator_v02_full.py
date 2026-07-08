# -*- coding: utf-8 -*-
"""
捕食者実験 v02 本実行スクリプト
3方式(PermBan/NoMemory/FCA) × 認識あり/なし を比較する。
診断ログ: 前半/後半の衝突回数、成功率、平均歩数、どのゲート(gap_mid最短 / gap_top迂回)を使ったか
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predator_v02_env.py'), encoding='utf-8').read())

def argmax_rt(qvals, rng):
    finite = np.isfinite(qvals)
    if not finite.any(): return rng.randint(len(qvals))
    mv = np.max(qvals[finite]); cand = np.where((qvals==mv)&finite)[0]
    return rng.choice(cand)

def run_condition(method, recognition, seed, n_episodes=N_EPISODES, diagnose=False):
    rng = np.random.RandomState(seed)
    env = PredatorEnvV2(recognition=recognition)
    Q = {}
    def get_Q(s):
        if s not in Q: Q[s] = np.zeros(N_ACTIONS)
        return Q[s]
    pain_memory = {}
    perm_ban = set()
    LAMBDA_GLOBAL = 0.97

    def fca_key(state, next_pos):
        # FCA: 状態(認識文脈含む)＋移動先
        return (state, next_pos)
    def permban_key(state, next_pos):
        # PermBan: エージェント位置＋移動先のみ。捕食者文脈は無視(=硬直)
        return (state[0], next_pos)

    def is_tabooed(state, action_idx):
        next_pos = move(state[0], ACTIONS[action_idx])
        if next_pos == state[0]:  # 壁・移動不可なら判定不要
            return False
        if method == "PermBan":
            return permban_key(state, next_pos) in perm_ban
        elif method == "FCA":
            return pain_memory.get(fca_key(state, next_pos), 0.0) > 0.3
        return False

    rewards, collisions_pre, collisions_post = [], 0, 0
    success_last100, steps_last100 = [], []
    gap_used_last100 = {"mid": 0, "top": 0}

    for ep in range(n_episodes):
        state = env.reset(ep)
        total_r, ep_coll, ep_succ, nsteps = 0.0, False, False, 0
        crossed_at = None
        for t in range(MAX_STEPS):
            s = state
            qv = get_Q(s).copy()
            tab = [i for i in range(N_ACTIONS) if is_tabooed(s, i)]
            if len(tab) < N_ACTIONS:
                for i in tab: qv[i] = -np.inf
            if rng.rand() < EPSILON:
                ch = [i for i in range(N_ACTIONS) if i not in tab] or list(range(N_ACTIONS))
                a = rng.choice(ch)
            else:
                a = argmax_rt(qv, rng)
            ns, r, done, info = env.step(a)
            total_r += r; nsteps += 1
            best = np.max(get_Q(ns)) if not done else 0.0
            get_Q(s)[a] += LR*(r + GAMMA*best - get_Q(s)[a])
            if env.pos == (3,3): crossed_at = "mid"
            elif env.pos == (3,6): crossed_at = "top"
            if info["collided"]:
                ep_coll = True
                intended = move(s[0], ACTIONS[a])  # 移動しようとした先(=衝突した危険マス)
                if method == "FCA":
                    k = fca_key(s, intended)
                    pain_memory[k] = pain_memory.get(k, 0.0) + 1.0
                elif method == "PermBan":
                    perm_ban.add(permban_key(s, intended))
            if info["success"]: ep_succ = True
            state = ns
            if done: break
        if method == "FCA":
            for k in list(pain_memory.keys()):
                pain_memory[k] *= LAMBDA_GLOBAL
                if pain_memory[k] < 0.05: del pain_memory[k]
        rewards.append(total_r)
        if ep < SWITCH_EP and ep_coll: collisions_pre += 1
        if ep >= SWITCH_EP and ep_coll: collisions_post += 1
        if ep >= n_episodes - 100:
            success_last100.append(1.0 if ep_succ else 0.0)
            if ep_succ: steps_last100.append(nsteps)
            if crossed_at: gap_used_last100[crossed_at] += 1

    rewards = np.array(rewards)
    out = {
        "rew_pre": float(np.mean(rewards[:SWITCH_EP])),
        "rew_post_first50": float(np.mean(rewards[SWITCH_EP:SWITCH_EP+50])),
        "rew_last100": float(np.mean(rewards[-100:])),
        "collisions_pre": int(collisions_pre),
        "collisions_post": int(collisions_post),
        "success_last100": float(np.mean(success_last100)) if success_last100 else 0.0,
        "avg_steps_last100": float(np.mean(steps_last100)) if steps_last100 else -1,
        "gap_mid_last100": gap_used_last100["mid"],
        "gap_top_last100": gap_used_last100["top"],
    }
    if diagnose:
        out["n_states"] = len(Q)
    return out

if __name__ == "__main__":
    conditions = [("PermBan",False),("NoMemory",False),("FCA",False),
                  ("PermBan",True),("NoMemory",True),("FCA",True)]
    N_SEEDS = 15
    results = {}
    for method, recog in conditions:
        label = method + ("+認識" if recog else "")
        keys = ["rew_pre","rew_post_first50","rew_last100","collisions_pre","collisions_post","success_last100","avg_steps_last100","gap_mid_last100","gap_top_last100"]
        agg = {k: [] for k in keys}
        for seed in range(N_SEEDS):
            r = run_condition(method, recog, seed)
            for k in keys: agg[k].append(r[k])
        results[label] = {k: float(np.mean([v for v in agg[k] if v!=-1] or [-1])) for k in keys}
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'predator_v02_results_15seed.json'),'w'), ensure_ascii=False, indent=1)
    print("=== 捕食者v02 全6条件×15シード ===")
    print(f'{"条件":16s} {"前半報酬":>7s} {"切替直後":>7s} {"末尾報酬":>7s} {"前半衝突":>7s} {"後半衝突":>7s} {"成功率":>6s} {"mid":>5s} {"top":>5s}')
    for label, r in results.items():
        print(f'{label:16s} {r["rew_pre"]:7.2f} {r["rew_post_first50"]:7.2f} {r["rew_last100"]:7.2f} {r["collisions_pre"]:7.1f} {r["collisions_post"]:7.1f} {r["success_last100"]:6.2f} {r["gap_mid_last100"]:5.0f} {r["gap_top_last100"]:5.0f}')
