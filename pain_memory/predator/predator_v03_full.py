# -*- coding: utf-8 -*-
"""
捕食者実験 v03 本実行スクリプト（関係的な痛み版）
3方式(PermBan/NoMemory/FCA) × 認識あり/なし を比較。
診断ログ: 前半/後半の死亡回数、成功率、末尾の累積痛み。
PermBanは「死んだ行動のみ」永久禁止（接触=死は絶対的なので）。FCAは関係痛も死も
記憶に蓄積し減衰させる。
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predator_v03_env.py'), encoding='utf-8').read())

def argmax_rt(qv, rng):
    f = np.isfinite(qv)
    if not f.any(): return rng.randint(len(qv))
    mv = np.max(qv[f]); return rng.choice(np.where((qv==mv)&f)[0])

def run_condition(method, recognition, seed, n_episodes=N_EPISODES, diagnose=False):
    rng = np.random.RandomState(seed)
    env = PredatorEnvV3(recognition=recognition)
    Q = {}
    def get_Q(s):
        if s not in Q: Q[s] = np.zeros(N_ACTIONS)
        return Q[s]
    pain_memory = {}
    perm_ban = set()
    LAMBDA_GLOBAL = 0.97
    PAIN_THRESHOLD = 0.3  # FCAがタブー化する痛み蓄積の閾値

    def fca_key(state, next_pos):
        return (state, next_pos)
    def permban_key(state, next_pos):
        return (state[0], next_pos)

    def is_tabooed(state, ai):
        npx = move(state[0], ACTIONS[ai])
        if method == "PermBan":
            return permban_key(state, npx) in perm_ban
        elif method == "FCA":
            return pain_memory.get(fca_key(state, npx), 0.0) > PAIN_THRESHOLD
        return False

    rewards, deaths_pre, deaths_post = [], 0, 0
    success_last100, total_pain_last100 = [], []

    for ep in range(n_episodes):
        state = env.reset(ep)
        total_r, ep_death, ep_succ, ep_pain = 0.0, False, False, 0.0
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
            intended = move(s[0], ACTIONS[a])
            ns, r, done, info = env.step(a)
            total_r += r
            best = np.max(get_Q(ns)) if not done else 0.0
            get_Q(s)[a] += LR*(r + GAMMA*best - get_Q(s)[a])
            # 痛み(関係痛 or 死)を記憶に記録
            pain_signal = 0.0
            if info["collided"]:
                ep_death = True
                pain_signal = 2.0  # 死は強い痛み
            elif info["pain"] != 0.0:
                pain_signal = -info["pain"]  # 正値化
            ep_pain += (pain_signal if pain_signal>0 else 0)
            if pain_signal > 0:
                if method == "FCA":
                    k = fca_key(s, intended)
                    pain_memory[k] = pain_memory.get(k, 0.0) + pain_signal
                elif method == "PermBan":
                    if info["collided"]:  # PermBanは死んだ行動のみ永久禁止
                        perm_ban.add(permban_key(s, intended))
            if info["success"]: ep_succ = True
            state = ns
            if done: break
        if method == "FCA":
            for k in list(pain_memory.keys()):
                pain_memory[k] *= LAMBDA_GLOBAL
                if pain_memory[k] < 0.05: del pain_memory[k]
        rewards.append(total_r)
        if ep < SWITCH_EP and ep_death: deaths_pre += 1
        if ep >= SWITCH_EP and ep_death: deaths_post += 1
        if ep >= n_episodes-100:
            success_last100.append(1.0 if ep_succ else 0.0)
            total_pain_last100.append(ep_pain)

    rewards = np.array(rewards)
    out = {
        "rew_pre": float(np.mean(rewards[:SWITCH_EP])),
        "rew_post_first50": float(np.mean(rewards[SWITCH_EP:SWITCH_EP+50])),
        "rew_last100": float(np.mean(rewards[-100:])),
        "deaths_pre": int(deaths_pre),
        "deaths_post": int(deaths_post),
        "success_last100": float(np.mean(success_last100)) if success_last100 else 0.0,
        "avg_pain_last100": float(np.mean(total_pain_last100)) if total_pain_last100 else 0.0,
    }
    if diagnose: out["n_states"] = len(Q)
    return out

if __name__ == "__main__":
    conditions = [("PermBan",False),("NoMemory",False),("FCA",False),
                  ("PermBan",True),("NoMemory",True),("FCA",True)]
    N_SEEDS = 15
    results = {}
    for method, recog in conditions:
        label = method + ("+認識" if recog else "")
        keys = ["rew_pre","rew_post_first50","rew_last100","deaths_pre","deaths_post","success_last100","avg_pain_last100"]
        agg = {k: [] for k in keys}
        for seed in range(N_SEEDS):
            r = run_condition(method, recog, seed)
            for k in keys: agg[k].append(r[k])
        results[label] = {k: float(np.mean(agg[k])) for k in keys}
    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'predator_v03_results_15seed.json'),'w'), ensure_ascii=False, indent=1)
    print("=== 捕食者v03(関係的な痛み) 全6条件×15シード ===")
    print(f'{"条件":16s} {"前半報酬":>7s} {"切替直後":>7s} {"末尾報酬":>7s} {"前半死":>6s} {"後半死":>6s} {"成功率":>6s} {"末尾痛み":>7s}')
    for label, r in results.items():
        print(f'{label:16s} {r["rew_pre"]:7.2f} {r["rew_post_first50"]:7.2f} {r["rew_last100"]:7.2f} {r["deaths_pre"]:6.1f} {r["deaths_post"]:6.1f} {r["success_last100"]:6.2f} {r["avg_pain_last100"]:7.3f}')
