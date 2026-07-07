# -*- coding: utf-8 -*-
"""
毒シミュレーション v2 本実行:
- 痛み: 薬物動態モデル(PK波形)
- P_local: 単純合計方式(決着済み)
- タブー判定: PermBan(粗いキー) / NoMemory / FCA-固定閾値 / FCA-zscore
  → FCA-固定閾値とFCA-zscoreは完全に独立したmemory/Q/visit_countを持つ
- 全条件 × 認識あり/なし × 20シード
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'poison_v2_env.py'), encoding='utf-8').read())

def run_condition(method, recognition, seed, n_episodes=N_EPISODES):
    rng = np.random.RandomState(seed)
    env = PoisonEnvV2(recognition=recognition)
    Q = {}
    def get_Q(s):
        if s not in Q: Q[s] = np.zeros(N_ACTIONS)
        return Q[s]

    def argmax_rt(qvals):
        finite_mask = np.isfinite(qvals)
        if not finite_mask.any():
            return rng.randint(len(qvals))
        max_val = np.max(qvals[finite_mask])
        candidates = np.where((qvals == max_val) & finite_mask)[0]
        return rng.choice(candidates)

    pain_memory = {}      # FCA-固定閾値専用 (完全独立)
    pain_memory_z = {}    # FCA-zscore専用 (完全独立): key -> list of fold_values (履歴)
    perm_ban = set()
    visit_count_z = {}    # zscore専用の訪問回数(独立)

    def fca_key(state): return state
    def permban_key(state): return (state[0][0], state[0][1])  # 認識を無視した粗いキー

    TAU_VISIT = 8.0  # 低訪問補正の緩やかさ

    def is_tabooed(state, action_idx):
        a = ACTIONS[action_idx]
        if a != "ACT": return False
        if state[0][0] != POISON_CELL: return False
        if method == "PermBan":
            return permban_key(state) in perm_ban
        elif method == "FCA_fixed":
            return pain_memory.get(fca_key(state), 0.0) < -3.0  # 固定閾値: 蓄積痛みが-3を下回ったらタブー
        elif method == "FCA_zscore":
            key = fca_key(state)
            hist = pain_memory_z.get(key, [])
            n = visit_count_z.get(key, 0)
            if n < 2 or len(hist) < 2:
                return False  # サンプル不足時は判定しない(タブーしない側にフォールバック)
            mu = np.mean(hist)
            sigma = np.sqrt(np.var(hist) + 1e-3)
            z_raw = (hist[-1] - mu) / sigma
            z_eff = z_raw * (1 - np.exp(-n / TAU_VISIT))
            return z_eff < -1.5  # 直近の痛みが自身の履歴平均より著しく悪い場合にタブー
        return False  # NoMemory

    rewards_per_ep = []
    poisoned_hits_pre = 0
    safe_flags_last100 = []

    for ep in range(n_episodes):
        state = env.reset(ep)
        total_r = 0.0
        p_local = 0.0
        episode_poisoned_key = None
        ep_safe = False
        ep_poisoned = False

        for t in range(MAX_STEPS):
            s = state
            qvals = get_Q(s).copy()
            tabooed = [i for i in range(N_ACTIONS) if is_tabooed(s, i)]
            if len(tabooed) < N_ACTIONS:
                for i in tabooed: qvals[i] = -np.inf
            if rng.rand() < EPSILON:
                choices = [i for i in range(N_ACTIONS) if i not in tabooed] or list(range(N_ACTIONS))
                a_idx = rng.choice(choices)
            else:
                a_idx = argmax_rt(qvals)

            next_state, r, done, info = env.step(a_idx)
            total_r += r
            best_next = np.max(get_Q(next_state)) if not done else 0.0
            get_Q(s)[a_idx] += LR * (r + GAMMA*best_next - get_Q(s)[a_idx])

            if info["pain_now"] != 0.0:
                p_local += info["pain_now"]  # A_sum方式(決着済み)
            if info["poisoned_hit"]:
                episode_poisoned_key = s
                ep_poisoned = True
            if info["success"] and not env.has_antidote and not env.poison_active():
                ep_safe = True

            state = next_state
            if done: break

        if episode_poisoned_key is not None:
            if method == "FCA_fixed":
                key = fca_key(episode_poisoned_key)
                pain_memory[key] = LAMBDA_GLOBAL * pain_memory.get(key, 0.0) + p_local
            elif method == "FCA_zscore":
                key = fca_key(episode_poisoned_key)
                pain_memory_z.setdefault(key, []).append(p_local)
                visit_count_z[key] = visit_count_z.get(key, 0) + 1
                if len(pain_memory_z[key]) > 50:
                    pain_memory_z[key] = pain_memory_z[key][-50:]
            elif method == "PermBan":
                perm_ban.add(permban_key(episode_poisoned_key))

        if method == "FCA_fixed":
            for k in list(pain_memory.keys()):
                pain_memory[k] *= LAMBDA_GLOBAL
                if abs(pain_memory[k]) < P_LIMIT:
                    del pain_memory[k]

        rewards_per_ep.append(total_r)
        if ep < SWITCH_EP and ep_poisoned:
            poisoned_hits_pre += 1
        if ep >= n_episodes - 100:
            safe_flags_last100.append(1.0 if ep_safe else 0.0)

    rewards_per_ep = np.array(rewards_per_ep)
    return {
        "rew_pre": float(np.mean(rewards_per_ep[:SWITCH_EP])),
        "rew_post": float(np.mean(rewards_per_ep[SWITCH_EP:])),
        "rew_post_first50": float(np.mean(rewards_per_ep[SWITCH_EP:SWITCH_EP+50])),
        "rew_last100": float(np.mean(rewards_per_ep[-100:])),
        "poisoned_hits_pre": int(poisoned_hits_pre),
        "safe_last100": float(np.mean(safe_flags_last100)) if safe_flags_last100 else 0.0,
    }

conditions = [
    ("PermBan", False), ("NoMemory", False), ("FCA_fixed", False), ("FCA_zscore", False),
    ("PermBan", True), ("NoMemory", True), ("FCA_fixed", True), ("FCA_zscore", True),
]
results = {}
for method, recog in conditions:
    label = method + ("+認識" if recog else "")
    agg = {k: [] for k in ["rew_pre","rew_post","rew_post_first50","rew_last100","poisoned_hits_pre","safe_last100"]}
    for seed in range(20):
        r = run_condition(method, recog, seed=seed)
        for k in agg: agg[k].append(r[k])
    results[label] = agg

out_dir = os.path.dirname(os.path.abspath(__file__))
json.dump(results, open(os.path.join(out_dir, 'poison_v2_results.json'), 'w'), ensure_ascii=False, indent=1)
def mean(a): return sum(a)/len(a)
print("=== 毒シミュレーションv2 最終結果 (PK波形 × 8条件 × 20シード) ===")
for label, agg in results.items():
    print(f"{label:16s} rew_pre={mean(agg['rew_pre']):7.3f} rew_post={mean(agg['rew_post']):7.3f} first50={mean(agg['rew_post_first50']):7.3f} last100={mean(agg['rew_last100']):7.3f} poisoned_pre={mean(agg['poisoned_hits_pre']):5.2f} safe_last100={mean(agg['safe_last100']):5.3f}")
