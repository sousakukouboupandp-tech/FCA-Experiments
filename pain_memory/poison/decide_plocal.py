# -*- coding: utf-8 -*-
"""
決着実験1: P_localの扱い方(3パターン)を、同一条件(FCA+認識)・同一20シードで比較する。
A: 単純合計 (正規化なし)
B: エピソード長で正規化
C: ピーク保持項を混ぜる (alpha=0.5)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'poison_v2_env.py'), encoding='utf-8').read())

def run_fca_recog(seed, plocal_mode, n_episodes=N_EPISODES):
    rng = np.random.RandomState(seed)
    env = PoisonEnvV2(recognition=True)
    Q = {}
    def get_Q(s):
        if s not in Q: Q[s] = np.zeros(N_ACTIONS)
        return Q[s]

    pain_memory = {}  # fca_key -> P_global

    def fca_key(state): return state

    def is_tabooed(state, action_idx):
        a = ACTIONS[action_idx]
        if a != "ACT": return False
        if state[0][0] != POISON_CELL: return False
        return pain_memory.get(fca_key(state), 0.0) < -THETA*10  # 負値なのでスケール調整(閾値はデータ見て調整)

    rewards_per_ep = []
    safe_flags_last100 = []

    for ep in range(n_episodes):
        state = env.reset(ep)
        total_r = 0.0
        p_local = 0.0
        episode_pains = []  # (t_in_state, pain値) の記録、Cパターンのpeak計算用
        episode_poisoned_state_key = None
        ep_safe = False

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
                a_idx = int(np.argmax(qvals))

            next_state, r, done, info = env.step(a_idx)
            total_r += r

            best_next = np.max(get_Q(next_state)) if not done else 0.0
            get_Q(s)[a_idx] += LR * (r + GAMMA*best_next - get_Q(s)[a_idx])

            if info["pain_now"] != 0.0:
                p_local = GAMMA_LOCAL * p_local + info["pain_now"]
                episode_pains.append(info["pain_now"])
            if info["poisoned_hit"]:
                episode_poisoned_state_key = fca_key(s)
            if info["success"]:
                if not env.has_antidote and not env.poison_active():
                    ep_safe = True

            state = next_state
            if done: break

        # エピソード終了時: P_localをP_globalへ統合(モード別)
        if episode_poisoned_state_key is not None:
            key = episode_poisoned_state_key
            if plocal_mode == "A_sum":
                fold_value = p_local
            elif plocal_mode == "B_norm":
                fold_value = p_local / max(1, t+1)
            elif plocal_mode == "C_peak":
                peak = min(episode_pains) if episode_pains else 0.0  # 最も負(強い)値
                fold_value = 0.5*p_local + 0.5*peak
            pain_memory[key] = LAMBDA_GLOBAL * pain_memory.get(key, 0.0) + fold_value

        # 全キーに対してエピソード間減衰を適用(訪問がなくても時間経過で薄れる)
        for k in list(pain_memory.keys()):
            pain_memory[k] *= LAMBDA_GLOBAL
            if abs(pain_memory[k]) < P_LIMIT:
                del pain_memory[k]

        rewards_per_ep.append(total_r)
        if ep >= n_episodes - 100:
            safe_flags_last100.append(1.0 if ep_safe else 0.0)

    rewards_per_ep = np.array(rewards_per_ep)
    return {
        "rew_last100": float(np.mean(rewards_per_ep[-100:])),
        "rew_post_first50": float(np.mean(rewards_per_ep[SWITCH_EP:SWITCH_EP+50])),
        "safe_last100": float(np.mean(safe_flags_last100)) if safe_flags_last100 else 0.0,
    }

print("=== P_local扱い方 3パターン比較 (FCA+認識, 20シード) ===")
for mode in ["A_sum", "B_norm", "C_peak"]:
    results = [run_fca_recog(seed, mode) for seed in range(20)]
    rew = np.mean([r["rew_last100"] for r in results])
    first50 = np.mean([r["rew_post_first50"] for r in results])
    safe = np.mean([r["safe_last100"] for r in results])
    print(f"{mode:8s}: rew_last100={rew:7.3f}  rew_post_first50={first50:7.3f}  safe_last100={safe:5.3f}")
