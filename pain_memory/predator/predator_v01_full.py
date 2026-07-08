# -*- coding: utf-8 -*-
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predator_v01_env.py'), encoding='utf-8').read())

def argmax_rt(qvals, rng):
    finite_mask = np.isfinite(qvals)
    if not finite_mask.any():
        return rng.randint(len(qvals))
    max_val = np.max(qvals[finite_mask])
    candidates = np.where((qvals == max_val) & finite_mask)[0]
    return rng.choice(candidates)

def run_condition(method, recognition, seed, use_phase=True, n_episodes=N_EPISODES):
    """
    method: "PermBan" | "NoMemory" | "FCA"
    use_phase: FCAのタブーキーに移動方向(位相相当)を含めるか(穴2の検証)
    """
    rng = np.random.RandomState(seed)
    env = PredatorEnv(recognition=recognition)
    Q = {}
    def get_Q(s):
        if s not in Q: Q[s] = np.zeros(N_ACTIONS)
        return Q[s]

    pain_memory = {}
    perm_ban = set()
    LAMBDA_GLOBAL = 0.97

    def fca_key(state, next_pos):
        if not recognition:
            return (state[0], next_pos)
        if use_phase:
            return (state, next_pos)  # 状態全体+移動先
        else:
            return (state[0], state[1], next_pos)  # 位置+捕食者位置+移動先（方向は捨てる）

    def permban_key(state, next_pos):
        return (state[0], next_pos)  # 自分の位置+移動先。文脈は考慮しない

    def is_tabooed(state, action_idx):
        next_pos = move(state[0], ACTIONS[action_idx])
        if method == "PermBan":
            return permban_key(state, next_pos) in perm_ban
        elif method == "FCA":
            k = fca_key(state, next_pos)
            return pain_memory.get(k, 0.0) > 0.3
        return False

    rewards_per_ep = []
    collisions_pre = 0
    success_flags_last100 = []

    for ep in range(n_episodes):
        state = env.reset(ep)
        total_r = 0.0
        ep_collided = False
        ep_success = False

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
                a_idx = argmax_rt(qvals, rng)

            next_state, r, done, info = env.step(a_idx)
            total_r += r
            best_next = np.max(get_Q(next_state)) if not done else 0.0
            get_Q(s)[a_idx] += LR * (r + GAMMA*best_next - get_Q(s)[a_idx])

            if info["collided"]:
                ep_collided = True
                if method == "FCA":
                    k = fca_key(s, env.pos)  # env.pos = 実際に移動した先(衝突したマス)
                    pain_memory[k] = pain_memory.get(k, 0.0) + 1.0
                elif method == "PermBan":
                    perm_ban.add(permban_key(s, env.pos))
            if info["success"]:
                ep_success = True

            state = next_state
            if done: break

        if method == "FCA":
            for k in list(pain_memory.keys()):
                pain_memory[k] *= LAMBDA_GLOBAL
                if pain_memory[k] < 0.05:
                    del pain_memory[k]

        rewards_per_ep.append(total_r)
        if ep < SWITCH_EP and ep_collided:
            collisions_pre += 1
        if ep >= n_episodes - 100:
            success_flags_last100.append(1.0 if ep_success else 0.0)

    rewards_per_ep = np.array(rewards_per_ep)
    return {
        "rew_pre": float(np.mean(rewards_per_ep[:SWITCH_EP])),
        "rew_post_first50": float(np.mean(rewards_per_ep[SWITCH_EP:SWITCH_EP+50])),
        "rew_last100": float(np.mean(rewards_per_ep[-100:])),
        "collisions_pre": int(collisions_pre),
        "success_last100": float(np.mean(success_flags_last100)) if success_flags_last100 else 0.0,
    }

print("run_condition定義完了")
