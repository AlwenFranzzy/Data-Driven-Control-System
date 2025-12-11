"""
qlearning_module.py
-------------------
Q-learning module with model-assisted safety for the Modular Wastewater project.

Functions:
- discretize_state(state, env, n_bins_level=10)
- q_learning_model_assisted(env, mlp_model, scaler, n_bins_level=10,
                            episodes=2000, steps_per_episode=24*7, verbose=True)
- get_greedy_policy_from_qtable(q_table, env, n_bins_level=10)
- simulate_policy(policy_fn, env_seed=None, total_hours=24*7*2, env_kwargs=None)
"""

import numpy as np
import random
from typing import Callable, Tuple, List

# We import WastewaterEnv lazily in the __main__ demo to avoid circular import if not needed.
# from env_module import WastewaterEnv

# ----------------------------
# State discretization
# ----------------------------
def discretize_state(state: Tuple[float, int, int], env, n_bins_level: int = 10) -> Tuple[int, int, int]:
    """
    Discretize continuous state (level, hour, day) to indices for Q-table.

    level -> n_bins_level bins (0 .. n_bins_level-1) scaled by env.tank_capacity
    hour  -> 0..23
    day   -> 0..6
    """
    level, hour, day = state
    # Normalize and convert to bin index
    frac = 0.0
    if env.tank_capacity > 0:
        frac = float(level) / float(env.tank_capacity)
    # clamp between 0 and 1
    frac = max(0.0, min(1.0, frac))
    level_bin = int(min(n_bins_level - 1, int(frac * n_bins_level)))
    hour_i = int(hour) % 24
    day_i = int(day) % 7
    return (level_bin, hour_i, day_i)

# ----------------------------
# Q-Learning with model-assisted safety
# ----------------------------
def q_learning_model_assisted(env,
                              mlp_model,
                              scaler,
                              n_bins_level: int = 10,
                              episodes: int = 2000,
                              steps_per_episode: int = 24*7,
                              alpha: float = 0.1,
                              gamma: float = 0.95,
                              epsilon_start: float = 1.0,
                              epsilon_min: float = 0.05,
                              verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
    """
    Train a tabular Q-learning agent with an MLP-based safety penalty.

    Parameters
    ----------
    env : WastewaterEnv
        Environment instance (must implement reset() and step(action)).
    mlp_model : sklearn-like regressor
        Model predicting next tank level; used to anticipate overflow.
    scaler : sklearn.preprocessing.StandardScaler
        Scaler used to transform features before passing to mlp_model.
    n_bins_level : int
        Number of discretization bins for tank level.
    episodes : int
        Number of training episodes.
    steps_per_episode : int
        Steps per episode.
    alpha, gamma : float
        Learning rate and discount factor.
    epsilon_start, epsilon_min : float
        Epsilon-greedy exploration parameters.
    verbose : bool
        Print periodic progress.

    Returns
    -------
    q_table : np.ndarray
        Learned Q-table shape (n_bins_level, 24, 7, n_actions).
    rewards : list[float]
        Episode cumulative (combined) rewards.
    """
    n_actions = 3
    q_table = np.zeros((n_bins_level, 24, 7, n_actions), dtype=float)

    epsilon = epsilon_start
    decay = (epsilon_start - epsilon_min) / max(1, episodes)
    rewards = []

    for ep in range(episodes):
        state = env.reset()
        s_idx = discretize_state(state, env, n_bins_level)
        ep_reward = 0.0

        for _ in range(steps_per_episode):
            # epsilon-greedy
            if random.random() < epsilon:
                action = random.choice([0, 1, 2])
            else:
                action = int(np.argmax(q_table[s_idx]))

            # Step environment with chosen action
            next_state, reward, _, info = env.step(action)

            # Model-assisted safety: predict next level and penalize predicted overflow
            feat = np.array([[state[0], state[1], state[2], info['inflow'], action]])
            try:
                feat_scaled = scaler.transform(feat)
                pred_next = float(mlp_model.predict(feat_scaled)[0])
            except Exception:
                # If model/scaler fail, fallback to no safety prediction
                pred_next = next_state[0]

            predicted_overflow = max(0.0, pred_next - env.tank_capacity)
            safety_penalty = - (predicted_overflow / env.tank_capacity) * 0.5  # scale penalty

            combined_reward = reward + safety_penalty

            # Q-learning update
            ns_idx = discretize_state(next_state, env, n_bins_level)
            old_q = q_table[s_idx + (action,)]
            td_target = combined_reward + gamma * np.max(q_table[ns_idx])
            q_table[s_idx + (action,)] = old_q + alpha * (td_target - old_q)

            # advance
            state = next_state
            s_idx = ns_idx
            ep_reward += combined_reward

        # update epsilon
        epsilon = max(epsilon_min, epsilon - decay)
        rewards.append(ep_reward)

        if verbose and ((ep + 1) % max(1, episodes // 10) == 0 or ep == 0):
            recent = np.mean(rewards[-10:]) if len(rewards) >= 1 else 0.0
            print(f"Episode {ep+1}/{episodes}, avg reward last10: {recent:.4f}, epsilon: {epsilon:.3f}")

    return q_table, rewards

# ----------------------------
# Helpers: policies & simulation
# ----------------------------
def get_greedy_policy_from_qtable(q_table: np.ndarray, env, n_bins_level: int = 10) -> Callable:
    """
    Return a policy function (state, env) -> action that picks the greedy action
    from the provided q_table.
    """
    def policy_fn(state, _env):
        idx = discretize_state(state, env, n_bins_level)
        return int(np.argmax(q_table[idx]))
    return policy_fn

def simulate_policy(policy_fn: Callable,
                    env_seed: int = None,
                    total_hours: int = 24*7*2,
                    env_class=None,
                    env_kwargs: dict = None):
    """
    Simulate a policy for a number of hours and collect stats.

    Parameters
    ----------
    policy_fn : callable
        Function(policy_state, env) -> action
    env_seed : int | None
        Seed for environment (optional)
    total_hours : int
        Number of steps to simulate
    env_class : class
        Environment class to instantiate (must accept seed in kwargs).
    env_kwargs : dict
        Additional kwargs passed to env_class constructor.

    Returns
    -------
    levels : np.ndarray
        Recorded tank levels after each step.
    wasted_total : float
        Total wasted water (overflow + released).
    plants_total : float
        Total water used for plants.
    """
    if env_class is None:
        raise ValueError("env_class must be provided (e.g., env_module.WastewaterEnv)")

    if env_kwargs is None:
        env_kwargs = {}
    if env_seed is not None:
        env_kwargs = dict(env_kwargs, seed=env_seed)

    e = env_class(**env_kwargs)
    s = e.reset()
    levels = []
    wasted_total = 0.0
    plants_total = 0.0

    for _ in range(total_hours):
        a = policy_fn(s, e)
        s, _, _, info = e.step(a)
        levels.append(e.tank_level)
        wasted_total += info.get('wasted_overflow', 0.0) + info.get('wasted_action', 0.0)
        plants_total += info.get('water_used_plants', 0.0)

    return np.array(levels), wasted_total, plants_total

# ----------------------------
# Demo: run training if executed directly
# ----------------------------
if __name__ == "__main__":
    # Quick demo that uses env_module, data_module and model_module to train MLP then run Q-learning.
    from environment import WastewaterEnv
    from data import collect_data
    from model import train_mlp

    # 1) Prepare data & model (light / quick demo)
    env_demo = WastewaterEnv(tank_capacity=100.0, seed=42)
    df_demo = collect_data(env_demo, n_steps=2000)
    mlp_model, scaler, score = train_mlp(df_demo)

    # 2) Q-learning
    env_rl = WastewaterEnv(tank_capacity=100.0, seed=123)
    q_table, rewards = q_learning_model_assisted(env_rl, mlp_model, scaler,
                                                 n_bins_level=10, episodes=200, steps_per_episode=24*3,
                                                 verbose=True)

    # 3) Evaluate policies
    greedy_policy = get_greedy_policy_from_qtable(q_table, env_rl, n_bins_level=10)
    dummy = lambda s, e: 0

    levels_dummy, wasted_dummy, plants_dummy = simulate_policy(dummy, env_seed=7,
                                                               env_class=WastewaterEnv,
                                                               env_kwargs={'tank_capacity': 100.0})
    levels_learned, wasted_learned, plants_learned = simulate_policy(greedy_policy, env_seed=7,
                                                                     env_class=WastewaterEnv,
                                                                     env_kwargs={'tank_capacity': 100.0})

    print("Wasted - dummy:", wasted_dummy, "learned:", wasted_learned)
    print("Plants - dummy:", plants_dummy, "learned:", plants_learned)
