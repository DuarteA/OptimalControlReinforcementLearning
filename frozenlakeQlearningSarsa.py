import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium as gym

def make_env(env_id="FrozenLake-v1", is_slippery=True):
    return gym.make(env_id, is_slippery=is_slippery)

def reset_env(env):
    res = env.reset()
    if isinstance(res, tuple):
        # new API: (obs, info)
        return res[0]
    return res

def step_env(env, action):
    res = env.step(action)
    # new API: (obs, reward, terminated, truncated, info)
    if len(res) == 5:
        obs, reward, terminated, truncated, info = res
        return obs, reward, (terminated or truncated), info
    # old API: (obs, reward, done, info)
    if len(res) == 4:
        obs, reward, done, info = res
        return obs, reward, done, info
    raise RuntimeError("Unexpected env.step() return signature")

def epsilon_greedy(Q, state, nA, eps):
    if random.random() < eps:
        return random.randrange(nA)
    else:
        return int(np.argmax(Q[state]))

def run_q_learning(env, num_episodes=5000, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.999):
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    rewards = []
    eps = eps_start
    for ep in range(1, num_episodes+1):
        state = reset_env(env)
        done = False
        total_r = 0.0
        while not done:
            action = epsilon_greedy(Q, state, nA, eps)
            next_state, reward, done, _ = step_env(env, action)
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])
            state = next_state
            total_r += reward
        rewards.append(total_r)
        eps = max(eps_end, eps * eps_decay)
    return Q, rewards

def run_sarsa(env, num_episodes=5000, alpha=0.1, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.999):
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    rewards = []
    eps = eps_start
    for ep in range(1, num_episodes+1):
        state = reset_env(env)
        action = epsilon_greedy(Q, state, nA, eps)
        done = False
        total_r = 0.0
        while not done:
            next_state, reward, done, _ = step_env(env, action)
            next_action = epsilon_greedy(Q, next_state, nA, eps)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            total_r += reward
        rewards.append(total_r)
        eps = max(eps_end, eps * eps_decay)
    return Q, rewards

def moving_average(x, w=100):
    return np.convolve(x, np.ones(w)/w, mode='valid')

def policy_from_Q(Q):
    return np.argmax(Q, axis=1)

def plot_results(rew_q, rew_sarsa, Q_q, Q_sarsa, env, window=100, save_path=None):
    ma_q = moving_average(rew_q, window)
    ma_sarsa = moving_average(rew_sarsa, window)
    episodes = np.arange(len(ma_q)) + window

    # success rate (reward==1)
    succ_q = moving_average([1.0 if r > 0 else 0.0 for r in rew_q], window)
    succ_sarsa = moving_average([1.0 if r > 0 else 0.0 for r in rew_sarsa], window)

    plt.figure(figsize=(10,6))
    plt.plot(episodes, ma_q, label='Q-learning (avg reward)')
    plt.plot(episodes, ma_sarsa, label='SARSA (avg reward)')
    plt.xlabel('Episode')
    plt.ylabel(f'Moving avg reward (window={window})')
    plt.legend()
    plt.grid(True)

    plt.twinx()
    plt.plot(episodes, succ_q, '--', color='C0', alpha=0.6, label='Q-learning (win rate)')
    plt.plot(episodes, succ_sarsa, '--', color='C1', alpha=0.6, label='SARSA (win rate)')
    plt.ylabel('Win rate')
    plt.legend(loc='lower right')

    plt.title('FrozenLake: Q-learning vs SARSA')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    # Print deterministic policies
    def show_policy(Q, name):
        pi = policy_from_Q(Q)
        nS = env.observation_space.n
        size = int(math.sqrt(nS))
        arrows = {0:'←',1:'↓',2:'→',3:'↑'}
        grid = np.array([arrows[a] for a in pi]).reshape((size,size))
        print(f"\n{name} deterministic policy (state grid):")
        for row in grid:
            print(' '.join(row))

    show_policy(Q_q, "Q-learning")
    show_policy(Q_sarsa, "SARSA")


# --- added: animation helpers for FrozenLake ---
def _get_desc_chars(env):
    desc = env.unwrapped.desc  # dtype=bytes
    return np.array([[c.decode("utf-8") for c in row] for row in desc])

def _grid_image_from_desc(desc_chars):
    # map chars to RGB colors
    n = desc_chars.shape[0]
    img = np.ones((n, n, 3), dtype=np.float32) * 1.0  # default white
    for i in range(n):
        for j in range(n):
            c = desc_chars[i, j]
            if c == "S":       # start
                img[i, j] = np.array([0.8, 1.0, 0.8])
            elif c == "F":     # frozen
                img[i, j] = np.array([1.0, 1.0, 1.0])
            elif c == "H":     # hole
                img[i, j] = np.array([0.2, 0.2, 0.2])
            elif c == "G":     # goal
                img[i, j] = np.array([1.0, 0.85, 0.2])
    return img

def _state_to_coords(state, size):
    return (state // size, state % size)

def animate_policy(env, policy_fn, title="Policy animation", max_steps=100, interval=350, save_path=None):
    desc_chars = _get_desc_chars(env)
    base_img = _grid_image_from_desc(desc_chars)
    n = desc_chars.shape[0]

    # run one episode collecting states
    states = []
    obs = reset_env(env)
    done = False
    steps = 0
    while not done and steps < max_steps:
        states.append(obs)
        a = policy_fn(obs)
        obs, _, done, _ = step_env(env, int(a))
        steps += 1
    # include terminal state for frame
    states.append(obs)

    # build matplotlib animation
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(base_img, origin='upper')
    scat = ax.scatter([], [], s=300, c='red', edgecolors='black', linewidths=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    def init():
        if len(states) > 0:
            r0, c0 = _state_to_coords(int(states[0]), n)
            scat.set_offsets([[c0, r0]])
        return (scat,)

    def update(frame):
        s = states[frame]
        r, c = _state_to_coords(int(s), n)
        # scatter expects (x, y) in data coords: use column, row
        scat.set_offsets([[c, r]])
        return (scat,)

    anim = animation.FuncAnimation(fig, update, frames=len(states),
                                  init_func=init, interval=interval, blit=True)

    if save_path:
        anim.save(save_path, dpi=150)
    plt.show()
    return anim

def main():
    random.seed(0)
    np.random.seed(0)

    # Use is_slippery=True to test stochastic environment (default FrozenLake). Set False for deterministic.
    env = make_env("FrozenLake-v1", is_slippery=True)

    episodes = 8000
    # run Q-learning
    Q_q, rew_q = run_q_learning(env, num_episodes=episodes, alpha=0.1, gamma=0.99,
                                eps_start=1.0, eps_end=0.05, eps_decay=0.9995)
    # run SARSA
    Q_sarsa, rew_sarsa = run_sarsa(env, num_episodes=episodes, alpha=0.1, gamma=0.99,
                                   eps_start=1.0, eps_end=0.05, eps_decay=0.9995)

    
    # animate initial (random) policy
    nA = env.action_space.n
    random_policy = lambda s: random.randrange(nA)
    animate_policy(env, random_policy, title="Initial (random) policy", max_steps=50, interval=300)

    # animate final deterministic policy from Q-learning
    pi_final = policy_from_Q(Q_q)
    final_policy = lambda s: int(pi_final[int(s)])
    animate_policy(env, final_policy, title="Final Q-learning deterministic policy", max_steps=50, interval=300)

    plot_results(rew_q, rew_sarsa, Q_q, Q_sarsa, env, window=200)


    env.close()

if __name__ == "__main__":
    main()