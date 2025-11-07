import random
import collections
import numpy as np
import math
import time
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import imageio

# -------- config / hyperparams --------
ENV_ID = "LunarLander-v3"
SEED = 123
NUM_EPISODES = 800            # increase for stronger results
EVAL_EVERY = 50
EVAL_EPISODES = 8

# DQN hyperparams
DQN_BATCH = 64
DQN_LR = 1e-3
DQN_GAMMA = 0.99
DQN_REPLAY_SIZE = 50000
DQN_START_TRAIN = 1000
DQN_TRAIN_PER_STEP = 1
DQN_TARGET_UPDATE = 1000     # steps
DQN_EPS_START = 1.0
DQN_EPS_END = 0.02
DQN_EPS_DECAY = 0.995        # per episode decay

# SARSA hyperparams
SARSA_LR = 1e-3
SARSA_GAMMA = 0.99
SARSA_EPS_START = 1.0
SARSA_EPS_END = 0.02
SARSA_EPS_DECAY = 0.995

# network
HIDDEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------- utilities --------
def make_env(seed=None):
    env = gym.make(ENV_ID)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env

def soft_update(src, dst, tau):
    for s, d in zip(src.parameters(), dst.parameters()):
        d.data.copy_(tau * s.data + (1.0 - tau) * d.data)


# -------- simple MLP Q network --------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# -------- replay buffer for DQN --------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, done = map(np.array, zip(*batch))
        return s, a, r, ns, done

    def __len__(self):
        return len(self.buffer)


# -------- DQN agent --------
class DQNAgent:
    def __init__(self, obs_dim, n_actions):
        self.n_actions = n_actions
        self.q = QNetwork(obs_dim, n_actions).to(DEVICE)
        self.target_q = QNetwork(obs_dim, n_actions).to(DEVICE)
        self.target_q.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=DQN_LR)
        self.replay = ReplayBuffer(DQN_REPLAY_SIZE)
        self.total_steps = 0
        self.eps = DQN_EPS_START

    def select_action(self, state, eval_mode=False):
        if (not eval_mode) and (random.random() < self.eps):
            return random.randrange(self.n_actions)
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(s)
        return int(qvals.argmax().item())

    def store(self, s, a, r, ns, done):
        self.replay.push(s, a, r, ns, done)
        self.total_steps += 1

    def train_step(self):
        if len(self.replay) < DQN_BATCH or self.total_steps < DQN_START_TRAIN:
            return
        for _ in range(DQN_TRAIN_PER_STEP):
            s, a, r, ns, done = self.replay.sample(DQN_BATCH)
            s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
            a = torch.tensor(a, dtype=torch.int64, device=DEVICE).unsqueeze(1)
            r = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
            ns = torch.tensor(ns, dtype=torch.float32, device=DEVICE)
            done = torch.tensor(done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

            q_vals = self.q(s).gather(1, a)
            with torch.no_grad():
                next_q = self.target_q(ns).max(1)[0].unsqueeze(1)
                target = r + (1 - done) * DQN_GAMMA * next_q
            loss = nn.functional.mse_loss(q_vals, target)

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.opt.step()

        # target update
        if self.total_steps % DQN_TARGET_UPDATE == 0:
            self.target_q.load_state_dict(self.q.state_dict())

    def decay_epsilon(self):
        self.eps = max(DQN_EPS_END, self.eps * DQN_EPS_DECAY)


# -------- SARSA agent (on-policy) --------
class SARSAAgent:
    def __init__(self, obs_dim, n_actions):
        self.n_actions = n_actions
        self.q = QNetwork(obs_dim, n_actions).to(DEVICE)
        self.opt = optim.Adam(self.q.parameters(), lr=SARSA_LR)
        self.eps = SARSA_EPS_START

    def select_action(self, state, eval_mode=False):
        if (not eval_mode) and (random.random() < self.eps):
            return random.randrange(self.n_actions)
        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(s)
        return int(qvals.argmax().item())

    def update(self, s, a, r, ns, na, done):
        s_t = torch.tensor([s], dtype=torch.float32, device=DEVICE)
        ns_t = torch.tensor([ns], dtype=torch.float32, device=DEVICE)
        q_sa = self.q(s_t)[0, a]
        with torch.no_grad():
            q_ns_na = 0.0 if done else self.q(ns_t)[0, na]
            target = r + SARSA_GAMMA * q_ns_na
        loss = nn.functional.mse_loss(q_sa, torch.tensor(target, dtype=torch.float32, device=DEVICE))
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

    def decay_epsilon(self):
        self.eps = max(SARSA_EPS_END, self.eps * SARSA_EPS_DECAY)


# -------- evaluation helper --------
def evaluate_agent(agent, env, episodes=5):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            a = agent.select_action(obs, eval_mode=True)
            obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            total += r
        returns.append(total)
    return np.mean(returns), np.std(returns)


# -------- train / compare --------
def run_compare(num_episodes=NUM_EPISODES):
    env = make_env(seed=SEED)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # create agents
    dqn = DQNAgent(obs_dim, n_actions)
    sarsa = SARSAAgent(obs_dim, n_actions)

    # records
    dqn_returns = []
    sarsa_returns = []
    dqn_eval = []
    sarsa_eval = []
    steps = 0

    # Train DQN
    print("Training DQN...")
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            a = dqn.select_action(obs)
            next_obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            dqn.store(obs, a, r, next_obs, float(done))
            dqn.train_step()
            obs = next_obs
            total += r
            steps += 1
        dqn.decay_epsilon()
        dqn_returns.append(total)
        if ep % EVAL_EVERY == 0 or ep == 1:
            mean, std = evaluate_agent(dqn, make_env(seed=SEED + 100))
            dqn_eval.append((ep, mean, std))
            print(f"DQN ep {ep:4d} return {total:7.1f} eval_mean {mean:6.1f} eps {dqn.eps:.3f}")

    # Train SARSA
    print("\nTraining SARSA...")
    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        a = sarsa.select_action(obs)
        done = False
        total = 0.0
        while not done:
            next_obs, r, term, trunc, _ = env.step(a)
            done = term or trunc
            next_a = sarsa.select_action(next_obs)
            sarsa.update(obs, a, r, next_obs, next_a, done)
            obs = next_obs
            a = next_a
            total += r
            steps += 1
        sarsa.decay_epsilon()
        sarsa_returns.append(total)
        if ep % EVAL_EVERY == 0 or ep == 1:
            mean, std = evaluate_agent(sarsa, make_env(seed=SEED + 200))
            sarsa_eval.append((ep, mean, std))
            print(f"SARSA ep {ep:4d} return {total:7.1f} eval_mean {mean:6.1f} eps {sarsa.eps:.3f}")

    env.close()

    # plotting
    def smooth(x, w=10):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    plt.figure(figsize=(12,6))
    plt.plot(range(len(dqn_returns)), dqn_returns, alpha=0.3, label='DQN returns (train)')
    plt.plot(range(len(sarsa_returns)), sarsa_returns, alpha=0.3, label='SARSA returns (train)')
    #plt.plot(range(len(smooth(dqn_returns))), smooth(dqn_returns), label='DQN smoothed')
    #plt.plot(range(len(smooth(sarsa_returns))), smooth(sarsa_returns), label='SARSA smoothed')
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lunar_compare_returns.png")
    plt.show()

    # Figure 2: only DQN returns (raw + smoothed)
    plt.figure(figsize=(10,5))
    plt.plot(range(len(dqn_returns)), dqn_returns, color='C0', alpha=0.4, label='DQN returns (train)')
    #if len(dqn_returns) >= 10:
    #    plt.plot(range(len(smooth(dqn_returns))), smooth(dqn_returns), color='C1', label='DQN smoothed')
    plt.title("DQN Training Returns")
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lunar_dqn_returns.png")
    plt.show()

    print("\nEvaluation summary (ep, mean, std)")
    print("DQN eval:", dqn_eval)
    print("SARSA eval:", sarsa_eval)
    return {
        "dqn_returns": dqn_returns,
        "sarsa_returns": sarsa_returns,
        "dqn_eval": dqn_eval,
        "sarsa_eval": sarsa_eval
   
        # return trained agents so caller can visualize policies
        , "dqn_agent": dqn, "sarsa_agent": sarsa
    }

def render_policy_live(agent, env_id=ENV_ID, episodes=3, seed=None, sleep=0.02, render_mode="human"):
    """
    Run a few episodes showing the policy live. Uses render_mode='human' by default.
    Note: this requires a display (not headless). Use save_policy_animation for GIF capture.
    """
    try:
        env = gym.make(env_id, render_mode=render_mode)
    except Exception as e:
        print("render_policy_live: could not create env:", e)
        return
    for ep in range(episodes):
        obs, _ = env.reset(seed=(None if seed is None else seed + ep))
        done = False
        total = 0.0
        # Attempt initial render (some envs only render after reset)
        try:
            frame = env.render()
            if frame is None and render_mode == "human":
                # human may not return frames; that's fine
                pass
        except Exception:
            pass
        while not done:
            a = agent.select_action(obs, eval_mode=True)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            total += r
            if render_mode == "human":
                time.sleep(sleep)
        print(f"[live] Episode {ep+1} return {total:.2f}")
    try:
        env.close()
    except Exception:
        pass


def save_policy_animation(agent, env_id=ENV_ID, filename="policy.gif", episode=0, fps=30, max_frames=600):
    """
    Capture a single episode with render_mode='rgb_array' and save a GIF.
    Prints diagnostics if no frames captured.
    Requires imageio (pip install imageio).
    """
    try:
        env = gym.make(env_id, render_mode="rgb_array")
    except Exception as e:
        print("save_policy_animation: could not create env with rgb_array:", e)
        print("Trying fallback: creating env without explicit render_mode.")
        try:
            env = gym.make(env_id)
        except Exception as e2:
            print("Fallback env creation failed:", e2)
            return False

    frames = []
    obs, info = env.reset()
    # try immediate render after reset
    try:
        frame = env.render()
    except Exception:
        frame = None
    if frame is not None:
        frames.append(frame)
    # step and collect frames, with fallback to info.get('rgb_array') if present
    done = False
    steps = 0
    while not done and steps < max_frames:
        a = agent.select_action(obs, eval_mode=True)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        steps += 1
        # primary: env.render()
        try:
            frame = env.render()
        except Exception:
            frame = None
        # fallback: some envs provide rgb array in info
        if frame is None and isinstance(info, dict):
            frame = info.get("rgb_array", None) or info.get("frame", None)
        if frame is not None:
            frames.append(frame)

    try:
        env.close()
    except Exception:
        pass

    if len(frames) == 0:
        print("save_policy_animation: no frames captured. Diagnostics:")
        print(" - env_id:", env_id)
        print(" - render_mode attempted: 'rgb_array'")
        print(" - max_frames:", max_frames)
        print(" - Try: install Box2D / run locally with display / or change env to CartPole-v1 for simple testing.")
        return False

    # basic diagnostics
    import numpy as _np
    print("save_policy_animation: captured frames:", len(frames))
    try:
        arr = _np.array(frames[0])
        print("frame shape dtype:", arr.shape, arr.dtype)
    except Exception:
        pass

    try:
        imageio.mimsave(filename, frames, fps=fps)
        print("Saved animation to", filename)
        return True
    except Exception as e:
        print("Failed to save animation:", e)
        return False

if __name__ == "__main__":
    start = time.time()
    results = run_compare()
    print("Done in %.1f sec" % (time.time() - start))

    # show the trained DQN policy live (requires pyglet/display)
    try:
        render_policy_live(results["dqn_agent"], env_id=ENV_ID, episodes=3, render_mode="human")
    except Exception as e:
        print("live render failed:", e)

    # save a GIF of the trained DQN policy (requires imageio)
    try:
        save_policy_animation(results["dqn_agent"], env_id=ENV_ID, filename="dqn_policy.gif", fps=30)
    except Exception as e:
        print("gif save failed:", e)