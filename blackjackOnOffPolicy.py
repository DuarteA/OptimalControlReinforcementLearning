import collections
import math
import random
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Card draw distribution (infinite deck approximation, as in Sutton & Barto)
CARD_VALUES = [1,2,3,4,5,6,7,8,9,10]
CARD_PROBS = np.array([1/13]*9 + [4/13])  # 1-9: 1/13 each, 10: 4/13

# State representation: (player_sum, dealer_showing, usable_ace) 
# player_sum in 0..31 (we'll consider 12..21 relevant), dealer_showing 1..10, usable_ace True/False
ACTIONS = [0, 1]  # 0: stick, 1: hit
GAMMA = 1.0

def draw_card():
    return np.random.choice(CARD_VALUES, p=CARD_PROBS)

def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21

def sum_hand(hand):
    s = sum(hand)
    if 1 in hand and s + 10 <= 21:
        return s + 10
    return s

def is_bust(hand):
    return sum_hand(hand) > 21

# Helper: apply a drawn card to player's sum and usable_ace
def next_player_state(player_sum, usable_ace_flag, card):
    # reconstruct minimal representation: if usable_ace_flag is True, subtract 10 to get "raw" sum without ACE counting 11
    if usable_ace_flag:
        raw = player_sum - 10
        hand_sum = raw + card
        # if new sum with ace counted as 11 still <=21, keep usable ace
        if 1 == card:
            # adding an Ace when already had usable ace -> one ace becomes 11, the new one maybe becomes 1
            # But easier: treat effect via typical ace fallback:
            s = raw + card
            usable = True if (1 in [1] and s + 10 <= 21) else False  # trivial
        # simpler approach: compute raw hand values by simulating possible ace adjustments
    # Simpler: keep track only with numeric operations like Sutton: add card, if card==1 treat as 11 when possible
    # We'll use a simplified numeric update:
    s = player_sum + card
    usable = usable_ace_flag
    if card == 1:
        # if adding an ace can be counted as 11 without busting
        if player_sum + 11 <= 21:
            s = player_sum + 11
            usable = True
        else:
            s = player_sum + 1
    # if we bust and had a usable ace, convert usable ace from 11 to 1 (subtract 10)
    if s > 21 and usable:
        s -= 10
        usable = False
    return s, usable

# For dealer dynamics (dealer shows upcard and draws until 17 or more),
# compute distribution over final dealer totals given dealer's current sum and usable ace.
from functools import lru_cache

@lru_cache(maxsize=None)
def dealer_final_probs(dealer_sum, usable_ace_flag):
    # returns dict: total (17..21), 'bust' -> prob
    if dealer_sum >= 17:
        if dealer_sum > 21:
            return {'bust': 1.0}
        return {dealer_sum: 1.0}
    probs = collections.defaultdict(float)
    for card, p in zip(CARD_VALUES, CARD_PROBS):
        s = dealer_sum
        usable = usable_ace_flag
        if card == 1:
            if s + 11 <= 21:
                s = s + 11
                usable = True
            else:
                s = s + 1
        else:
            s = s + card
        if s > 21 and usable:
            s -= 10
            usable = False
        if s > 21:
            probs['bust'] += p
        else:
            sub = dealer_final_probs(s, usable)
            for k, v in sub.items():
                probs[k] += p * v
    return dict(probs)

# Enumerate all possible player states (we focus on player_sum 12..21 and useable ace True/False)
ALL_STATES = []
for player_sum in range(12, 22):
    for dealer_show in range(1, 11):
        for ua in (False, True):
            ALL_STATES.append((player_sum, dealer_show, ua))
ALL_STATES = tuple(ALL_STATES)

def get_transitions(state, action):
    # returns list of (prob, next_state, reward, done)
    player_sum, dealer_show, usable = state
    results = []
    if action == 1:  # hit
        # draw a card
        for card, p in zip(CARD_VALUES, CARD_PROBS):
            s = player_sum
            ua = usable
            # add card
            if card == 1:
                if s + 11 <= 21:
                    s = s + 11
                    ua = True
                else:
                    s = s + 1
            else:
                s = s + card
            if s > 21 and ua:
                s -= 10
                ua = False
            if s > 21:
                # bust -> terminal with reward -1
                results.append((p, None, -1.0, True))
            else:
                # non-terminal next state
                results.append((p, (s, dealer_show, ua), 0.0, False))
    else:  # stick: simulate dealer
        # Dealer initial hand: dealer_show + unknown second card drawn according to CARD_PROBS
        # but dealer_final_probs needs initial sum with that second card; we marginalize over second card
        # First, assemble distribution over dealer final outcomes
        # Start by considering dealer second card:
        # second card may be Ace (1) or others: compute initial sum and usable flag, then dealer_final_probs
        aggregated = collections.defaultdict(float)
        for card, p_card in zip(CARD_VALUES, CARD_PROBS):
            # dealer's initial sum
            dealer_sum = dealer_show
            usable_d = False
            if card == 1:
                if dealer_sum + 11 <= 21:
                    dealer_sum = dealer_sum + 11
                    usable_d = True
                else:
                    dealer_sum = dealer_sum + 1
            else:
                dealer_sum = dealer_sum + card
            if dealer_sum > 21 and usable_d:
                dealer_sum -= 10
                usable_d = False
            sub = dealer_final_probs(dealer_sum, usable_d)
            for k, v in sub.items():
                aggregated[k] += p_card * v
        # Now compare player's total vs dealer outcomes
        for outcome, p_out in aggregated.items():
            if outcome == 'bust':
                reward = 1.0
            else:
                deal_sum = outcome
                if deal_sum > player_sum:
                    reward = -1.0
                elif deal_sum < player_sum:
                    reward = 1.0
                else:
                    reward = 0.0
            results.append((p_out, None, reward, True))
    return results

# Value iteration to compute optimal V and optimal greedy policy
def value_iteration(states, theta=1e-6):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            v = V[s]
            action_values = []
            for a in ACTIONS:
                q = 0.0
                for p, ns, r, done in get_transitions(s, a):
                    if done:
                        q += p * r
                    else:
                        q += p * (r + GAMMA * V[ns])
                action_values.append(q)
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    # derive greedy policy and Q-values
    pi = {}
    Q = {}
    for s in states:
        qvals = {}
        for a in ACTIONS:
            q = 0.0
            for p, ns, r, done in get_transitions(s, a):
                if done:
                    q += p * r
                else:
                    q += p * (r + GAMMA * V[ns])
            qvals[a] = q
            Q[(s,a)] = q
        # break ties by choosing hit (1) if equal
        best_a = max(ACTIONS, key=lambda a: (qvals[a], a))
        pi[s] = best_a
    return V, Q, pi

# Monte Carlo episode generator using gymnasium environment to match "Blackjack-v1"
def sample_episode_gym(env, policy, behavior_policy=None):
    # behavior_policy(s) -> action (if provided, overrides policy)
    obs, _ = env.reset()
    episode = []
    done = False
    while not done:
        s = obs  # (player_sum, dealer_show, usable_ace)
        # Map env obs to our state format (player_sum, dealer_show, usable)
        state = (s[0], s[1], s[2])
        if behavior_policy is not None:
            a = behavior_policy(state)
        else:
            # fallback to hit (1) for states not in the provided policy (e.g. player_sum < 12)
            a = policy.get(state, 1)
        obs, reward, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        episode.append((state, a, reward))
        if done:
            break
    return episode


def mc_on_policy_eval_trace(env_id, pi, num_episodes=20000, record_sa=None):
    env = gym.make(env_id)
    returns_sum = collections.defaultdict(float)
    returns_count = collections.defaultdict(int)
    Q = {}
    trace = []
    for ep in range(num_episodes):
        ep_data = sample_episode_gym(env, pi)
        G = 0.0
        visited = set()
        for t in reversed(range(len(ep_data))):
            s, a, r = ep_data[t]
            G = r + GAMMA * G
            if (s,a) not in visited:
                visited.add((s,a))
                returns_sum[(s,a)] += G
                returns_count[(s,a)] += 1
                Q[(s,a)] = returns_sum[(s,a)] / returns_count[(s,a)]
        # record current estimate for the requested state-action (may be missing -> nan)
        if record_sa is not None:
            trace.append(Q.get(record_sa, np.nan))
    env.close()
    return Q, np.array(trace)


def mc_off_policy_eval_trace(env_id, pi, behavior_eps=1, num_episodes=20000, record_sa=None):
    env = gym.make(env_id)
    C = collections.defaultdict(float)  # cumulative sum of weights
    Q = collections.defaultdict(float)
    trace = []
    for ep in range(num_episodes):
        def b_policy(s):
            if random.random() < behavior_eps:
                return random.choice(ACTIONS)
            else:
                return pi.get(s, 1)
        ep_data = sample_episode_gym(env, pi, behavior_policy=b_policy)
        G = 0.0
        W = 1.0
        visited = set()
        for t in reversed(range(len(ep_data))):
            s, a, r = ep_data[t]
            G = r + GAMMA * G
            if (s,a) not in visited:
                visited.add((s,a))
                C[(s,a)] += W
                Q[(s,a)] += (W / C[(s,a)]) * (G - Q[(s,a)])
            pi_action = pi.get(s, 1)
            pi_prob = 1.0 if a == pi_action else 0.0
            b_prob = (behavior_eps / len(ACTIONS)) + (1 - behavior_eps) * (1.0 if a == pi_action else 0.0)
            if b_prob == 0:
                W = 0.0
            else:
                W = W * (pi_prob / b_prob)
            if W == 0.0:
                break
        # record current estimate for the requested state-action
        if record_sa is not None:
            trace.append(Q.get(record_sa, np.nan))
    env.close()
    return Q, np.array(trace)


def mc_off_policy_eval_trace_ordinary(env_id, pi, behavior_eps=1.0, num_episodes=20000, record_sa=None):
    """
    Off-policy first-visit MC using ordinary (unweighted) importance sampling.
    Returns (Q_estimates, trace_array) where trace_array contains the running
    ordinary-IS estimate for record_sa after each episode (nan if not seen yet).
    """
    env = gym.make(env_id)
    trace = []
    # accumulator and count for ordinary IS estimator for record_sa
    sum_num = 0.0
    count = 0
    for ep in range(num_episodes):
        def b_policy(s):
            if random.random() < behavior_eps:
                return random.choice(ACTIONS)
            else:
                return pi.get(s, 1)
        ep_data = sample_episode_gym(env, pi, behavior_policy=b_policy)
        T = len(ep_data)
        # compute returns G_t for each t (from t to end)
        G = [0.0] * T
        g = 0.0
        for t in range(T-1, -1, -1):
            _, _, r = ep_data[t]
            g = r + GAMMA * g
            G[t] = g
        # compute ratios r_t = pi(a_t|s_t) / b(a_t|s_t)
        ratios = []
        for (s, a, _) in ep_data:
            pi_action = pi.get(s, 1)
            pi_prob = 1.0 if a == pi_action else 0.0
            b_prob = (behavior_eps / len(ACTIONS)) + (1 - behavior_eps) * (1.0 if a == pi_action else 0.0)
            # if b_prob == 0, set ratio = 0 to avoid div by zero; ordinary IS will then get W=0 for any t before that step
            ratios.append(0.0 if b_prob == 0 else (pi_prob / b_prob))
        # for each first-visit (s,a) occurrence, compute W_t = product_{k=t}^{T-1} ratios[k]
        seen = set()
        for t in range(T):
            s, a, _ = ep_data[t]
            if (s,a) in seen:
                continue
            seen.add((s,a))
            if record_sa is not None and (s,a) == record_sa:
                # compute W_t
                W_t = 1.0
                for k in range(t, T):
                    W_t *= ratios[k]
                    if W_t == 0.0:
                        break
                # ordinary IS: accumulate numerator and count
                sum_num += W_t * G[t]
                count += 1
        # append current estimator for record_sa
        trace.append((sum_num / count) if count > 0 else np.nan)
    env.close()
    # Q estimates returned empty dict; caller uses trace only
    return {}, np.array(trace)

if __name__ == "__main__":
    # compute exact optimal policy by DP
    V, Q_exact, pi_star = value_iteration(ALL_STATES)
    ENV_ID = "Blackjack-v1"

    # choose a state to monitor — use a safe fallback if the preferred state isn't in pi_star
    record_state = (20, 10, False)
    if record_state not in pi_star:
        # pi_star maps state -> action; get any valid state
        record_state = next(iter(pi_star.keys()))
    record_key = (record_state, pi_star[record_state])

    print(f"\nCollecting traces for variance plot on state-action {record_key} ...")

    # on-policy trace
    Q_on_trace, trace_on = mc_on_policy_eval_trace(ENV_ID, pi_star, num_episodes=20000, record_sa=record_key)
  # off-policy weighted IS (existing)
    Q_off_w, trace_off_w = mc_off_policy_eval_trace(ENV_ID, pi_star, behavior_eps=0.6, num_episodes=20000, record_sa=record_key)
    # off-policy ordinary IS (new) - use same behavior (uniform random) for dramatic variance
    Q_off_o, trace_off_o = mc_off_policy_eval_trace_ordinary(ENV_ID, pi_star, behavior_eps=0.6, num_episodes=20000, record_sa=record_key)

    def rolling_var(arr, window=500):
        out = np.empty(len(arr))
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            out[i] = np.nanvar(arr[start:i+1])
        return out

    window = 500
    rv_on = rolling_var(trace_on, window)
    rv_off_w = rolling_var(trace_off_w, window)
    rv_off_o = rolling_var(trace_off_o, window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Q estimate traces
    ax1.plot(trace_on, color='C0', alpha=0.9, label='On-policy')
    ax1.plot(np.linspace(0, len(trace_on)-1, len(trace_off_w)), trace_off_w, color='C1', alpha=0.7, label='Off-policy (weighted IS)')
    ax1.plot(np.linspace(0, len(trace_on)-1, len(trace_off_o)), trace_off_o, color='C2', alpha=0.5, label='Off-policy (ordinary IS)')
    ax1.set_ylabel(f'Q{record_key}')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # Bottom: rolling variance
    ax2.plot(rv_on, label='On-policy (rolling var)', color='C0')
    ax2.plot(np.linspace(0, len(rv_on)-1, len(rv_off_w)), rv_off_w, label='Off-policy weighted (rolling var)', color='C1')
    ax2.plot(np.linspace(0, len(rv_on)-1, len(rv_off_o)), rv_off_o, label='Off-policy ordinary (rolling var)', color='C2')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel(f'Rolling variance (window={window}) of Q{record_key}')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('blackjack_variance_cmp.png')
    plt.show()
    print("Saved variance comparison to blackjack_variance_cmp.png")