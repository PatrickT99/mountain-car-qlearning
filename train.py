"""
Mountain Car Training Script (Q-Learning)

This script trains a Q-Learning agent to solve the Gymnasium MountainCar-v0 environment.
It uses state discretisation (bucketing) to handle the continuous observation space.

Key Features:
- Discretisation of continuous state space into a grid.
- Epsilon-Greedy strategy with exponential decay.
- Optimistic Initialisation (using np.zeros) to encourage exploration.
- Moving average visualisation of training performance.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# --- Hyperparameters & Constants ---
LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPISODES = 14_000
SHOW_EVERY = 2_000

# Exploration settings
START_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY_RATE = 0.9995

# Discretisation settings
NUM_BINS = [30, 30]  # [Position, Velocity] resolution

MODEL_FILENAME = "qt.npy"
GRAPH_FILENAME = "Total_Rewards_Per_Episode.png"


def get_discrete_state(state: np.ndarray, env: gym.Env, bins: List[int], bin_width: np.ndarray) -> Tuple[int, ...]:
    """
    Converts continuous state into discrete tuple coordinates.
    """
    discrete_state = (state - env.observation_space.low) / bin_width
    return tuple(np.clip(discrete_state.astype(int), 0, np.array(bins) - 1))


def train_agent():
    """
    Main training loop for the Q-Learning agent.
    """
    # Initialise Environments
    env = gym.make("MountainCar-v0")
    render_env = gym.make("MountainCar-v0", render_mode="human")

    # Calculate bin width
    bin_width = (env.observation_space.high - env.observation_space.low) / NUM_BINS

    # Initialise Q-table
    # Using np.zeros (Optimistic Initialisation) as rewards are always negative.
    # This encourages the agent to explore unvisited states (value 0) over known bad states (value -1).
    q_table = np.zeros(NUM_BINS + [env.action_space.n])

    epsilon = START_EPSILON
    rewards_per_episode = []

    print(f"Starting training for {EPISODES} episodes...")

    for episode in range(1, EPISODES + 1):
        # Render periodically to check progress
        active_env = render_env if episode % SHOW_EVERY == 0 else env

        state, _ = active_env.reset()
        discrete_state = get_discrete_state(state, active_env, NUM_BINS, bin_width)

        done = False
        total_reward = 0

        while not done:
            # Epsilon-Greedy Action Selection
            if np.random.random() < epsilon:
                action = active_env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[discrete_state])  # Exploit

            new_state, reward, terminated, truncated, _ = active_env.step(action)
            new_discrete_state = get_discrete_state(new_state, active_env, NUM_BINS, bin_width)
            done = terminated or truncated

            # Bellman Equation Update
            current_q = q_table[discrete_state + (action,)]
            max_future_q = np.max(q_table[new_discrete_state])

            if terminated:
                # Goal reached: no future Q-value
                new_present_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * reward
            else:
                # Standard update
                new_present_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[discrete_state + (action,)] = new_present_q
            discrete_state = new_discrete_state
            total_reward += reward

        # Decay Epsilon
        epsilon = max(epsilon * DECAY_RATE, MIN_EPSILON)
        rewards_per_episode.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode: {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.4f}, Success: {terminated}")

    # Cleanup and Save
    env.close()
    render_env.close()

    np.save(MODEL_FILENAME, q_table)
    print(f"\nTraining complete. Q-table saved to {MODEL_FILENAME}")

    return rewards_per_episode


def plot_results(rewards: List[float], window_size: int = 200):
    """Calculates moving average and saves the training graph."""
    if len(rewards) < window_size:
        print("Not enough episodes to calculate moving average.")
        return

    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

    plt.figure(figsize=(10, 5))
    plt.plot(moving_avg, label=f"Moving Avg (Window {window_size})")
    plt.title("Mountain Car - Agent Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(GRAPH_FILENAME)
    print(f"Training graph saved to {GRAPH_FILENAME}")
    print(f"\nMax Reward: {np.max(rewards)}, Min Reward: {np.min(rewards)}, Average Reward: {np.average(rewards)}")

    # plt.show() # Uncomment if running locally with a display


if __name__ == "__main__":
    rewards = train_agent()
    plot_results(rewards)