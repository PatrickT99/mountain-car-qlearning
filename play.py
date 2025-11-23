"""
Mountain Car Inference Script

This script loads a pre-trained Q-learning model (qt.npy) and visualises the
agent's performance in the Gymnasium MountainCar-v0 environment.

It automatically detects the quantisation resolution (grid size) from the
saved Q-table, allowing it to work with models of varying resolutions
(e.g., 20x20, 40x40, 50x50) without manual configuration.
"""

import gymnasium as gym
import numpy as np
import os

# Constants
MODEL_PATH = "qt.npy"
EPISODES_TO_PLAY = 5


def load_q_table(filepath: str) -> np.ndarray:
    """Loads the Q-table and validates file existence."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file '{filepath}' not found. Please train the agent first.")
    return np.load(filepath)


def get_discrete_state(state: np.ndarray, env: gym.Env, bins: tuple) -> tuple:
    """
    Converts the continuous state (position, velocity) into a discrete tuple index.

    Args:
        state: The continuous state from the environment.
        env: The gymnasium environment (needed for bounds).
        bins: A tuple representing the grid size (e.g., (40, 40)).

    Returns:
        A tuple of integers representing the grid coordinates.
    """
    # Calculate the size of each bucket based on the grid dimensions
    bin_window = (env.observation_space.high - env.observation_space.low) / bins

    discrete_state = (state - env.observation_space.low) / bin_window

    # Clip to ensure indices stay within the grid (0 to bins-1)
    return tuple(np.clip(discrete_state.astype(int), 0, np.array(bins) - 1))


def play_episodes(env: gym.Env, q_table: np.ndarray, episodes: int = 5):
    """Runs the agent in the environment using the learned Q-table."""

    # Infer grid size from the Q-table shape (excluding the action dimension)
    # Shape is typically (bin_x, bin_y, actions), so we take [:-1]
    num_bins = q_table.shape[:-1]
    print(f"Loaded Q-Table with grid size: {num_bins}")

    for episode in range(episodes):
        state, _ = env.reset()
        discrete_state = get_discrete_state(state, env, num_bins)
        done = False
        total_reward = 0

        while not done:
            # Greedy Action: Always choose the best known action
            action = np.argmax(q_table[discrete_state])

            new_state, reward, terminated, truncated, _ = env.step(action)

            discrete_state = get_discrete_state(new_state, env, num_bins)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1}: Reward = {total_reward}, Success = {terminated}")


if __name__ == "__main__":
    # Initialise Environment
    # render_mode="human" allows us to watch the car
    env = gym.make("MountainCar-v0", render_mode="human")

    try:
        q_table = load_q_table(MODEL_PATH)
        play_episodes(env, q_table, EPISODES_TO_PLAY)
    except FileNotFoundError as e:
        print(e)
    finally:
        env.close()