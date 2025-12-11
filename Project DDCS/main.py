"""
main_module.py
---------------
Main script for running the full Modular Data-Driven Control System (DDCS)
including:
1. Environment setup
2. Data collection
3. MLP system identification
4. Q-learning agent training
5. Visualization of system performance
"""

import numpy as np
import matplotlib.pyplot as plt

from environment import WastewaterEnv
from data import collect_data
from model import train_mlp
from qlearning_module import q_learning_model_assisted

# ----------------------------
# Visualization helper
# ----------------------------
def plot_results(reward_history):
    """
    Visualize tank level evolution and reward per episode.

    Parameters
    ----------
    reward_history : list
        Average reward per episode.
    """
    plt.figure(figsize=(12, 5))

    # === Subplot : Reward per episode ===
    plt.subplot()
    plt.plot(reward_history, label="Average Reward", color='blue', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Agent Learning Progress")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    print("🚀 Starting Modular Data-Driven Control System...")

    # === 1. Initialize environment ===
    env = WastewaterEnv(tank_capacity=100.0, seed=42)
    print("✅ Environment initialized.")

    # === 2. Collect data for system identification ===
    df = collect_data(env, n_steps=2000)
    print(f"✅ Data collected: {len(df)} samples")

    # === 3. Train MLP model ===
    model, scaler, score = train_mlp(df)
    print(f"✅ MLP model trained with R² = {score:.4f}")

    # === 4. Train Q-learning agent ===
    q_table, avg_rewards = q_learning_model_assisted(env, model, scaler, episodes=2000)
    print("✅ Q-learning training completed.")

    # === 5. Visualization ===
    plot_results(avg_rewards)
    print("📊 Visualization complete.")
