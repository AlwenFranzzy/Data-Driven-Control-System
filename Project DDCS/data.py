"""
data_module.py
---------------
Data collection module for Modular Wastewater Simulation.

Contains:
- collect_data(): simulate random actions to generate dataset
  for system identification (MLP model training).
"""

import random
import pandas as pd
from environment import WastewaterEnv

# ----------------------------
# Data collection function
# ----------------------------
def collect_data(env, n_steps=2000):
    """
    Run the wastewater environment for 'n_steps' steps using random actions.
    Collect state transitions for system identification.

    Parameters
    ----------
    env : WastewaterEnv
        Environment instance from env_module.
    n_steps : int
        Number of simulation steps (default 2000)

    Returns
    -------
    df : pandas.DataFrame
        Columns: ['level','hour','day','inflow','action','next_level']
    """
    state = env.reset()
    records = []

    for _ in range(n_steps):
        # Random action: 0=HOLD, 1=RELEASE, 2=WATER_PLANTS
        action = random.choice([0, 1, 2])

        # Step environment
        next_state, _, _, info = env.step(action)

        # Record transition (state + action -> next level)
        records.append([
            state[0],  # current tank level
            state[1],  # hour
            state[2],  # day
            info['inflow'],  # inflow
            action,
            next_state[0]  # next tank level
        ])

        # Update state
        state = next_state

    # Create DataFrame
    df = pd.DataFrame(records, columns=['level', 'hour', 'day', 'inflow', 'action', 'next_level'])
    return df


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Example: quick test of data collection
    env = WastewaterEnv(tank_capacity=100.0, seed=42)
    df = collect_data(env, n_steps=1000)
    print("✅ Data collected:", len(df), "rows")
    print(df.head())
