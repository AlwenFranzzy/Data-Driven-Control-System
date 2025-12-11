"""
model_module.py
----------------
System Identification Module using MLP.

Contains:
- train_mlp(): trains a neural network (MLPRegressor)
  to model the wastewater system dynamics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# ----------------------------
# MLP Training Function
# ----------------------------
def train_mlp(df):
    """
    Train a Multi-Layer Perceptron (MLP) model to predict next tank level
    based on current state and action.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing columns:
        ['level','hour','day','inflow','action','next_level']

    Returns
    -------
    model : MLPRegressor
        Trained MLP model.
    scaler : StandardScaler
        Fitted scaler for input normalization.
    score : float
        R^2 test score of model on held-out test set.
    """

    # === 1. Pisahkan fitur dan target ===
    X = df[['level', 'hour', 'day', 'inflow', 'action']].values
    y = df['next_level'].values

    # === 2. Standarisasi fitur (agar training stabil) ===
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # === 3. Split data menjadi train/test ===
    Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2, random_state=1)

    # === 4. Definisikan arsitektur MLP ===
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=1
    )

    # === 5. Training ===
    model.fit(Xtr, ytr)

    # === 6. Evaluasi (R² score) ===
    score = model.score(Xte, yte)

    print(f"✅ MLP training done. Test R² = {score:.4f}")
    return model, scaler, score


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Example test using synthetic dataset
    from environment import WastewaterEnv
    from data import collect_data

    env = WastewaterEnv(tank_capacity=100.0, seed=42)
    df = collect_data(env, n_steps=2000)

    model, scaler, score = train_mlp(df)
    print("Model R² score:", score)
