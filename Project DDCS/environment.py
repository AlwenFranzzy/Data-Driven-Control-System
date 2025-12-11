"""
env_module.py
--------------
Environment module for Modular Wastewater Simulation

Contains:
- SensorModule: simulates daily inflow and demand patterns.
- RewardModule: defines reward function for reinforcement learning.
- WastewaterEnv: main environment class combining sensor + reward.
"""

import numpy as np
import random

# ----------------------------
# Sensor Module
# ----------------------------
class SensorModule:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self, hour, day):
        """
        Generate inflow (L) and demand (L) for given hour and day.
        Pattern:
        - Shower: morning & evening peaks
        - Laundry: midday (weekend heavier)
        - AC: small inflow midday
        """
        # Shower pattern
        shower = 0.0
        if 6 <= hour <= 8:
            shower = random.uniform(5, 15)
        elif 19 <= hour <= 21:
            shower = random.uniform(3, 10)

        # Laundry pattern
        laundry = 0.0
        if 13 <= hour <= 15:
            laundry = random.uniform(10, 20) if day >= 5 else random.uniform(0, 5)

        # AC condensation
        ac = 0.0
        if 11 <= hour <= 17:
            ac = random.uniform(1, 4)

        total_inflow = shower + laundry + ac

        # Demand pattern (household usage)
        demand = 0.0
        if 5 <= hour <= 22 and random.random() < 0.3:
            demand = random.uniform(3, 7)

        return total_inflow, demand

# ----------------------------
# Reward Module
# ----------------------------
class RewardModule:
    def __init__(self, tank_capacity):
        self.cap = float(tank_capacity)

    def compute(self, fulfilled_demand, wasted_by_action, wasted_by_overflow, water_used_plants):
        """
        Compute reward in range [-1, 1]
        Components:
        + demand fulfillment (+)
        + plant watering (+ small)
        - manual release (−)
        - overflow (− large)
        """
        r_fulfill = 0.6 * (1.0 if fulfilled_demand > 0 else 0.0)
        r_plants = 0.2 * (water_used_plants / 20.0)
        r_release = -0.3 * (wasted_by_action / 20.0)
        r_overflow = -1.0 * (wasted_by_overflow / self.cap)
        reward = r_fulfill + r_plants + r_release + r_overflow
        return max(-1.0, min(1.0, reward))  # clip to [-1, 1]

# ----------------------------
# Environment (Wastewater)
# ----------------------------
class WastewaterEnv:
    def __init__(self, tank_capacity=100.0, release_amount=20.0, watering_amount=15.0, seed=None):
        self.tank_capacity = float(tank_capacity)
        self.release_amount = float(release_amount)
        self.watering_amount = float(watering_amount)
        self.sensor = SensorModule(seed=seed)
        self.reward_module = RewardModule(self.tank_capacity)
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.tank_level = 0.0
        self.hour = 0
        self.day = 0
        return (self.tank_level, self.hour, self.day)

    def step(self, action):
        """
        action: 0 = HOLD, 1 = RELEASE (waste), 2 = WATER_PLANTS
        Returns:
        next_state, reward, done, info
        """
        inflow, demand = self.sensor.generate(self.hour, self.day)

        # Supply water for demand
        water_supplied = min(self.tank_level, demand)
        self.tank_level -= water_supplied

        # Add inflow to tank
        self.tank_level += inflow

        # Initialize
        wasted_by_action = 0.0
        water_used_plants = 0.0

        # Execute action
        if action == 1:  # RELEASE
            water_to_release = min(self.tank_level, self.release_amount)
            self.tank_level -= water_to_release
            wasted_by_action = water_to_release

        elif action == 2:  # WATER_PLANTS
            water_to_use = min(self.tank_level, self.watering_amount)
            self.tank_level -= water_to_use
            water_used_plants = water_to_use

        # Overflow check
        wasted_by_overflow = 0.0
        if self.tank_level > self.tank_capacity:
            wasted_by_overflow = self.tank_level - self.tank_capacity
            self.tank_level = self.tank_capacity

        # Compute reward
        reward = self.reward_module.compute(
            water_supplied, wasted_by_action, wasted_by_overflow, water_used_plants
        )

        # Advance time
        self.hour = (self.hour + 1) % 24
        if self.hour == 0:
            self.day = (self.day + 1) % 7

        next_state = (self.tank_level, self.hour, self.day)
        info = {
            "inflow": inflow,
            "demand": demand,
            "wasted_overflow": wasted_by_overflow,
            "wasted_action": wasted_by_action,
            "water_used_plants": water_used_plants,
            "water_supplied": water_supplied,
        }

        done = False
        return next_state, reward, done, info
