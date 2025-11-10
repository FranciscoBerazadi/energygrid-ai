import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

GRID_CONFIG = {
    "num_nodes": 15,
    "max_power_capacity": 100.0,
    "min_power_capacity": 0.0,
    "transmission_loss": 0.05,
    "base_demand": 50.0,
    "peak_demand": 90.0
}

RL_CONFIG = {
    "state_dim": 30,
    "action_dim": 15,
    "hidden_dim": 256,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 128,
    "buffer_size": 100000
}

TRAINING_CONFIG = {
    "episodes": 10000,
    "max_steps": 1000,
    "warmup_steps": 1000,
    "update_interval": 50,
    "eval_interval": 100,
    "save_interval": 1000
}

DEMAND_PREDICTION_CONFIG = {
    "sequence_length": 24,
    "prediction_horizon": 12,
    "lstm_units": 128,
    "dropout_rate": 0.2,
    "learning_rate": 0.0005
}

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False
}