import argparse
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np

def train_demand_predictor():
    from data.data_loader import EnergyDataset
    from models.demand_predictor import DemandPredictor
    from training.trainer import DemandPredictorTrainer
    from config.settings import DEMAND_PREDICTION_CONFIG
    
    print("Training Demand Prediction Model...")
    
    dataset = EnergyDataset("data/energy_data.csv")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    input_dim = dataset.processed_data[0][0].shape[1] if dataset.processed_data else 2
    
    model = DemandPredictor(
        input_dim=input_dim,
        sequence_length=DEMAND_PREDICTION_CONFIG['sequence_length'],
        prediction_horizon=DEMAND_PREDICTION_CONFIG['prediction_horizon']
    )
    
    trainer = DemandPredictorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config={
            'epochs': 100,
            'learning_rate': DEMAND_PREDICTION_CONFIG['learning_rate']
        }
    )
    
    trainer.train()

def train_rl_agent():
    from environments.grid_env import SmartGridEnvironment
    from models.rl_agent import DDPGAgent
    from training.replay_buffer import ReplayBuffer
    from training.trainer import RLTrainer
    from config.settings import RL_CONFIG, TRAINING_CONFIG
    
    print("Training RL Agent for Grid Optimization...")
    
    env = SmartGridEnvironment()
    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=RL_CONFIG['hidden_dim'],
        lr_actor=RL_CONFIG['learning_rate'],
        lr_critic=RL_CONFIG['learning_rate'],
        gamma=RL_CONFIG['gamma'],
        tau=RL_CONFIG['tau']
    )
    
    replay_buffer = ReplayBuffer(capacity=RL_CONFIG['buffer_size'])
    
    trainer = RLTrainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        config=TRAINING_CONFIG
    )
    
    trainer.train()

def main():
    parser = argparse.ArgumentParser(description='Train EnergyGrid AI models')
    parser.add_argument('--model', type=str, choices=['demand', 'rl', 'both'], required=True,
                       help='Which model to train')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--episodes', type=int, default=10000, help='RL training episodes')
    parser.add_argument('--data-path', type=str, default='data/energy_data.csv', help='Data path for demand prediction')
    
    args = parser.parse_args()
    
    print(f"Training {args.model} model(s)...")
    
    if args.model in ['demand', 'both']:
        train_demand_predictor()
    
    if args.model in ['rl', 'both']:
        train_rl_agent()

if __name__ == "__main__":
    main()