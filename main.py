import argparse
import torch
import numpy as np
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='EnergyGrid AI: Smart Grid Optimization System')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'api', 'demo'], 
                       required=True, help='Operation mode')
    parser.add_argument('--config', type=str, default='config/settings.py', help='Config file path')
    parser.add_argument('--model-path', type=str, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--render', action='store_true', help='Render environment during training')
    
    args = parser.parse_args()
    
    print("EnergyGrid AI System Starting...")
    print(f"Mode: {args.mode}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'evaluate':
        evaluate_model(args)
    elif args.mode == 'api':
        start_api_server(args)
    elif args.mode == 'demo':
        run_demo(args)

def train_model(args):
    from environments.grid_env import SmartGridEnvironment
    from models.rl_agent import DDPGAgent
    from training.replay_buffer import ReplayBuffer
    from training.trainer import RLTrainer
    from config.settings import RL_CONFIG, TRAINING_CONFIG
    
    print("Initializing training environment...")
    
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
    
    training_config = {
        'episodes': args.episodes or TRAINING_CONFIG['episodes'],
        'max_steps': TRAINING_CONFIG['max_steps'],
        'warmup_steps': TRAINING_CONFIG['warmup_steps'],
        'update_interval': TRAINING_CONFIG['update_interval'],
        'eval_interval': TRAINING_CONFIG['eval_interval'],
        'save_interval': TRAINING_CONFIG['save_interval'],
        'batch_size': RL_CONFIG['batch_size']
    }
    
    trainer = RLTrainer(env, agent, replay_buffer, training_config)
    trainer.train()

def evaluate_model(args):
    from environments.grid_env import SmartGridEnvironment
    from models.rl_agent import DDPGAgent
    from utils.visualization import GridVisualizer
    
    print("Initializing evaluation environment...")
    
    env = SmartGridEnvironment()
    agent = DDPGAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    
    if args.model_path and Path(args.model_path).exists():
        agent.load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        print("No model specified or model not found. Using untrained agent.")
    
    visualizer = GridVisualizer()
    
    total_rewards = []
    efficiencies = []
    
    for episode in range(10):
        state = env.reset()
        episode_reward = 0
        episode_data = []
        
        for step in range(1000):
            action = agent.select_action(state, noise_scale=0.0)
            next_state, reward, done, info = env.step(action)
            
            episode_data.append({
                'step': step,
                'state': state.copy(),
                'action': action.copy(),
                'reward': reward,
                'info': info
            })
            
            state = next_state
            episode_reward += reward
            
            if args.render:
                env.render()
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        efficiencies.append(info.get('efficiency', 0))
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Efficiency = {info.get('efficiency', 0):.3f}")
        
        if episode == 0:
            visualizer.plot_grid_status(env.grid_simulator, save_path=f"evaluation_episode_{episode}.png")
    
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Efficiency: {np.mean(efficiencies):.3f} ± {np.std(efficiencies):.3f}")
    print(f"Best Reward: {np.max(total_rewards):.2f}")
    print(f"Worst Reward: {np.min(total_rewards):.2f}")

def start_api_server(args):
    from api.app import app
    from config.settings import API_CONFIG
    
    print(f"Starting EnergyGrid AI API server on {API_CONFIG['host']}:{API_CONFIG['port']}")
    print("API endpoints available:")
    print("  GET  /api/grid/status     - Get current grid status")
    print("  POST /api/grid/optimize   - Optimize grid operations")
    print("  POST /api/demand/predict  - Predict energy demand")
    print("  GET  /api/system/health   - System health check")
    
    app.run(
        host=API_CONFIG['host'],
        port=API_CONFIG['port'],
        debug=API_CONFIG['debug']
    )

def run_demo(args):
    from environments.grid_env import SmartGridEnvironment
    from models.rl_agent import DDPGAgent
    from utils.visualization import GridVisualizer
    
    print("Running EnergyGrid AI Demo...")
    
    env = SmartGridEnvironment()
    visualizer = GridVisualizer()
    
    print("Demo: Smart Grid Optimization with AI")
    print("=" * 50)
    
    state = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = np.random.uniform(0, 1, env.action_dim)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        
        if step % 10 == 0:
            print(f"Step {step}: Reward = {reward:.2f}, Efficiency = {info.get('efficiency', 0):.3f}")
            print(f"Blackout nodes: {len(info.get('blackout_nodes', []))}")
        
        state = next_state
        
        if done:
            print("Grid instability detected! Ending demo.")
            break
    
    print("\nDemo Summary:")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Efficiency: {info.get('efficiency', 0):.3f}")
    print(f"Total Blackout Events: {sum(1 for node in env.grid_simulator.nodes.values() if node['is_blackout'])}")
    
    visualizer.plot_grid_status(env.grid_simulator, save_path="demo_results.png")
    print("Grid visualization saved as 'demo_results.png'")

if __name__ == "__main__":
    main()