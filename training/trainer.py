import torch
import numpy as np
import pandas as pd
from pathlib import Path
import time

class RLTrainer:
    def __init__(self, env, agent, replay_buffer, config):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = config
        
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_efficiencies = []
        self.best_reward = -np.inf
        
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
    
    def train(self):
        print("Starting RL training for Smart Grid Optimization...")
        print(f"Training for {self.config['episodes']} episodes")
        print(f"Device: {self.agent.device}")
        
        start_time = time.time()
        
        for episode in range(self.config['episodes']):
            state = self.env.reset()
            episode_reward = 0
            episode_losses = []
            steps = 0
            
            for step in range(self.config['max_steps']):
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                if len(self.replay_buffer) > self.config['warmup_steps']:
                    if steps % self.config['update_interval'] == 0:
                        batch = self.replay_buffer.sample(self.config['batch_size'])
                        critic_loss, actor_loss = self.agent.update_parameters(batch)
                        episode_losses.append((critic_loss, actor_loss))
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            self.episode_efficiencies.append(info.get('efficiency', 0))
            
            if episode_losses:
                avg_critic_loss = np.mean([loss[0] for loss in episode_losses])
                avg_actor_loss = np.mean([loss[1] for loss in episode_losses])
                self.episode_losses.append((avg_critic_loss, avg_actor_loss))
            else:
                self.episode_losses.append((0, 0))
            
            if episode % self.config['eval_interval'] == 0:
                eval_reward = self.evaluate()
                self.print_progress(episode, eval_reward, episode_reward)
                
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    self.save_checkpoint(episode, is_best=True)
            
            if episode % self.config['save_interval'] == 0:
                self.save_checkpoint(episode)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        self.save_training_results()
    
    def evaluate(self, num_episodes=5):
        total_reward = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.config['max_steps']):
                action = self.agent.select_action(state, noise_scale=0.0)
                next_state, reward, done, info = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            total_reward += episode_reward
        
        return total_reward / num_episodes
    
    def print_progress(self, episode, eval_reward, train_reward):
        avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        avg_efficiency = np.mean(self.episode_efficiencies[-100:]) if self.episode_efficiencies else 0
        
        if self.episode_losses and episode > 0:
            critic_loss, actor_loss = self.episode_losses[-1]
            loss_info = f"Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}"
        else:
            loss_info = "Loss: N/A"
        
        print(f"Episode {episode:5d} | "
              f"Train Reward: {train_reward:8.2f} | "
              f"Eval Reward: {eval_reward:8.2f} | "
              f"Avg Reward: {avg_reward:8.2f} | "
              f"Efficiency: {avg_efficiency:.3f} | "
              f"{loss_info}")
    
    def save_checkpoint(self, episode, is_best=False):
        checkpoint = {
            'episode': episode,
            'agent_state': self.agent.save_model(self.results_dir / f"checkpoint_episode_{episode}.pth"),
            'replay_buffer': self.replay_buffer.get_state(),
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'episode_efficiencies': self.episode_efficiencies,
            'best_reward': self.best_reward
        }
        
        if is_best:
            torch.save(checkpoint, self.results_dir / "best_model.pth")
            print(f"New best model saved with reward: {self.best_reward:.2f}")
        
        torch.save(checkpoint, self.results_dir / "latest_model.pth")
    
    def save_training_results(self):
        results = {
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
            'episode_efficiencies': self.episode_efficiencies,
            'best_reward': self.best_reward
        }
        
        df = pd.DataFrame({
            'episode': range(len(self.episode_rewards)),
            'reward': self.episode_rewards,
            'critic_loss': [loss[0] for loss in self.episode_losses],
            'actor_loss': [loss[1] for loss in self.episode_losses],
            'efficiency': self.episode_efficiencies
        })
        
        df.to_csv(self.results_dir / "training_results.csv", index=False)
        
        summary = {
            'total_episodes': len(self.episode_rewards),
            'final_avg_reward': np.mean(self.episode_rewards[-100:]),
            'best_reward': self.best_reward,
            'final_efficiency': np.mean(self.episode_efficiencies[-100:]),
            'training_time': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
        
        with open(self.results_dir / "training_summary.json", 'w') as f:
            import json
            json.dump(summary, f, indent=2)

class DemandPredictorTrainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = np.inf
        
    def train(self):
        print("Training Demand Prediction Model...")
        
        for epoch in range(self.config['epochs']):
            train_loss = self._train_epoch()
            val_loss = self._validate_epoch()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.model.save_model("best_demand_predictor.pth")
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:4d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        print(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")
        self.save_training_results()
    
    def _train_epoch(self):
        self.model.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.model.device), target.to(self.model.device)
            
            self.model.optimizer.zero_grad()
            output = self.model.model(data)
            loss = self.model.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
            self.model.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def _validate_epoch(self):
        return self.model.validate(self.val_loader)
    
    def save_training_results(self):
        results = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        df = pd.DataFrame({
            'epoch': range(len(self.train_losses)),
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        })
        
        df.to_csv("demand_prediction_training.csv", index=False)