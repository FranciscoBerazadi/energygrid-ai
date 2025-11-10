import gym
from gym import spaces
import numpy as np

class SmartGridEnvironment(gym.Env):
    def __init__(self, num_nodes=15, max_steps=1000):
        super(SmartGridEnvironment, self).__init__()
        
        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.current_step = 0
        
        self.state_dim = num_nodes * 6
        self.action_dim = num_nodes
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        
        from models.grid_simulator import PowerGridSimulator
        self.grid_simulator = PowerGridSimulator(num_nodes=num_nodes)
        
        from models.demand_predictor import DemandPredictor
        self.demand_predictor = DemandPredictor(input_dim=2, sequence_length=24, prediction_horizon=12)
        
        self.demand_history = []
        self.episode_history = []
    
    def reset(self):
        self.current_step = 0
        self.demand_history = []
        self.episode_history = []
        
        state = self.grid_simulator.reset()
        
        initial_demands = self._generate_demands()
        self.demand_history.extend(initial_demands)
        
        return state
    
    def step(self, actions):
        self.current_step += 1
        
        current_demands = self._generate_demands()
        self.demand_history.extend(current_demands)
        
        if len(self.demand_history) > 1000:
            self.demand_history = self.demand_history[-1000:]
        
        state, reward, done, info = self.grid_simulator.step(actions, current_demands)
        
        self.episode_history.append({
            'step': self.current_step,
            'actions': actions.copy(),
            'reward': reward,
            'demands': current_demands.copy(),
            'efficiency': info['efficiency'],
            'blackouts': len(info['blackout_nodes'])
        })
        
        if self.current_step >= self.max_steps:
            done = True
        
        info['grid_status'] = self.grid_simulator.get_grid_status()
        info['demand_prediction'] = self._predict_future_demand()
        
        return state, reward, done, info
    
    def _generate_demands(self):
        base_pattern = 50 + 20 * np.sin(2 * np.pi * self.current_step / 24)
        noise = np.random.normal(0, 5, self.num_nodes)
        
        node_variations = np.array([
            np.sin(2 * np.pi * (i / self.num_nodes) + self.current_step / 100) * 10 
            for i in range(self.num_nodes)
        ])
        
        demands = base_pattern + node_variations + noise
        return np.maximum(10, demands)
    
    def _predict_future_demand(self):
        if len(self.demand_history) < 24:
            return np.zeros(12)
        
        recent_demands = self.demand_history[-24:]
        recent_temperatures = np.random.normal(20, 5, 24)
        
        input_sequence = np.column_stack([recent_demands, recent_temperatures])
        
        try:
            prediction = self.demand_predictor.predict(input_sequence)
            return prediction
        except:
            return np.full(12, np.mean(recent_demands))
    
    def render(self, mode='human'):
        status = self.grid_simulator.get_grid_status()
        
        print(f"Step: {self.current_step}")
        print(f"Total Generation: {status['total_generation']:.2f} MW")
        print(f"Total Demand: {status['total_demand']:.2f} MW")
        print(f"Efficiency: {status['supply_demand_ratio']:.2%}")
        print(f"Blackout Nodes: {status['blackout_nodes']}/{self.num_nodes}")
        print(f"Grid Stability: {status['grid_stability']:.2%}")
        print("-" * 50)
    
    def get_episode_statistics(self):
        if not self.episode_history:
            return {}
        
        rewards = [step['reward'] for step in self.episode_history]
        efficiencies = [step['efficiency'] for step in self.episode_history]
        blackouts = [step['blackouts'] for step in self.episode_history]
        
        return {
            'total_reward': sum(rewards),
            'average_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'average_efficiency': np.mean(efficiencies),
            'total_blackouts': sum(blackouts),
            'max_blackouts': np.max(blackouts),
            'episode_length': len(self.episode_history)
        }