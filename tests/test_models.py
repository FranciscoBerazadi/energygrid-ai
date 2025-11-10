import torch
import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

class TestEnergyGridModels(unittest.TestCase):
    def test_rl_agent_initialization(self):
        from models.rl_agent import DDPGAgent
        
        state_dim = 30
        action_dim = 15
        
        agent = DDPGAgent(state_dim, action_dim)
        
        self.assertEqual(agent.actor.fc1.in_features, state_dim)
        self.assertEqual(agent.actor.action_out.out_features, action_dim)
        self.assertEqual(agent.critic.fc1.in_features, state_dim + action_dim)
    
    def test_demand_predictor_forward(self):
        from models.demand_predictor import LSTMDemandPredictor
        
        batch_size = 32
        sequence_length = 24
        input_dim = 5
        output_dim = 12
        
        model = LSTMDemandPredictor(input_dim=input_dim, output_dim=output_dim)
        
        test_input = torch.randn(batch_size, sequence_length, input_dim)
        output = model(test_input)
        
        self.assertEqual(output.shape, (batch_size, output_dim))
    
    def test_grid_simulator_step(self):
        from models.grid_simulator import PowerGridSimulator
        
        simulator = PowerGridSimulator(num_nodes=5)
        
        initial_state = simulator.reset()
        self.assertEqual(len(initial_state), 5 * 6)
        
        actions = np.random.uniform(0, 1, 5)
        demands = np.random.uniform(10, 50, 5)
        
        next_state, reward, done, info = simulator.step(actions, demands)
        
        self.assertEqual(len(next_state), 5 * 6)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIn('efficiency', info)
        self.assertIn('blackout_nodes', info)
    
    def test_replay_buffer(self):
        from training.replay_buffer import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=100)
        
        state = np.random.randn(30)
        action = np.random.randn(15)
        reward = 1.0
        next_state = np.random.randn(30)
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        self.assertEqual(len(buffer), 1)
        
        batch = buffer.sample(1)
        self.assertEqual(batch['states'].shape, (1, 30))
        self.assertEqual(batch['actions'].shape, (1, 15))
    
    def test_data_loader(self):
        from data.data_loader import EnergyDataset
        
        dataset = EnergyDataset("test_data.csv", sequence_length=24, prediction_horizon=12)
        
        self.assertGreater(len(dataset), 0)
        
        sample_sequence, sample_target = dataset[0]
        
        self.assertEqual(sample_sequence.shape[0], 24)
        self.assertEqual(sample_target.shape[0], 12)

if __name__ == '__main__':
    unittest.main()