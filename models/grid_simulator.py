import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class PowerGridSimulator:
    def __init__(self, num_nodes=15, max_capacity=100.0, transmission_loss=0.05):
        self.num_nodes = num_nodes
        self.max_capacity = max_capacity
        self.transmission_loss = transmission_loss
        
        self.nodes = self.initialize_nodes()
        self.connections = self.initialize_connections()
        self.history = []
        
    def initialize_nodes(self):
        nodes = {}
        for i in range(self.num_nodes):
            nodes[i] = {
                'generation_capacity': np.random.uniform(10, 30),
                'current_generation': 0.0,
                'demand': 0.0,
                'storage_capacity': np.random.uniform(5, 20),
                'current_storage': np.random.uniform(0, 5),
                'voltage': 1.0,
                'frequency': 60.0,
                'is_blackout': False
            }
        return nodes
    
    def initialize_connections(self):
        connections = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if np.random.random() < 0.3:
                    capacity = np.random.uniform(5, 25)
                    connections[i, j] = capacity
                    connections[j, i] = capacity
        return connections
    
    def update_demands(self, demands):
        for i, demand in enumerate(demands):
            if i < self.num_nodes:
                self.nodes[i]['demand'] = demand
    
    def calculate_power_flow(self):
        power_flow = np.zeros((self.num_nodes, self.num_nodes))
        power_deficit = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            available_power = self.nodes[i]['current_generation'] + self.nodes[i]['current_storage']
            required_power = self.nodes[i]['demand']
            
            if available_power >= required_power:
                self.nodes[i]['current_storage'] = min(
                    self.nodes[i]['storage_capacity'],
                    self.nodes[i]['current_storage'] + (available_power - required_power) * 0.9
                )
            else:
                power_deficit[i] = required_power - available_power
        
        for i in range(self.num_nodes):
            if power_deficit[i] > 0:
                for j in range(self.num_nodes):
                    if self.connections[i, j] > 0 and power_deficit[i] > 0:
                        available_from_j = (self.nodes[j]['current_generation'] + 
                                          self.nodes[j]['current_storage'] - 
                                          self.nodes[j]['demand'])
                        
                        if available_from_j > 0:
                            transfer_amount = min(
                                power_deficit[i],
                                available_from_j,
                                self.connections[i, j]
                            )
                            
                            actual_transfer = transfer_amount * (1 - self.transmission_loss)
                            power_flow[j, i] = actual_transfer
                            
                            self.nodes[j]['current_storage'] -= transfer_amount
                            power_deficit[i] -= actual_transfer
        
        return power_flow, power_deficit
    
    def apply_actions(self, actions):
        for i, action in enumerate(actions):
            if i < self.num_nodes:
                generation_change = action * self.nodes[i]['generation_capacity']
                self.nodes[i]['current_generation'] = np.clip(
                    self.nodes[i]['current_generation'] + generation_change,
                    0, self.nodes[i]['generation_capacity']
                )
    
    def check_blackouts(self):
        blackout_nodes = []
        for i in range(self.num_nodes):
            total_supply = (self.nodes[i]['current_generation'] + 
                          self.nodes[i]['current_storage'])
            
            if total_supply < self.nodes[i]['demand'] * 0.8:
                self.nodes[i]['is_blackout'] = True
                blackout_nodes.append(i)
            else:
                self.nodes[i]['is_blackout'] = False
        
        return blackout_nodes
    
    def step(self, actions, demands):
        self.update_demands(demands)
        self.apply_actions(actions)
        
        power_flow, power_deficit = self.calculate_power_flow()
        blackout_nodes = self.check_blackouts()
        
        total_demand = sum(node['demand'] for node in self.nodes.values())
        total_supply = sum(node['current_generation'] for node in self.nodes.values())
        efficiency = total_supply / total_demand if total_demand > 0 else 1.0
        
        state = self.get_state()
        reward = self.calculate_reward(efficiency, blackout_nodes, power_deficit)
        done = len(blackout_nodes) > self.num_nodes * 0.3
        
        self.history.append({
            'total_demand': total_demand,
            'total_supply': total_supply,
            'efficiency': efficiency,
            'blackouts': len(blackout_nodes),
            'reward': reward
        })
        
        return state, reward, done, {
            'power_flow': power_flow,
            'power_deficit': power_deficit,
            'blackout_nodes': blackout_nodes,
            'efficiency': efficiency
        }
    
    def calculate_reward(self, efficiency, blackout_nodes, power_deficit):
        reward = efficiency * 10
        
        reward -= len(blackout_nodes) * 20
        
        total_deficit = np.sum(power_deficit)
        reward -= total_deficit * 0.1
        
        storage_utilization = sum(
            node['current_storage'] / node['storage_capacity'] 
            for node in self.nodes.values() if node['storage_capacity'] > 0
        ) / self.num_nodes
        reward += storage_utilization * 2
        
        generation_stability = -np.std([node['current_generation'] for node in self.nodes.values()])
        reward += generation_stability * 0.5
        
        return reward
    
    def get_state(self):
        state = []
        for i in range(self.num_nodes):
            node = self.nodes[i]
            state.extend([
                node['current_generation'],
                node['demand'],
                node['current_storage'],
                node['voltage'],
                node['frequency'],
                float(node['is_blackout'])
            ])
        return np.array(state)
    
    def reset(self):
        for node in self.nodes.values():
            node['current_generation'] = 0.0
            node['current_storage'] = np.random.uniform(0, 5)
            node['is_blackout'] = False
        
        initial_demands = np.random.uniform(10, 30, self.num_nodes)
        self.update_demands(initial_demands)
        
        return self.get_state()
    
    def get_grid_status(self):
        total_generation = sum(node['current_generation'] for node in self.nodes.values())
        total_demand = sum(node['demand'] for node in self.nodes.values())
        blackout_count = sum(1 for node in self.nodes.values() if node['is_blackout'])
        
        return {
            'total_generation': total_generation,
            'total_demand': total_demand,
            'supply_demand_ratio': total_generation / total_demand if total_demand > 0 else 0,
            'blackout_nodes': blackout_count,
            'grid_stability': 1.0 - (blackout_count / self.num_nodes)
        }