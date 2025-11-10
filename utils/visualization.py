import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class GridVisualizer:
    def __init__(self):
        self.fig_size = (12, 8)
        plt.style.use('seaborn-v0_8')
    
    def plot_grid_status(self, grid_simulator, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        nodes = list(range(grid_simulator.num_nodes))
        generations = [grid_simulator.nodes[i]['current_generation'] for i in nodes]
        demands = [grid_simulator.nodes[i]['demand'] for i in nodes]
        storages = [grid_simulator.nodes[i]['current_storage'] for i in nodes]
        blackouts = [grid_simulator.nodes[i]['is_blackout'] for i in nodes]
        
        axes[0, 0].bar(nodes, generations, alpha=0.7, label='Generation', color='green')
        axes[0, 0].bar(nodes, demands, alpha=0.7, label='Demand', color='red')
        axes[0, 0].set_title('Generation vs Demand by Node')
        axes[0, 0].set_xlabel('Node ID')
        axes[0, 0].set_ylabel('Power (MW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].bar(nodes, storages, color='orange', alpha=0.7)
        axes[0, 1].set_title('Energy Storage by Node')
        axes[0, 1].set_xlabel('Node ID')
        axes[0, 1].set_ylabel('Storage (MWh)')
        axes[0, 1].grid(True, alpha=0.3)
        
        blackout_colors = ['red' if blackout else 'green' for blackout in blackouts]
        axes[1, 0].bar(nodes, [1] * len(nodes), color=blackout_colors, alpha=0.7)
        axes[1, 0].set_title('Node Status (Green: Normal, Red: Blackout)')
        axes[1, 0].set_xlabel('Node ID')
        axes[1, 0].set_ylabel('Status')
        axes[1, 0].set_yticks([])
        
        status = grid_simulator.get_grid_status()
        metrics = ['Total Generation', 'Total Demand', 'Efficiency', 'Blackout Nodes']
        values = [status['total_generation'], status['total_demand'], 
                 status['supply_demand_ratio'], status['blackout_nodes']]
        
        axes[1, 1].bar(metrics, values, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
        axes[1, 1].set_title('Grid Overview Metrics')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_progress(self, rewards, losses, efficiencies, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(len(rewards))
        
        axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        if losses:
            axes[0, 1].plot(episodes, losses, 'r-', alpha=0.7, linewidth=1)
            axes[0, 1].set_title('Training Losses')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        if efficiencies:
            axes[1, 0].plot(episodes, efficiencies, 'g-', alpha=0.7, linewidth=1)
            axes[1, 0].set_title('Grid Efficiency')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Efficiency')
            axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(rewards, bins=50, alpha=0.7, color='purple')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, grid_simulator, episode_history):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Power Flow Network', 'Supply-Demand Balance', 
                          'Node Status', 'Performance Metrics'),
            specs=[[{"type": "scatter"}, {"type": "xy"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        node_x = np.random.rand(grid_simulator.num_nodes)
        node_y = np.random.rand(grid_simulator.num_nodes)
        
        for i in range(grid_simulator.num_nodes):
            for j in range(i + 1, grid_simulator.num_nodes):
                if grid_simulator.connections[i, j] > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=[node_x[i], node_x[j], None],
                            y=[node_y[i], node_y[j], None],
                            mode='lines',
                            line=dict(width=grid_simulator.connections[i, j] / 5, color='gray'),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
        
        node_colors = []
        for i in range(grid_simulator.num_nodes):
            if grid_simulator.nodes[i]['is_blackout']:
                node_colors.append('red')
            else:
                utilization = grid_simulator.nodes[i]['current_generation'] / grid_simulator.nodes[i]['generation_capacity']
                node_colors.append(f'rgba(0, 255, 0, {utilization})')
        
        fig.add_trace(
            go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(size=20, color=node_colors),
                text=[f"Node {i}<br>Gen: {grid_simulator.nodes[i]['current_generation']:.1f}<br>Demand: {grid_simulator.nodes[i]['demand']:.1f}" 
                      for i in range(grid_simulator.num_nodes)],
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=1
        )
        
        if episode_history:
            steps = [step['step'] for step in episode_history]
            supplies = [sum(step['demands']) * step['efficiency'] for step in episode_history]
            demands = [sum(step['demands']) for step in episode_history]
            
            fig.add_trace(
                go.Scatter(x=steps, y=supplies, name='Supply', line=dict(color='green')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=steps, y=demands, name='Demand', line=dict(color='red')),
                row=1, col=2
            )
        
        node_status = ['Normal' if not grid_simulator.nodes[i]['is_blackout'] else 'Blackout' 
                      for i in range(grid_simulator.num_nodes)]
        status_counts = pd.Series(node_status).value_counts()
        
        fig.add_trace(
            go.Bar(x=status_counts.index, y=status_counts.values,
                  marker_color=['green', 'red']),
            row=2, col=1
        )
        
        status = grid_simulator.get_grid_status()
        fig.add_trace(
            go.Indicator(
                mode = "gauge+number+delta",
                value = status['supply_demand_ratio'] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Grid Efficiency (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 80], 'color': "lightgray"},
                        {'range': [80, 95], 'color': "yellow"},
                        {'range': [95, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True, title_text="Smart Grid Real-time Dashboard")
        fig.show()

class DemandVisualizer:
    def __init__(self):
        self.fig_size = (10, 6)
    
    def plot_demand_prediction(self, actual_demand, predicted_demand, horizons=None, save_path=None):
        plt.figure(figsize=self.fig_size)
        
        time_steps = range(len(actual_demand))
        
        plt.plot(time_steps, actual_demand, 'b-', label='Actual Demand', linewidth=2)
        plt.plot(time_steps, predicted_demand, 'r--', label='Predicted Demand', linewidth=2)
        
        if horizons:
            for horizon in horizons:
                if horizon < len(actual_demand):
                    plt.axvline(x=horizon, color='gray', linestyle=':', alpha=0.7, 
                               label=f'Horizon {horizon}')
        
        plt.fill_between(time_steps, actual_demand, predicted_demand, 
                        alpha=0.2, color='red', label='Prediction Error')
        
        plt.title('Energy Demand Prediction vs Actual')
        plt.xlabel('Time Steps')
        plt.ylabel('Demand (MW)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_prediction_errors(self, errors_by_horizon, save_path=None):
        plt.figure(figsize=self.fig_size)
        
        horizons = list(errors_by_horizon.keys())
        mae_values = [errors_by_horizon[horizon]['mae'] for horizon in horizons]
        mape_values = [errors_by_horizon[horizon]['mape'] for horizon in horizons]
        
        x = np.arange(len(horizons))
        width = 0.35
        
        plt.bar(x - width/2, mae_values, width, label='MAE', alpha=0.7)
        plt.bar(x + width/2, mape_values, width, label='MAPE (%)', alpha=0.7)
        
        plt.xlabel('Prediction Horizon')
        plt.ylabel('Error')
        plt.title('Prediction Error by Horizon')
        plt.xticks(x, [f'H+{h}' for h in horizons])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()