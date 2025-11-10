import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class GridMetrics:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_efficiency_metrics(self, actual_supply, actual_demand, predicted_demand=None):
        metrics = {}
        
        metrics['supply_demand_ratio'] = actual_supply / actual_demand if actual_demand > 0 else 0
        metrics['power_deficit'] = max(0, actual_demand - actual_supply)
        metrics['power_surplus'] = max(0, actual_supply - actual_demand)
        metrics['utilization_rate'] = actual_supply / actual_demand if actual_demand > 0 else 0
        
        if predicted_demand is not None:
            metrics['demand_prediction_error'] = abs(actual_demand - predicted_demand)
            metrics['prediction_accuracy'] = 1 - (metrics['demand_prediction_error'] / actual_demand) if actual_demand > 0 else 0
        
        return metrics
    
    def calculate_reliability_metrics(self, blackout_nodes, total_nodes, blackout_duration=None):
        metrics = {}
        
        metrics['system_availability'] = 1 - (blackout_nodes / total_nodes) if total_nodes > 0 else 1
        metrics['blackout_frequency'] = blackout_nodes / total_nodes if total_nodes > 0 else 0
        
        if blackout_duration is not None:
            metrics['average_blackout_duration'] = np.mean(blackout_duration) if blackout_duration else 0
            metrics['max_blackout_duration'] = np.max(blackout_duration) if blackout_duration else 0
        
        return metrics
    
    def calculate_economic_metrics(self, generation_costs, transmission_costs, blackout_costs):
        metrics = {}
        
        metrics['total_operational_cost'] = generation_costs + transmission_costs
        metrics['blackout_cost'] = blackout_costs
        metrics['total_cost'] = metrics['total_operational_cost'] + metrics['blackout_cost']
        metrics['cost_efficiency'] = 1 / metrics['total_cost'] if metrics['total_cost'] > 0 else 0
        
        return metrics
    
    def calculate_sustainability_metrics(self, renewable_generation, total_generation, carbon_emissions):
        metrics = {}
        
        metrics['renewable_penetration'] = renewable_generation / total_generation if total_generation > 0 else 0
        metrics['carbon_intensity'] = carbon_emissions / total_generation if total_generation > 0 else 0
        metrics['sustainability_index'] = metrics['renewable_penetration'] * (1 - metrics['carbon_intensity'])
        
        return metrics
    
    def update_metrics_history(self, metrics_dict):
        self.metrics_history.append(metrics_dict)
    
    def get_performance_summary(self, window_size=100):
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        if len(df) > window_size:
            recent_df = df.tail(window_size)
        else:
            recent_df = df
        
        summary = {}
        
        efficiency_metrics = ['supply_demand_ratio', 'utilization_rate']
        for metric in efficiency_metrics:
            if metric in recent_df.columns:
                summary[f'avg_{metric}'] = recent_df[metric].mean()
                summary[f'std_{metric}'] = recent_df[metric].std()
        
        reliability_metrics = ['system_availability', 'blackout_frequency']
        for metric in reliability_metrics:
            if metric in recent_df.columns:
                summary[f'avg_{metric}'] = recent_df[metric].mean()
        
        if 'total_cost' in recent_df.columns:
            summary['avg_total_cost'] = recent_df['total_cost'].mean()
            summary['cost_reduction'] = (df['total_cost'].iloc[0] - df['total_cost'].iloc[-1]) / df['total_cost'].iloc[0] if df['total_cost'].iloc[0] > 0 else 0
        
        if 'sustainability_index' in recent_df.columns:
            summary['avg_sustainability'] = recent_df['sustainability_index'].mean()
            summary['sustainability_improvement'] = (df['sustainability_index'].iloc[-1] - df['sustainability_index'].iloc[0]) / abs(df['sustainability_index'].iloc[0]) if df['sustainability_index'].iloc[0] != 0 else 0
        
        return summary

class DemandPredictionMetrics:
    def __init__(self):
        self.actuals = []
        self.predictions = []
    
    def update(self, actual, predicted):
        self.actuals.extend(actual)
        self.predictions.extend(predicted)
    
    def calculate_metrics(self):
        if not self.actuals or not self.predictions:
            return {}
        
        actuals = np.array(self.actuals)
        predictions = np.array(self.predictions)
        
        metrics = {}
        metrics['mse'] = mean_squared_error(actuals, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(actuals, predictions)
        metrics['mape'] = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        metrics['r2'] = r2_score(actuals, predictions)
        
        metrics['bias'] = np.mean(predictions - actuals)
        metrics['std_error'] = np.std(predictions - actuals)
        
        return metrics
    
    def get_horizon_metrics(self, horizon):
        if len(self.actuals) < horizon:
            return {}
        
        horizon_actuals = []
        horizon_predictions = []
        
        for i in range(0, len(self.actuals) - horizon + 1, horizon):
            horizon_actuals.append(self.actuals[i + horizon - 1])
            horizon_predictions.append(self.predictions[i + horizon - 1])
        
        return self._calculate_slice_metrics(horizon_actuals, horizon_predictions)
    
    def _calculate_slice_metrics(self, actuals, predictions):
        actuals = np.array(actuals)
        predictions = np.array(predictions)
        
        return {
            'mse': mean_squared_error(actuals, predictions),
            'mae': mean_absolute_error(actuals, predictions),
            'mape': np.mean(np.abs((actuals - predictions) / actuals)) * 100,
            'bias': np.mean(predictions - actuals)
        }