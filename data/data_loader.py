import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class EnergyDataset(Dataset):
    def __init__(self, data_path, sequence_length=24, prediction_horizon=12):
        self.data_path = Path(data_path)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.data = self.load_data()
        self.processed_data = self.preprocess_data()
    
    def load_data(self):
        try:
            data = pd.read_csv(self.data_path)
        except FileNotFoundError:
            data = self.generate_synthetic_data()
        return data
    
    def generate_synthetic_data(self):
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='H')
        base_demand = 50.0
        seasonal_variation = 20 * np.sin(2 * np.pi * dates.dayofyear / 365)
        daily_pattern = 15 * np.sin(2 * np.pi * dates.hour / 24)
        noise = np.random.normal(0, 5, len(dates))
        
        demand = base_demand + seasonal_variation + daily_pattern + noise
        temperature = 15 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 3, len(dates))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'energy_demand': np.maximum(10, demand),
            'temperature': temperature,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month
        })
        
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(self.data_path, index=False)
        return data
    
    def preprocess_data(self):
        data = self.data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        
        numeric_cols = ['energy_demand', 'temperature']
        for col in numeric_cols:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
        
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon):
            seq = data.iloc[i:i + self.sequence_length][numeric_cols].values
            target = data.iloc[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]['energy_demand'].values
            sequences.append(seq)
            targets.append(target)
        
        return list(zip(sequences, targets))
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        sequence, target = self.processed_data[idx]
        return torch.FloatTensor(sequence), torch.FloatTensor(target)

class RealTimeDataStream:
    def __init__(self, update_interval=300):
        self.update_interval = update_interval
        self.current_data = None
        self.historical_data = []
    
    def update_stream(self, new_data):
        self.historical_data.append(new_data)
        if len(self.historical_data) > 1000:
            self.historical_data.pop(0)
        self.current_data = new_data
    
    def get_latest_sequence(self, sequence_length):
        if len(self.historical_data) < sequence_length:
            return None
        return self.historical_data[-sequence_length:]
    
    def get_statistics(self):
        if not self.historical_data:
            return {}
        data_array = np.array(self.historical_data)
        return {
            'mean': float(np.mean(data_array)),
            'std': float(np.std(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'count': len(self.historical_data)
        }