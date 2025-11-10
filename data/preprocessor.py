import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

class EnergyDataPreprocessor:
    def __init__(self):
        self.demand_scaler = StandardScaler()
        self.temperature_scaler = StandardScaler()
        self.feature_scalers = {}
    
    def preprocess_training_data(self, data):
        processed_data = data.copy()
        
        processed_data['energy_demand'] = self.demand_scaler.fit_transform(
            processed_data[['energy_demand']]
        )
        
        processed_data['temperature'] = self.temperature_scaler.fit_transform(
            processed_data[['temperature']]
        )
        
        cyclical_features = ['hour', 'day_of_week', 'month']
        for feature in cyclical_features:
            if feature in processed_data.columns:
                processed_data[f'{feature}_sin'] = np.sin(2 * np.pi * processed_data[feature] / processed_data[feature].max())
                processed_data[f'{feature}_cos'] = np.cos(2 * np.pi * processed_data[feature] / processed_data[feature].max())
        
        feature_columns = [col for col in processed_data.columns if col not in ['timestamp', 'hour', 'day_of_week', 'month']]
        
        for col in feature_columns:
            if col not in ['energy_demand', 'temperature']:
                self.feature_scalers[col] = StandardScaler()
                processed_data[col] = self.feature_scalers[col].fit_transform(processed_data[[col]])
        
        return processed_data[feature_columns]
    
    def preprocess_inference_data(self, data):
        processed_data = data.copy()
        
        if 'energy_demand' in processed_data.columns:
            processed_data['energy_demand'] = self.demand_scaler.transform(
                processed_data[['energy_demand']]
            )
        
        if 'temperature' in processed_data.columns:
            processed_data['temperature'] = self.temperature_scaler.transform(
                processed_data[['temperature']]
            )
        
        cyclical_features = ['hour', 'day_of_week', 'month']
        for feature in cyclical_features:
            if feature in processed_data.columns:
                processed_data[f'{feature}_sin'] = np.sin(2 * np.pi * processed_data[feature] / 24 if feature == 'hour' else 7 if feature == 'day_of_week' else 12)
                processed_data[f'{feature}_cos'] = np.cos(2 * np.pi * processed_data[feature] / 24 if feature == 'hour' else 7 if feature == 'day_of_week' else 12)
        
        feature_columns = [col for col in processed_data.columns if col not in ['timestamp', 'hour', 'day_of_week', 'month']]
        
        for col in feature_columns:
            if col in self.feature_scalers and col not in ['energy_demand', 'temperature']:
                processed_data[col] = self.feature_scalers[col].transform(processed_data[[col]])
        
        return processed_data[feature_columns]
    
    def inverse_transform_demand(self, scaled_demand):
        return self.demand_scaler.inverse_transform(scaled_demand.reshape(-1, 1)).flatten()

class FeatureEngineer:
    def __init__(self):
        self.rolling_windows = [3, 6, 12, 24]
    
    def create_temporal_features(self, data):
        engineered = data.copy()
        
        if 'energy_demand' in engineered.columns:
            for window in self.rolling_windows:
                engineered[f'demand_rolling_mean_{window}'] = engineered['energy_demand'].rolling(window=window, min_periods=1).mean()
                engineered[f'demand_rolling_std_{window}'] = engineered['energy_demand'].rolling(window=window, min_periods=1).std()
            
            engineered['demand_lag_1'] = engineered['energy_demand'].shift(1)
            engineered['demand_lag_24'] = engineered['energy_demand'].shift(24)
            engineered['demand_trend'] = engineered['energy_demand'].diff()
        
        engineered['is_weekend'] = engineered.get('day_of_week', 0) >= 5
        engineered['is_peak_hour'] = engineered.get('hour', 0).between(17, 21)
        
        return engineered.fillna(method='bfill').fillna(method='ffill')