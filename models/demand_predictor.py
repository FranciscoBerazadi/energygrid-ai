import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LSTMDemandPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=12, dropout=0.2):
        super(LSTMDemandPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)
        
        lstm_out = lstm_out.transpose(0, 1)
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        attended_out = attended_out.transpose(0, 1)
        
        last_timestep = attended_out[:, -1, :]
        x = F.relu(self.fc1(last_timestep))
        x = self.dropout(x)
        predictions = self.fc2(x)
        
        return predictions

class DemandPredictor:
    def __init__(self, input_dim, sequence_length=24, prediction_horizon=12):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = LSTMDemandPredictor(input_dim=input_dim, output_dim=prediction_horizon)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.HuberLoss()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train(self, train_loader, val_loader, epochs=100):
        self.model.train()
        
        for epoch in range(epochs):
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = self.validate(val_loader)
            
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()
        
        self.model.train()
        return val_loss / len(val_loader)
    
    def predict(self, input_sequence):
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
            prediction = self.model(input_tensor).cpu().numpy().flatten()
        return prediction
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        self.sequence_length = checkpoint['sequence_length']
        self.prediction_horizon = checkpoint['prediction_horizon']