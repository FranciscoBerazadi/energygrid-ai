from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime, timedelta

app = Flask(__name__)

class EnergyGridAPI:
    def __init__(self):
        self.grid_simulator = None
        self.rl_agent = None
        self.demand_predictor = None
        self.is_initialized = False
        
        self.initialize_system()
    
    def initialize_system(self):
        try:
            from models.grid_simulator import PowerGridSimulator
            from models.rl_agent import DDPGAgent
            from models.demand_predictor import DemandPredictor
            
            self.grid_simulator = PowerGridSimulator()
            self.rl_agent = DDPGAgent(state_dim=90, action_dim=15)
            
            model_path = Path("trained_models/best_model.pth")
            if model_path.exists():
                self.rl_agent.load_model(model_path)
            
            self.demand_predictor = DemandPredictor(input_dim=2)
            demand_model_path = Path("trained_models/best_demand_predictor.pth")
            if demand_model_path.exists():
                self.demand_predictor.load_model(demand_model_path)
            
            self.is_initialized = True
            print("EnergyGrid AI System initialized successfully")
            
        except Exception as e:
            print(f"Initialization failed: {e}")
            self.is_initialized = False
    
    def get_grid_status(self):
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        status = self.grid_simulator.get_grid_status()
        state = self.grid_simulator.get_state()
        
        current_demands = [self.grid_simulator.nodes[i]['demand'] for i in range(self.grid_simulator.num_nodes)]
        current_generations = [self.grid_simulator.nodes[i]['current_generation'] for i in range(self.grid_simulator.num_nodes)]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "grid_status": status,
            "node_demands": current_demands,
            "node_generations": current_generations,
            "total_nodes": self.grid_simulator.num_nodes,
            "blackout_nodes": [i for i in range(self.grid_simulator.num_nodes) if self.grid_simulator.nodes[i]['is_blackout']]
        }
    
    def optimize_grid(self, demands=None):
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        if demands is None:
            demands = [self.grid_simulator.nodes[i]['demand'] for i in range(self.grid_simulator.num_nodes)]
        
        state = self.grid_simulator.get_state()
        actions = self.rl_agent.select_action(state, noise_scale=0.1)
        
        next_state, reward, done, info = self.grid_simulator.step(actions, demands)
        
        optimization_result = {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": actions.tolist(),
            "reward": float(reward),
            "efficiency": float(info['efficiency']),
            "blackout_nodes": info['blackout_nodes'],
            "power_deficit": float(np.sum(info['power_deficit'])),
            "grid_status": info['grid_status']
        }
        
        return optimization_result
    
    def predict_demand(self, historical_data=None, hours_ahead=12):
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        if historical_data is None:
            historical_data = self._generate_sample_historical_data()
        
        try:
            prediction = self.demand_predictor.predict(historical_data)
            return {
                "timestamp": datetime.now().isoformat(),
                "prediction_horizon": hours_ahead,
                "predicted_demand": prediction.tolist(),
                "confidence_interval": self._calculate_confidence_interval(prediction)
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _generate_sample_historical_data(self):
        base_demand = 50
        time_variation = 20 * np.sin(2 * np.pi * np.arange(24) / 24)
        temperature = 15 + 10 * np.sin(2 * np.pi * np.arange(24) / 24)
        
        demands = base_demand + time_variation + np.random.normal(0, 5, 24)
        historical_data = np.column_stack([demands, temperature])
        
        return historical_data
    
    def _calculate_confidence_interval(self, prediction, confidence=0.95):
        std_dev = np.std(prediction) * 0.1
        z_score = 1.96
        
        margin_of_error = z_score * std_dev
        
        return {
            "lower_bound": (prediction - margin_of_error).tolist(),
            "upper_bound": (prediction + margin_of_error).tolist(),
            "confidence_level": confidence
        }

energy_grid_api = EnergyGridAPI()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/grid/status', methods=['GET'])
def get_grid_status():
    status = energy_grid_api.get_grid_status()
    return jsonify(status)

@app.route('/api/grid/optimize', methods=['POST'])
def optimize_grid():
    data = request.get_json()
    demands = data.get('demands') if data else None
    
    result = energy_grid_api.optimize_grid(demands)
    return jsonify(result)

@app.route('/api/demand/predict', methods=['POST'])
def predict_demand():
    data = request.get_json()
    historical_data = data.get('historical_data') if data else None
    hours_ahead = data.get('hours_ahead', 12) if data else 12
    
    if historical_data:
        historical_data = np.array(historical_data)
    
    result = energy_grid_api.predict_demand(historical_data, hours_ahead)
    return jsonify(result)

@app.route('/api/grid/history', methods=['GET'])
def get_grid_history():
    if not energy_grid_api.is_initialized:
        return jsonify({"error": "System not initialized"})
    
    history = energy_grid_api.grid_simulator.history[-100:]
    return jsonify({
        "history": history,
        "total_records": len(history)
    })

@app.route('/api/system/health', methods=['GET'])
def system_health():
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "system_initialized": energy_grid_api.is_initialized,
        "grid_simulator_ready": energy_grid_api.grid_simulator is not None,
        "rl_agent_ready": energy_grid_api.rl_agent is not None,
        "demand_predictor_ready": energy_grid_api.demand_predictor is not None,
        "status": "healthy" if energy_grid_api.is_initialized else "degraded"
    }
    
    return jsonify(health_status)

@app.route('/api/grid/reset', methods=['POST'])
def reset_grid():
    if not energy_grid_api.is_initialized:
        return jsonify({"error": "System not initialized"})
    
    state = energy_grid_api.grid_simulator.reset()
    
    return jsonify({
        "message": "Grid reset successfully",
        "initial_state": state.tolist(),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)