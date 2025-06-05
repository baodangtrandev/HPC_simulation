import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNNModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=3, dropout=0.2):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]  # Take last timestep
        out = self.fc(out)
        return out

class RNNWalltimePredictor:
    def __init__(self):
        # RNN specific configurations
        self.configs = {
            "rnn": {
                "input_size": 4,
                "hidden_size": 64, 
                "num_layers": 3,
                "dropout": 0.2,
                "sequence_length": 10,
                "epochs": 10,
                "learning_rate": 0.001,
                "batch_size": 32
            }
        }
        
        # Cache cho model đã được huấn luyện
        self.trained_models = {}
        
        # Cache cho dữ liệu
        self.finished_jobs_dataset = None
        self.data_features = None
        self.scaler = None
        
        # RNN specific cache
        self.sequence_data = None
        
    def create_sequences(self, data, sequence_length):
        """Tạo sequences cho RNN từ dữ liệu thời gian"""
        if len(data) < sequence_length:
            # Nếu không đủ dữ liệu, padding với dữ liệu đầu tiên
            padded_data = np.tile(data[0], (sequence_length - len(data), 1))
            data = np.vstack([padded_data, data])
        
        sequences = []
        targets = []
        
        for i in range(len(data) - sequence_length + 1):
            seq = data[i:(i + sequence_length)]
            target = data[i + sequence_length - 1, -1]  # Target là runtime của timestep cuối
            sequences.append(seq[:, :-1])  # Loại bỏ cột target khỏi features
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def prepare_dataset(self, finished_jobs, data_size=1000):
        """Chuẩn bị dataset cho RNN"""
        neighbor_space = finished_jobs[-data_size:]
        
        # Trích xuất features
        raw_data = []
        for finished_job in neighbor_space:
            features = [
                finished_job.requested_resources,
                finished_job.requested_time, 
                finished_job.json_dict["exe_num"],
                finished_job.json_dict["uid"],
                int(finished_job.profile)  # Target (actual runtime)
            ]
            raw_data.append(features)
        
        data = np.array(raw_data)
        
        # Chuẩn hóa dữ liệu
        if self.scaler is None:
            self.scaler = StandardScaler()
            data_scaled = self.scaler.fit_transform(data)
        else:
            data_scaled = self.scaler.transform(data)
            
        return data_scaled
    
    def create_model(self, model_name):
        """Tạo RNN model"""
        if model_name == "rnn":
            config = self.configs["rnn"]
            return RNNModel(
                input_size=config["input_size"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                dropout=config["dropout"]
            )
        else:
            raise ValueError(f"Model '{model_name}' không được hỗ trợ.")
    
    def train_rnn_model(self, model, X_seq, y_seq, config):
        """Train RNN model"""
        model.to(device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(device)
        
        criterion = nn.HuberLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        model.train()
        for epoch in range(config["epochs"]):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{config["epochs"]}], Loss: {loss.item():.4f}')
    
    def predict(self, finished_jobs, job, data_size=1000, use_user_estimate=False, model_name="rnn"):
        """
        Dự đoán thời gian thực thi dựa trên RNN
        """
        # Kiểm tra dữ liệu
        neighbor_space = finished_jobs[-data_size:]
        if len(neighbor_space) < 10:  # Cần ít nhất 10 jobs cho sequence
            return job.requested_time
        
        try:
            # Chuẩn bị dữ liệu
            if (self.finished_jobs_dataset is None or 
                len(self.finished_jobs_dataset) != len(neighbor_space)):
                
                self.finished_jobs_dataset = self.prepare_dataset(finished_jobs, data_size)
                self.sequence_data = None  # Reset sequence cache
            
            # Tạo sequences
            if self.sequence_data is None:
                config = self.configs[model_name]
                X_seq, y_seq = self.create_sequences(
                    self.finished_jobs_dataset, 
                    config["sequence_length"]
                )
                self.sequence_data = (X_seq, y_seq)
            else:
                X_seq, y_seq = self.sequence_data
            
            # Train model nếu chưa được train
            if model_name not in self.trained_models:
                model = self.create_model(model_name)
                self.train_rnn_model(model, X_seq, y_seq, self.configs[model_name])
                self.trained_models[model_name] = model
            else:
                model = self.trained_models[model_name]
            
            # Chuẩn bị dữ liệu cho job cần dự đoán
            job_features = np.array([[
                job.requested_resources,
                job.requested_time,
                job.json_dict["exe_num"], 
                job.json_dict["uid"],
                0  # Placeholder cho target
            ]])
            
            # Scale job features
            job_scaled = self.scaler.transform(job_features)
            
            # Tạo sequence cho prediction (sử dụng dữ liệu gần nhất)
            recent_data = self.finished_jobs_dataset[-self.configs[model_name]["sequence_length"]:]
            pred_sequence = np.vstack([recent_data[1:, :-1], job_scaled[0, :-1]])
            
            # Predict
            model.eval()
            with torch.no_grad():
                pred_tensor = torch.FloatTensor(pred_sequence).unsqueeze(0).to(device)
                prediction = model(pred_tensor).cpu().numpy()[0, 0]
            
            # Inverse transform prediction
            pred_array = np.array([[0, 0, 0, 0, prediction]])
            pred_scaled = self.scaler.inverse_transform(pred_array)
            predict = pred_scaled[0, -1]
            
            # User estimate calibration
            if use_user_estimate:
                actual_runtimes = self.finished_jobs_dataset[:, -1]
                requested_times = self.finished_jobs_dataset[:, 1]
                
                # Inverse transform để lấy giá trị thực
                dummy_features = np.zeros((len(actual_runtimes), 5))
                dummy_features[:, 1] = requested_times
                dummy_features[:, -1] = actual_runtimes
                
                original_data = self.scaler.inverse_transform(dummy_features)
                actual_times = original_data[:, -1]
                request_times = original_data[:, 1]
                
                calibration_factor = np.mean(request_times / actual_times)
                predict = max(job.requested_time / calibration_factor, 1)
            
            return max(int(predict), 1)
            
        except Exception as e:
            print(f"RNN prediction error: {e}")
            return job.requested_time
    
    def clear_cache(self):
        """Xóa cache"""
        self.trained_models = {}
        self.finished_jobs_dataset = None
        self.sequence_data = None
        self.scaler = None
        return self
    
    def update_config(self, model_name, new_config):
        """Cập nhật cấu hình cho model"""
        if model_name in self.configs:
            self.configs[model_name] = {**self.configs[model_name], **new_config}
            if model_name in self.trained_models:
                del self.trained_models[model_name]
        return self

# Wrapper function với cùng interface như HDEM
def RNNWalltimePrediction(finished_jobs, job, data_size=1000, use_user_estimate=False, model_name="rnn"):
    """
    Wrapper function sử dụng RNN cho dự đoán walltime
    
    Parameters:
    - finished_jobs: Danh sách các công việc đã hoàn thành
    - job: Công việc cần dự đoán
    - data_size: Số lượng công việc gần đây nhất để xét
    - use_user_estimate: Có sử dụng ước tính của người dùng không
    - model_name: Tên model để sử dụng (mặc định "rnn")
    
    Returns:
    - Thời gian dự đoán
    """
    # Singleton pattern để tái sử dụng model đã train
    if not hasattr(RNNWalltimePrediction, "_instance"):
        RNNWalltimePrediction._instance = RNNWalltimePredictor()
    
    predictor = RNNWalltimePrediction._instance
    return predictor.predict(finished_jobs, job, data_size, use_user_estimate, model_name)

# Utility functions
def setup_rnn_config(input_size=4, hidden_size=64, num_layers=3, dropout=0.2, 
                     sequence_length=10, epochs=10, learning_rate=0.001):
    """
    Thiết lập cấu hình cho RNN model
    """
    if not hasattr(RNNWalltimePrediction, "_instance"):
        RNNWalltimePrediction._instance = RNNWalltimePredictor()
    
    predictor = RNNWalltimePrediction._instance
    new_config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers, 
        "dropout": dropout,
        "sequence_length": sequence_length,
        "epochs": epochs,
        "learning_rate": learning_rate
    }
    
    predictor.update_config("rnn", new_config)
    return predictor

def clear_rnn_cache():
    """
    Xóa cache để buộc train lại model
    """
    if hasattr(RNNWalltimePrediction, "_instance"):
        RNNWalltimePrediction._instance.clear_cache()
