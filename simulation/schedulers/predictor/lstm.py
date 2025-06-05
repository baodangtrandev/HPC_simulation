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

class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Define LSTM layer with batch_first=True for (batch, seq, feature) input format
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # Linear layer to map LSTM output to a single predicted value
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # Forward pass through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Select the output of the last time step
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through linear layer
        out = self.fc(out)
        return out

class LSTMWalltimePredictor:
    def __init__(self):
        # LSTM specific configurations
        self.configs = {
            "lstm": {
                "input_size": 4,
                "hidden_size": 64, 
                "num_layers": 3,
                "dropout": 0.2,
                "sequence_length": 10,
                "epochs": 50,
                "learning_rate": 0.001,
                "batch_size": 32,
                "patience": 5
            }
        }
        
        # Cache cho model đã được huấn luyện
        self.trained_models = {}
        
        # Cache cho dữ liệu
        self.finished_jobs_dataset = None
        self.data_features = None
        self.scaler = None
        
        # LSTM specific cache
        self.sequence_data = None
        
    def create_sequences(self, data, sequence_length):
        """Tạo sequences cho LSTM từ dữ liệu thời gian"""
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
        """Chuẩn bị dataset cho LSTM"""
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
        """Tạo LSTM model"""
        if model_name == "lstm":
            config = self.configs["lstm"]
            return LSTMModel(
                input_size=config["input_size"],
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                dropout=config["dropout"]
            )
        else:
            raise ValueError(f"Model '{model_name}' không được hỗ trợ.")
    
    def train_lstm_model(self, model, X_seq, y_seq, config):
        """Train LSTM model với early stopping"""
        model.to(device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(device)
        
        # Split data for validation (80-20 split)
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        criterion = nn.HuberLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            optimizer.zero_grad()
            train_outputs = model(X_train)
            train_loss = criterion(train_outputs, y_train)
            train_loss.backward()
            optimizer.step()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
            
            # Early stopping check
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{config["epochs"]}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    def predict(self, finished_jobs, job, data_size=1000, use_user_estimate=False, model_name="lstm"):
        """
        Dự đoán thời gian thực thi dựa trên LSTM
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
                self.train_lstm_model(model, X_seq, y_seq, self.configs[model_name])
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
            print(f"LSTM prediction error: {e}")
            return job.requested_time
    
    def evaluate_model(self, model, X_seq, y_seq, scaler):
        """Đánh giá model với RMSE, MAE, MSE, R2"""
        model.eval()
        predictions = []
        true_values = []
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(device)
            outputs = model(X_tensor).cpu().numpy()
            
            # Inverse transform predictions và targets
            for i in range(len(outputs)):
                pred_array = np.array([[0, 0, 0, 0, outputs[i, 0]]])
                true_array = np.array([[0, 0, 0, 0, y_seq[i]]])
                
                pred_inv = scaler.inverse_transform(pred_array)[0, -1]
                true_inv = scaler.inverse_transform(true_array)[0, -1]
                
                predictions.append(pred_inv)
                true_values.append(true_inv)
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        
        return rmse, mae, mse, r2
    
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
def LSTMWalltimePrediction(finished_jobs, job, data_size=1000, use_user_estimate=False, model_name="lstm"):
    """
    Wrapper function sử dụng LSTM cho dự đoán walltime
    
    Parameters:
    - finished_jobs: Danh sách các công việc đã hoàn thành
    - job: Công việc cần dự đoán
    - data_size: Số lượng công việc gần đây nhất để xét
    - use_user_estimate: Có sử dụng ước tính của người dùng không
    - model_name: Tên model để sử dụng (mặc định "lstm")
    
    Returns:
    - Thời gian dự đoán
    """
    # Singleton pattern để tái sử dụng model đã train
    # if not hasattr(LSTMWalltimePrediction, "_instance"):
    LSTMWalltimePrediction._instance = LSTMWalltimePredictor()
    
    predictor = LSTMWalltimePrediction._instance
    return predictor.predict(finished_jobs, job, data_size, use_user_estimate, model_name)

# Utility functions
def setup_lstm_config(input_size=4, hidden_size=64, num_layers=3, dropout=0.2, 
                      sequence_length=10, epochs=50, learning_rate=0.001, patience=5):
    """
    Thiết lập cấu hình cho LSTM model
    """
    if not hasattr(LSTMWalltimePrediction, "_instance"):
        LSTMWalltimePrediction._instance = LSTMWalltimePredictor()
    
    predictor = LSTMWalltimePrediction._instance
    new_config = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers, 
        "dropout": dropout,
        "sequence_length": sequence_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "patience": patience
    }
    
    predictor.update_config("lstm", new_config)
    return predictor

def clear_lstm_cache():
    """
    Xóa cache để buộc train lại model
    """
    if hasattr(LSTMWalltimePrediction, "_instance"):
        LSTMWalltimePrediction._instance.clear_cache()

def evaluate_lstm_performance(finished_jobs, data_size=1000):
    """
    Đánh giá performance của LSTM model
    """
    if not hasattr(LSTMWalltimePrediction, "_instance"):
        return None
    
    predictor = LSTMWalltimePrediction._instance
    
    if predictor.sequence_data is None:
        # Chuẩn bị dữ liệu trước
        predictor.prepare_dataset(finished_jobs, data_size)
        config = predictor.configs["lstm"]
        X_seq, y_seq = predictor.create_sequences(
            predictor.finished_jobs_dataset, 
            config["sequence_length"]
        )
        predictor.sequence_data = (X_seq, y_seq)
    
    if "lstm" not in predictor.trained_models:
        print("Model chưa được train. Vui lòng chạy prediction trước.")
        return None
    
    X_seq, y_seq = predictor.sequence_data
    model = predictor.trained_models["lstm"]
    
    return predictor.evaluate_model(model, X_seq, y_seq, predictor.scaler)

# Example usage và test
def test_lstm_model():
    """
    Test function để kiểm tra LSTM model
    """
    # Tạo mock data
    class MockJob:
        def __init__(self, resources, time, exe_num, uid, profile):
            self.requested_resources = resources
            self.requested_time = time
            self.json_dict = {"exe_num": exe_num, "uid": uid}
            self.profile = profile
    
    # Tạo finished jobs
    finished_jobs = []
    np.random.seed(42)
    for i in range(100):
        resources = np.random.randint(1, 16)
        time = np.random.randint(60, 3600)
        exe_num = np.random.randint(1, 100)
        uid = np.random.randint(1000, 9999)
        profile = time * (0.8 + 0.4 * np.random.random())  # Actual time ~ requested time
        
        finished_jobs.append(MockJob(resources, time, exe_num, uid, profile))
    
    # Tạo job cần predict
    test_job = MockJob(8, 1800, 50, 5555, 0)
    
    # Test prediction
    print("Testing LSTM model...")
    prediction = LSTMWalltimePrediction(finished_jobs, test_job, model_name="lstm")
    print(f"Prediction: {prediction}")
    
    # Test với user estimate
    prediction_with_estimate = LSTMWalltimePrediction(
        finished_jobs, test_job, use_user_estimate=True, model_name="lstm"
    )
    print(f"Prediction with user estimate: {prediction_with_estimate}")
    
    # Evaluate performance
    metrics = evaluate_lstm_performance(finished_jobs)
    if metrics:
        rmse, mae, mse, r2 = metrics
        print(f"Model Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R2: {r2:.4f}")

# Uncomment để test
# test_lstm_model()
