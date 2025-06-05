import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
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

# Global variables để cache model và scaler
_rnn_model = None
_rnn_scaler = None
_rnn_last_data_size = 0

def create_sequences(data, sequence_length=10):
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

def train_rnn_model(model, X_seq, y_seq, epochs=30, learning_rate=0.001):
    """Train RNN model"""
    model.to(device)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_seq).to(device)
    y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(device)
    
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'RNN Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

def RNNWalltimePredictor(finished_jobs, job, data_size=1000, use_user_estimate=False):
    """
    RNN Walltime Predictor - Interface tương tự KnnWalltimePredictor
    
    Parameters:
    - finished_jobs: Danh sách các công việc đã hoàn thành
    - job: Công việc cần dự đoán
    - data_size: Số lượng công việc gần đây nhất để xét
    - use_user_estimate: Có sử dụng ước tính của người dùng không
    
    Returns:
    - Thời gian dự đoán
    """
    global _rnn_model, _rnn_scaler, _rnn_last_data_size
    
    # Kiểm tra dữ liệu
    neighbor_space = finished_jobs[-data_size:]
    if len(neighbor_space) < 15:  # Cần ít nhất 15 jobs cho RNN
        return max(int(job.requested_time * 0.95), 1)  # RNN fallback với slight under-estimate
    
    try:
        # Chuẩn bị dữ liệu
        dataset = []
        for finished_job in neighbor_space:
            features = [
                finished_job.requested_resources,
                finished_job.requested_time,
                finished_job.json_dict["exe_num"],
                finished_job.json_dict["uid"],
                int(finished_job.profile)  # Target (actual runtime)
            ]
            dataset.append(features)
        
        data = np.array(dataset)
        
        # Tạo hoặc sử dụng scaler
        if _rnn_scaler is None or len(neighbor_space) != _rnn_last_data_size:
            _rnn_scaler = StandardScaler()
            data_scaled = _rnn_scaler.fit_transform(data)
            _rnn_last_data_size = len(neighbor_space)
        else:
            data_scaled = _rnn_scaler.transform(data)
        
        # Tạo sequences
        X_seq, y_seq = create_sequences(data_scaled, sequence_length=10)
        
        # Train model nếu chưa có hoặc dữ liệu thay đổi
        if _rnn_model is None or len(neighbor_space) != _rnn_last_data_size:
            _rnn_model = RNNModel(input_size=4, hidden_size=64, num_layers=3, dropout=0.2)
            train_rnn_model(_rnn_model, X_seq, y_seq, epochs=30)
        
        # Chuẩn bị dữ liệu cho job cần dự đoán
        job_features = np.array([[
            job.requested_resources,
            job.requested_time,
            job.json_dict["exe_num"],
            job.json_dict["uid"],
            0  # Placeholder cho target
        ]])
        
        # Scale job features
        job_scaled = _rnn_scaler.transform(job_features)
        
        # Tạo sequence cho prediction (sử dụng dữ liệu gần nhất)
        recent_data = data_scaled[-10:]  # Lấy 10 jobs gần nhất
        pred_sequence = np.vstack([recent_data[1:, :-1], job_scaled[0, :-1]])
        
        # Predict
        _rnn_model.eval()
        with torch.no_grad():
            pred_tensor = torch.FloatTensor(pred_sequence).unsqueeze(0).to(device)
            prediction = _rnn_model(pred_tensor).cpu().numpy()[0, 0]
        
        # Inverse transform prediction
        pred_array = np.array([[0, 0, 0, 0, prediction]])
        pred_scaled = _rnn_scaler.inverse_transform(pred_array)
        predict = pred_scaled[0, -1]
        
        # User estimate calibration (tương tự KNN)
        if use_user_estimate:
            # Tính calibration factor từ dữ liệu gần đây
            actual_runtimes = data_scaled[:, -1]
            requested_times = data_scaled[:, 1]
            
            # Inverse transform để lấy giá trị thực
            dummy_features = np.zeros((len(actual_runtimes), 5))
            dummy_features[:, 1] = requested_times
            dummy_features[:, -1] = actual_runtimes
            
            original_data = _rnn_scaler.inverse_transform(dummy_features)
            actual_times = original_data[:, -1]
            request_times = original_data[:, 1]
            
            calibration_factor = np.mean(request_times / actual_times)
            predict = max(job.requested_time / calibration_factor, 1)
        
        # RNN-specific adjustment (slight under-prediction tendency)
        predict = predict * 0.95
        
        return max(int(predict), 1)
        
    except Exception as e:
        print(f"RNN prediction error: {e}")
        return max(int(job.requested_time * 0.95), 1)

# Optional: Function để clear cache khi cần
def clear_rnn_cache():
    """Clear RNN model cache"""
    global _rnn_model, _rnn_scaler, _rnn_last_data_size
    _rnn_model = None
    _rnn_scaler = None
    _rnn_last_data_size = 0