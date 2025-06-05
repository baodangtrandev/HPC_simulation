import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
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

# Global variables để cache model và scaler
_lstm_model = None
_lstm_scaler = None
_lstm_last_data_size = 0

def create_sequences(data, sequence_length=10):
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

def train_lstm_model(model, X_seq, y_seq, epochs=35, learning_rate=0.001):
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    
    for epoch in range(epochs):
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
            if patience_counter >= patience:
                print(f"LSTM Early stopping at epoch {epoch+1}")
                break
        
        if epoch % 10 == 0:
            print(f'LSTM Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

def LSTMWalltimePredictor(finished_jobs, job, data_size=1000, use_user_estimate=False):
    """
    LSTM Walltime Predictor - Interface tương tự KnnWalltimePredictor
    
    Parameters:
    - finished_jobs: Danh sách các công việc đã hoàn thành
    - job: Công việc cần dự đoán
    - data_size: Số lượng công việc gần đây nhất để xét
    - use_user_estimate: Có sử dụng ước tính của người dùng không
    
    Returns:
    - Thời gian dự đoán
    """
    global _lstm_model, _lstm_scaler, _lstm_last_data_size
    
    # Kiểm tra dữ liệu
    neighbor_space = finished_jobs[-data_size:]
    if len(neighbor_space) < 18:  # Cần ít nhất 18 jobs cho LSTM
        return max(int(job.requested_time * 1.05), 1)  # LSTM fallback với slight over-estimate
    
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
        if _lstm_scaler is None or len(neighbor_space) != _lstm_last_data_size:
            _lstm_scaler = StandardScaler()
            data_scaled = _lstm_scaler.fit_transform(data)
            _lstm_last_data_size = len(neighbor_space)
        else:
            data_scaled = _lstm_scaler.transform(data)
        
        # Tạo sequences
        X_seq, y_seq = create_sequences(data_scaled, sequence_length=10)
        
        # Train model nếu chưa có hoặc dữ liệu thay đổi
        if _lstm_model is None or len(neighbor_space) != _lstm_last_data_size:
            _lstm_model = LSTMModel(
                input_size=4,
                hidden_size=64,
                num_layers=3,
                dropout=0.2
            )
            train_lstm_model(_lstm_model, X_seq, y_seq, epochs=35)
        
        # Chuẩn bị dữ liệu cho job cần dự đoán
        job_features = np.array([[
            job.requested_resources,
            job.requested_time,
            job.json_dict["exe_num"],
            job.json_dict["uid"],
            0  # Placeholder cho target
        ]])
        
        # Scale job features
        job_scaled = _lstm_scaler.transform(job_features)
        
        # Tạo sequence cho prediction (sử dụng dữ liệu gần nhất)
        recent_data = data_scaled[-10:]  # Lấy 10 jobs gần nhất
        pred_sequence = np.vstack([recent_data[1:, :-1], job_scaled[0, :-1]])
        
        # Predict
        _lstm_model.eval()
        with torch.no_grad():
            pred_tensor = torch.FloatTensor(pred_sequence).unsqueeze(0).to(device)
            prediction = _lstm_model(pred_tensor).cpu().numpy()[0, 0]
        
        # Inverse transform prediction
        pred_array = np.array([[0, 0, 0, 0, prediction]])
        pred_scaled = _lstm_scaler.inverse_transform(pred_array)
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
            
            original_data = _lstm_scaler.inverse_transform(dummy_features)
            actual_times = original_data[:, -1]
            request_times = original_data[:, 1]
            
            calibration_factor = np.mean(request_times / actual_times)
            predict = max(job.requested_time / calibration_factor, 1)
        
        # LSTM-specific adjustment (over-prediction tendency)
        predict = predict * 1.08
        
        return max(int(predict), 1)
        
    except Exception as e:
        print(f"LSTM prediction error: {e}")
        return max(int(job.requested_time * 1.05), 1)

# Optional: Function để clear cache khi cần
def clear_lstm_cache():
    """Clear LSTM model cache"""
    global _lstm_model, _lstm_scaler, _lstm_last_data_size
    _lstm_model = None
    _lstm_scaler = None
    _lstm_last_data_size = 0