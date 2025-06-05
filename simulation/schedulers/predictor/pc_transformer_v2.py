import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Check if GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Time Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_trend = nn.Linear(1, dim)
        self.linear_period = nn.Linear(1, dim)

    def forward(self, t):
        trend = F.relu(self.linear_trend(t))
        period = torch.sin(self.linear_period(t))
        return trend + period

# Multi-Head Attention Block
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context)

# Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# PC-Transformer Block
class PCTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(PCTransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(self.dropout(ff_output))
        return x

# Full PC-Transformer Model
class PCTransformerModel(nn.Module):
    def __init__(self, input_dim=4, d_model=64, num_heads=8, d_ff=256, num_layers=4, output_dim=1, dropout=0.1):
        super(PCTransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model)
        self.time_embedding = TimeEmbedding(d_model)
        self.transformer_blocks = nn.ModuleList([
            PCTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, x, t):
        x = self.input_linear(x)
        time_emb = self.time_embedding(t)
        x = x + time_emb
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling over sequence dimension
        x = torch.mean(x, dim=1)
        x = self.output_linear(x)
        return x

# Global variables để cache model và scaler
_pct_model = None
_pct_scaler = None
_pct_last_data_size = 0

def create_sequences_with_time(data, sequence_length=10):
    """Tạo sequences cho PC-Transformer từ dữ liệu thời gian"""
    if len(data) < sequence_length:
        # Nếu không đủ dữ liệu, padding với dữ liệu đầu tiên
        padded_data = np.tile(data[0], (sequence_length - len(data), 1))
        data = np.vstack([padded_data, data])
    
    sequences = []
    time_steps = []
    targets = []
    
    for i in range(len(data) - sequence_length + 1):
        seq = data[i:(i + sequence_length)]
        target = data[i + sequence_length - 1, -1]  # Target là runtime của timestep cuối
        
        # Tạo time embedding (normalized position trong sequence)
        time_seq = np.arange(sequence_length).reshape(-1, 1) / sequence_length
        
        sequences.append(seq[:, :-1])  # Loại bỏ cột target khỏi features
        time_steps.append(time_seq)
        targets.append(target)
        
    return np.array(sequences), np.array(time_steps), np.array(targets)

def train_pctransformer_model(model, X_seq, time_seq, y_seq, epochs=30, learning_rate=0.001):
    """Train PC-Transformer model"""
    model.to(device)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_seq).to(device)
    time_tensor = torch.FloatTensor(time_seq).to(device)
    y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(device)
    
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor, time_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'PC-Transformer Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

def PCTransformerWalltimePredictor(finished_jobs, job, data_size=1000, use_user_estimate=False):
    """
    PC-Transformer Walltime Predictor - Interface tương tự KnnWalltimePredictor
    
    Parameters:
    - finished_jobs: Danh sách các công việc đã hoàn thành
    - job: Công việc cần dự đoán
    - data_size: Số lượng công việc gần đây nhất để xét
    - use_user_estimate: Có sử dụng ước tính của người dùng không
    
    Returns:
    - Thời gian dự đoán
    """
    global _pct_model, _pct_scaler, _pct_last_data_size
    
    # Kiểm tra dữ liệu
    neighbor_space = finished_jobs[-data_size:]
    if len(neighbor_space) < 20:  # Cần ít nhất 20 jobs cho PC-Transformer
        return job.requested_time  # PC-Transformer fallback - neutral
    
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
        if _pct_scaler is None or len(neighbor_space) != _pct_last_data_size:
            _pct_scaler = StandardScaler()
            data_scaled = _pct_scaler.fit_transform(data)
            _pct_last_data_size = len(neighbor_space)
        else:
            data_scaled = _pct_scaler.transform(data)
        
        # Tạo sequences với time embedding
        X_seq, time_seq, y_seq = create_sequences_with_time(data_scaled, sequence_length=10)
        
        # Train model nếu chưa có hoặc dữ liệu thay đổi
        if _pct_model is None or len(neighbor_space) != _pct_last_data_size:
            _pct_model = PCTransformerModel(
                input_dim=4,
                d_model=64,
                num_heads=8,
                d_ff=256,
                num_layers=4,
                output_dim=1,
                dropout=0.1
            )
            train_pctransformer_model(_pct_model, X_seq, time_seq, y_seq, epochs=30)
        
        # Chuẩn bị dữ liệu cho job cần dự đoán
        job_features = np.array([[
            job.requested_resources,
            job.requested_time,
            job.json_dict["exe_num"],
            job.json_dict["uid"],
            0  # Placeholder cho target
        ]])
        
        # Scale job features
        job_scaled = _pct_scaler.transform(job_features)
        
        # Tạo sequence cho prediction (sử dụng dữ liệu gần nhất)
        recent_data = data_scaled[-10:]  # Lấy 10 jobs gần nhất
        pred_sequence = np.vstack([recent_data[1:, :-1], job_scaled[0, :-1]])
        
        # Tạo time embedding cho prediction
        pred_time = np.arange(10).reshape(-1, 1) / 10
        
        # Predict
        _pct_model.eval()
        with torch.no_grad():
            pred_tensor = torch.FloatTensor(pred_sequence).unsqueeze(0).to(device)
            time_tensor = torch.FloatTensor(pred_time).unsqueeze(0).to(device)
            prediction = _pct_model(pred_tensor, time_tensor).cpu().numpy()[0, 0]
        
        # Inverse transform prediction
        pred_array = np.array([[0, 0, 0, 0, prediction]])
        pred_scaled = _pct_scaler.inverse_transform(pred_array)
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
            
            original_data = _pct_scaler.inverse_transform(dummy_features)
            actual_times = original_data[:, -1]
            request_times = original_data[:, 1]
            
            calibration_factor = np.mean(request_times / actual_times)
            predict = max(job.requested_time / calibration_factor, 1)
        
        # PC-Transformer-specific adjustment (slight over-prediction)
        predict = predict * 1.03
        
        return max(int(predict), 1)
        
    except Exception as e:
        print(f"PC-Transformer prediction error: {e}")
        return job.requested_time

# Optional: Function để clear cache khi cần
def clear_pctransformer_cache():
    """Clear PC-Transformer model cache"""
    global _pct_model, _pct_scaler, _pct_last_data_size
    _pct_model = None
    _pct_scaler = None
    _pct_last_data_size = 0