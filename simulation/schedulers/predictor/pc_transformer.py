import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

class PCTransformerWalltimePredictor:
    def __init__(self):
        # PC-Transformer specific configurations
        self.configs = {
            "pctransformer": {
                "input_dim": 4,
                "d_model": 64,
                "num_heads": 8,
                "d_ff": 256,
                "num_layers": 4,
                "output_dim": 1,
                "dropout": 0.1,
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
        
        # PC-Transformer specific cache
        self.sequence_data = None
        
    def create_sequences(self, data, sequence_length):
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
    
    def prepare_dataset(self, finished_jobs, data_size=1000):
        """Chuẩn bị dataset cho PC-Transformer"""
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
        """Tạo PC-Transformer model"""
        if model_name == "pctransformer":
            config = self.configs["pctransformer"]
            return PCTransformerModel(
                input_dim=config["input_dim"],
                d_model=config["d_model"],
                num_heads=config["num_heads"],
                d_ff=config["d_ff"],
                num_layers=config["num_layers"],
                output_dim=config["output_dim"],
                dropout=config["dropout"]
            )
        else:
            raise ValueError(f"Model '{model_name}' không được hỗ trợ.")
    
    def train_pctransformer_model(self, model, X_seq, time_seq, y_seq, config):
        """Train PC-Transformer model với early stopping"""
        model.to(device)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(device)
        time_tensor = torch.FloatTensor(time_seq).to(device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(device)
        
        # Split data for validation (80-20 split)
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        time_train, time_val = time_tensor[:split_idx], time_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
        
        criterion = nn.HuberLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            optimizer.zero_grad()
            train_outputs = model(X_train, time_train)
            train_loss = criterion(train_outputs, y_train)
            train_loss.backward()
            optimizer.step()
            
            # Validation phase
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val, time_val)
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
    
    def predict(self, finished_jobs, job, data_size=1000, use_user_estimate=False, model_name="pctransformer"):
        """
        Dự đoán thời gian thực thi dựa trên PC-Transformer
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
                X_seq, time_seq, y_seq = self.create_sequences(
                    self.finished_jobs_dataset, 
                    config["sequence_length"]
                )
                self.sequence_data = (X_seq, time_seq, y_seq)
            else:
                X_seq, time_seq, y_seq = self.sequence_data
            
            # Train model nếu chưa được train
            if model_name not in self.trained_models:
                model = self.create_model(model_name)
                self.train_pctransformer_model(model, X_seq, time_seq, y_seq, self.configs[model_name])
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
            seq_length = self.configs[model_name]["sequence_length"]
            recent_data = self.finished_jobs_dataset[-seq_length:]
            pred_sequence = np.vstack([recent_data[1:, :-1], job_scaled[0, :-1]])
            
            # Tạo time embedding cho prediction
            pred_time = np.arange(seq_length).reshape(-1, 1) / seq_length
            
            # Predict
            model.eval()
            with torch.no_grad():
                pred_tensor = torch.FloatTensor(pred_sequence).unsqueeze(0).to(device)
                time_tensor = torch.FloatTensor(pred_time).unsqueeze(0).to(device)
                prediction = model(pred_tensor, time_tensor).cpu().numpy()[0, 0]
            
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
            print(f"PC-Transformer prediction error: {e}")
            return job.requested_time
    
    def evaluate_model(self, model, X_seq, time_seq, y_seq, scaler):
        """Đánh giá model với RMSE, MAE, MSE, R2"""
        model.eval()
        predictions = []
        true_values = []
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(device)
            time_tensor = torch.FloatTensor(time_seq).to(device)
            outputs = model(X_tensor, time_tensor).cpu().numpy()
            
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
def PCTransformerWalltimePrediction(finished_jobs, job, data_size=1000, use_user_estimate=False, model_name="pctransformer"):
    """
    Wrapper function sử dụng PC-Transformer cho dự đoán walltime
    
    Parameters:
    - finished_jobs: Danh sách các công việc đã hoàn thành
    - job: Công việc cần dự đoán
    - data_size: Số lượng công việc gần đây nhất để xét
    - use_user_estimate: Có sử dụng ước tính của người dùng không
    - model_name: Tên model để sử dụng (mặc định "pctransformer")
    
    Returns:
    - Thời gian dự đoán
    """
    # Singleton pattern để tái sử dụng model đã train
    if not hasattr(PCTransformerWalltimePrediction, "_instance"):
        PCTransformerWalltimePrediction._instance = PCTransformerWalltimePredictor()
    
    predictor = PCTransformerWalltimePrediction._instance
    return predictor.predict(finished_jobs, job, data_size, use_user_estimate, model_name)

# Utility functions
def setup_pctransformer_config(input_dim=4, d_model=64, num_heads=8, d_ff=256, num_layers=4, 
                               output_dim=1, dropout=0.1, sequence_length=10, epochs=50, 
                               learning_rate=0.001, patience=5):
    """
    Thiết lập cấu hình cho PC-Transformer model
    """
    if not hasattr(PCTransformerWalltimePrediction, "_instance"):
        PCTransformerWalltimePrediction._instance = PCTransformerWalltimePredictor()
    
    predictor = PCTransformerWalltimePrediction._instance
    new_config = {
        "input_dim": input_dim,
        "d_model": d_model,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "num_layers": num_layers,
        "output_dim": output_dim,
        "dropout": dropout,
        "sequence_length": sequence_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "patience": patience
    }
    
    predictor.update_config("pctransformer", new_config)
    return predictor

def clear_pctransformer_cache():
    """
    Xóa cache để buộc train lại model
    """
    if hasattr(PCTransformerWalltimePrediction, "_instance"):
        PCTransformerWalltimePrediction._instance.clear_cache()

def evaluate_pctransformer_performance(finished_jobs, data_size=1000):
    """
    Đánh giá performance của PC-Transformer model
    """
    if not hasattr(PCTransformerWalltimePrediction, "_instance"):
        return None
    
    predictor = PCTransformerWalltimePrediction._instance
    
    if predictor.sequence_data is None:
        # Chuẩn bị dữ liệu trước
        predictor.prepare_dataset(finished_jobs, data_size)
        config = predictor.configs["pctransformer"]
        X_seq, time_seq, y_seq = predictor.create_sequences(
            predictor.finished_jobs_dataset, 
            config["sequence_length"]
        )
        predictor.sequence_data = (X_seq, time_seq, y_seq)
    
    if "pctransformer" not in predictor.trained_models:
        print("Model chưa được train. Vui lòng chạy prediction trước.")
        return None
    
    X_seq, time_seq, y_seq = predictor.sequence_data
    model = predictor.trained_models["pctransformer"]
    
    return predictor.evaluate_model(model, X_seq, time_seq, y_seq, predictor.scaler)

# Example usage và test
def test_pctransformer_model():
    """
    Test function để kiểm tra PC-Transformer model
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
    print("Testing PC-Transformer model...")
    prediction = PCTransformerWalltimePrediction(finished_jobs, test_job, model_name="pctransformer")
    print(f"Prediction: {prediction}")
    
    # Test với user estimate
    prediction_with_estimate = PCTransformerWalltimePrediction(
        finished_jobs, test_job, use_user_estimate=True, model_name="pctransformer"
    )
    print(f"Prediction with user estimate: {prediction_with_estimate}")
    
    # Evaluate performance
    metrics = evaluate_pctransformer_performance(finished_jobs)
    if metrics:
        rmse, mae, mse, r2 = metrics
        print(f"Model Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R2: {r2:.4f}")

# Uncomment để test
# test_pctransformer_model()
