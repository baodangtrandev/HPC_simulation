import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

class HDEMWalltimePredictor:
    def __init__(self):
        # Cấu hình mặc định cho các model
        self.configs = {
            "knn": {"n_neighbors": 3, "weights": "uniform", "algorithm": "auto", "leaf_size": 30, "p": 2},
            "linearreg": {"fit_intercept": False, "copy_X": True, "n_jobs": -1},
            "mlp": {"hidden_layer_sizes": (100,), "activation": "relu", "solver": "adam", "max_iter": 500},
            "decisiontree": {"criterion": "squared_error", "max_depth": 20, "min_samples_split": 3},
            "randomforest": {"n_estimators": 100, "criterion": "squared_error", "max_depth": None, "min_samples_split": 2},
            "xgboost": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "verbosity": 0},  # Giảm verbosity xuống
            "gradientboosting": {"loss": "squared_error", "learning_rate": 0.1, "n_estimators": 100, "subsample": 1.0},
            "extratrees": {"n_estimators": 50, "max_depth": None, "min_samples_split": 2, "random_state": 42},
            "lightgbm": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": -1, "num_leaves": 31, "random_state": 42, "verbose": -1},  # Giảm verbosity xuống
            "catboost": {"iterations": 100, "learning_rate": 0.1, "depth": 6, "verbose": False},
            "lasso": {"alpha": 1.0, "fit_intercept": True, "max_iter": 1000, "tol": 0.0001, "random_state": 42},
            "svr": {"C": 1, "epsilon": 0.1, "kernel": "linear"},
            "elasticnet": {"alpha": 1, "l1_ratio": 0.5, "fit_intercept": True, "max_iter": 1000, "tol": 0.001},
            "ridge": {"alpha": 0.1, "fit_intercept": True, 'solver': 'auto'}
        }
        
        # Cache cho các model đã được huấn luyện
        self.trained_models = {}
        
        # Cache cho dữ liệu
        self.finished_jobs_dataset = None
        self.data_features = None
        
        # Cấu hình ensemble
        self.use_ensemble = False
        self.ensemble_models = []
        self.ensemble_weights = []
        self.meta_model = None
        
        # Tham số cho dynamic weighting
        self.window_size = 200
        self.lr = 0.1
        self.discount_factor = 0.6
        self.drift_threshold = 0.1
        
    # Hàm tạo model dựa trên tên
    def create_model(self, model_name):
        model_class = {
            "knn": KNeighborsRegressor,
            "linearreg": LinearRegression,
            "decisiontree": DecisionTreeRegressor,
            "randomforest": RandomForestRegressor,
            "xgboost": XGBRegressor,
            "gradientboosting": GradientBoostingRegressor,
            "mlp": MLPRegressor,
            "extratrees": ExtraTreesRegressor,
            "lightgbm": LGBMRegressor,
            "catboost": CatBoostRegressor,
            "lasso": Lasso,
            "svr": SVR,
            "elasticnet": ElasticNet,
            "ridge": Ridge,
        }.get(model_name)
        
        if model_class is None:
            raise ValueError(f"Model '{model_name}' không được hỗ trợ.")
        
        return model_class(**self.configs[model_name])
    
    # Xử lý dữ liệu từ finished_jobs để tạo dataset
    def prepare_dataset(self, finished_jobs, data_size=1000):
        dataset = []
        for finished_job in finished_jobs[-data_size:]:
            features = [
                finished_job.requested_resources,
                finished_job.requested_time,
                finished_job.json_dict["exe_num"],
                finished_job.json_dict["uid"],
                # Thêm các features khác nếu cần
            ]
            target = int(finished_job.profile)  # Giả sử profile chứa thời gian thực thi
            dataset.append((features, target))
        
        return dataset

    # Tạo các sub-ensembles 
    def setup_ensemble(self, model_groups):
        """
        Thiết lập ensemble với các nhóm model khác nhau
        
        model_groups: list[list[str]] - Danh sách các nhóm model, mỗi nhóm là 1 sub-ensemble
        """
        self.use_ensemble = True
        self.ensemble_models = []
        self.ensemble_weights = []
        
        for group in model_groups:
            models_in_group = []
            weights_in_group = []
            for model_name in group:
                model = self.create_model(model_name)
                models_in_group.append((model_name, model))
                weights_in_group.append(1.0 / len(group))
            
            self.ensemble_models.append(models_in_group)
            self.ensemble_weights.append(weights_in_group)
        
        # Meta model (default to GradientBoosting)
        self.meta_model = self.create_model("gradientboosting")
    
    # Lấy trung bình trọng số cho dự đoán từ sub-ensemble
    def weighted_prediction(self, models, weights, X):
        predictions = []
        for (model_name, model), weight in zip(models, weights):
            y_pred = model.predict([X])[0]
            predictions.append(y_pred * weight)
        return sum(predictions)
    
    # Hàm chính tương tự KnnWalltimePredictor
    def predict(self, finished_jobs, job, data_size=1000, use_user_estimate=False, model_name="knn"):
        """
        Dự đoán thời gian thực thi dựa trên các công việc đã hoàn thành trước đó
        
        Tham số:
        - finished_jobs: Danh sách các công việc đã hoàn thành
        - job: Công việc cần dự đoán thời gian
        - data_size: Số lượng công việc gần nhất để xem xét
        - use_user_estimate: Sử dụng ước tính của người dùng
        - model_name: Tên model để sử dụng (nếu không sử dụng ensemble)
        
        Trả về: Thời gian dự đoán
        """
        # Kiểm tra dữ liệu
        neighbor_space = finished_jobs[-data_size:]
        if len(neighbor_space) < 2:
            return job.requested_time
        
        # Chuẩn bị dữ liệu
        if self.finished_jobs_dataset is None or len(self.finished_jobs_dataset) != len(neighbor_space):
            dataset = []
            
            for finished_job in neighbor_space:
                features = [
                    finished_job.requested_resources,
                    finished_job.requested_time,
                    finished_job.json_dict["exe_num"],
                    finished_job.json_dict["uid"],
                    # Thêm các features khác nếu cần
                ]
                target = int(finished_job.profile)  # Thời gian thực thi thực tế
                dataset.append((features, target))
            
            self.finished_jobs_dataset = dataset
        
        # Tạo X_train, y_train
        X_train = np.array([x for x, y in self.finished_jobs_dataset])
        y_train = np.array([y for x, y in self.finished_jobs_dataset])
        
        # Thông tin công việc cần dự đoán
        job_info = np.array([
            job.requested_resources,
            job.requested_time,
            job.json_dict["exe_num"],
            job.json_dict["uid"],
            # Thêm các features khác nếu cần
        ])
        
        # Mặc định dự đoán bằng thời gian yêu cầu
        predict = job.requested_time
        
        # Nếu sử dụng ensemble
        if self.use_ensemble:
            # Train các model trong sub-ensemble nếu chưa được train
            sub_ensemble_preds = []
            
            for i, (models_in_group, weights_in_group) in enumerate(zip(self.ensemble_models, self.ensemble_weights)):
                # Train các model trong sub-ensemble
                for j, (model_name, model) in enumerate(models_in_group):
                    model_key = f"sub_{i}_{model_name}"
                    if model_key not in self.trained_models:
                        model.fit(X_train, y_train)
                        self.trained_models[model_key] = model
                    else:
                        model = self.trained_models[model_key]
                    
                    models_in_group[j] = (model_name, model)  # Cập nhật model đã train
                
                # Dự đoán từ sub-ensemble
                sub_pred = self.weighted_prediction(models_in_group, weights_in_group, job_info)
                sub_ensemble_preds.append(sub_pred)
            
            # Meta prediction
            if self.meta_model is None:
                predict = sum(sub_ensemble_preds) / len(sub_ensemble_preds)
            else:
                # Nếu có meta model, sử dụng nó để dự đoán
                if "meta_model" not in self.trained_models:
                    # Chuẩn bị dữ liệu cho meta model
                    meta_X_train = []
                    for x, _ in self.finished_jobs_dataset:
                        sub_preds = []
                        for models_in_group, weights_in_group in zip(self.ensemble_models, self.ensemble_weights):
                            sub_pred = self.weighted_prediction(models_in_group, weights_in_group, x)
                            sub_preds.append(sub_pred)
                        meta_X_train.append(sub_preds)
                    
                    # Train meta model
                    self.meta_model.fit(np.array(meta_X_train), y_train)
                    self.trained_models["meta_model"] = self.meta_model
                else:
                    self.meta_model = self.trained_models["meta_model"]
                
                # Dự đoán với meta model
                predict = self.meta_model.predict([sub_ensemble_preds])[0]
                
        else:
            # Sử dụng model đơn lẻ
            if model_name not in self.trained_models:
                model = self.create_model(model_name)
                model.fit(X_train, y_train)
                self.trained_models[model_name] = model
            else:
                model = self.trained_models[model_name]
            
            predict = model.predict([job_info])[0]
            
            # Hiệu chỉnh dự đoán nếu sử dụng ước tính người dùng (tương tự KnnWalltimePredictor)
            if use_user_estimate:
                # Extract actual runtimes from finished jobs
                actual_runtimes = y_train
                requested_times = X_train[:, 1]  # Giả sử thời gian yêu cầu ở cột thứ 2
                
                # Tính calibration factor
                calibration_factor = np.mean(requested_times / actual_runtimes)
                predict = max(job.requested_time // calibration_factor, 1)
        
        return max(predict, 1)  # Đảm bảo kết quả luôn lớn hơn 0
    
    # Utility functions
    def update_config(self, model_name, new_config):
        """Cập nhật cấu hình cho model"""
        if model_name in self.configs:
            self.configs[model_name] = {**self.configs[model_name], **new_config}
            # Reset trained model if exists
            if model_name in self.trained_models:
                del self.trained_models[model_name]
        return self
    
    def clear_cache(self):
        """Xóa cache để buộc train lại"""
        self.trained_models = {}
        self.finished_jobs_dataset = None
        return self

# Hàm wrapper tương tự KnnWalltimePredictor
def HDEMWalltimePrediction(finished_jobs, job, data_size=1000, use_user_estimate=False, model_name="xgboost"):
    """
    Wrapper function tương tự KnnWalltimePredictor nhưng sử dụng ML
    
    Parameters:
    - finished_jobs: Danh sách các công việc đã hoàn thành
    - job: Công việc cần dự đoán
    - data_size: Số lượng công việc gần đây nhất để xét
    - use_user_estimate: Có sử dụng ước tính của người dùng không
    - model_name: Tên model để sử dụng
    
    Returns:
    - Thời gian dự đoán
    """
    # Singleton pattern để tái sử dụng model đã train
    if not hasattr(HDEMWalltimePrediction, "_instance"):
        HDEMWalltimePrediction._instance = HDEMWalltimePredictor()
    
    predictor = HDEMWalltimePrediction._instance
    return predictor.predict(finished_jobs, job, data_size, use_user_estimate, model_name)

# Cấu hình ensemble


def setup_ml_ensemble(model_groups, meta_model_name="gradientboosting"):
    """
    Thiết lập ensemble cho dự đoán
    
    Parameters:
    - model_groups: Danh sách các nhóm model, mỗi nhóm là 1 sub-ensemble
    - meta_model_name: Tên model để sử dụng làm meta model
    """
    if not hasattr(HDEMWalltimePrediction, "_instance"):
        HDEMWalltimePrediction._instance = HDEMWalltimePredictor()
    
    predictor = HDEMWalltimePrediction._instance
    predictor.setup_ensemble(model_groups)
    predictor.meta_model = predictor.create_model(meta_model_name)
    predictor.clear_cache()  # Reset cache để chắc chắn train lại
    return predictor

# setup_ml_ensemble([
#     ["xgboost", "lightgbm", "catboost"],  # Sub-ensemble 1
#     ["randomforest", "extratrees", "gradientboosting"],  # Sub-ensemble 2
#     ["knn", "svr", "ridge"]  # Sub-ensemble 3
# ], meta_model_name="xgboost")
