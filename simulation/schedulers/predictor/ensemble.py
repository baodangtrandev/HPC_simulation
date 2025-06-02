import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression


class HistoricalModel:
    def __init__(self):
        self.user_jobs = {}

    def predict(self, user_id, finished_jobs):
        if user_id not in self.user_jobs:
            self.user_jobs[user_id] = []

        user_recent_jobs = self.user_jobs[user_id]
        if len(user_recent_jobs) < 5:
            return np.mean([job.actual_runtime for job in finished_jobs])

        prediction_deviation = np.mean(
            [
                job.actual_runtime - job.predicted_runtime
                for job in user_recent_jobs[-5:]
            ]
        )
        return prediction_deviation

    def update(self, user_id, job):
        if user_id not in self.user_jobs:
            self.user_jobs[user_id] = []
        self.user_jobs[user_id].append(job)


class EnsemblePredictor:
    def __init__(self, finished_jobs, job):
        self.finished_jobs = finished_jobs
        self.job = job

        self.historical_model = HistoricalModel()
        self.knn_model = KNeighborsRegressor(n_neighbors=5)
        self.rf_model = RandomForestRegressor(n_estimators=100)
        self.dnn_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500)

        self.meta_learner = LinearRegression()

    def fit(self):
        features, targets = self._extract_features_targets(self.finished_jobs)
        self.knn_model.fit(features, targets)
        self.rf_model.fit(features, targets)
        self.dnn_model.fit(features, targets)

        predictions = np.column_stack(
            (
                self.historical_model.predict(self.job.user_id, self.finished_jobs),
                self.knn_model.predict(features),
                self.rf_model.predict(features),
                self.dnn_model.predict(features),
            )
        )
        self.meta_learner.fit(predictions, targets)

    def predict(self):
        features = self._extract_features(self.job)
        predictions = np.array(
            [
                self.historical_model.predict(self.job.user_id, self.finished_jobs),
                self.knn_model.predict([features]),
                self.rf_model.predict([features]),
                self.dnn_model.predict([features]),
            ]
        )
        return self.meta_learner.predict([predictions])

    def _extract_features_targets(self, jobs):
        features = []
        targets = []
        for job in jobs:
            features.append(self._extract_features(job))
            targets.append(job.actual_runtime)
        return np.array(features), np.array(targets)

    def _extract_features(self, job):
        return [job.requested_resources, job.user_requested_time, job.submit_time]
