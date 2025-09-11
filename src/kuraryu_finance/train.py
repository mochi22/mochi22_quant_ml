import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


class LightGBMRegressorTrainer:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.models = []

    def train_one_fold(self, X_train, y_train, X_val, y_val):
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_eval = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1
        }

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            num_boost_round=100,
            # early_stopping_rounds=10,
            # verbose_eval=False
        )
        return model

    def run_cv(self, df, kfold):
        X = df[self.features]
        y = df[self.target]
        all_rmse = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
            print(f"\n🧪 Fold {fold + 1}")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[test_idx], y.iloc[test_idx]

            model = self.train_one_fold(X_train, y_train, X_val, y_val)
            self.models.append(model)

            y_pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            all_rmse.append(rmse)
            print(f"✅ RMSE: {rmse:.5f}")

        print(f"\n📊 Average RMSE across folds: {np.mean(all_rmse):.5f}")