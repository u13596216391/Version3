import xgboost as xgb

class XGBoostModel:
    def __init__(self, params):
        self.model = xgb.XGBRegressor(**params)

    def train(self, X_train, y_train, X_val, y_val):
        print("Training XGBoost model...")
        # 修改这里，根据XGBoost版本使用正确的参数
        try:
            # 尝试新版本的API
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping=True,
                n_iter_no_change=50,  # 替代early_stopping_rounds
                verbose=False
            )
        except TypeError:
            # 如果失败，尝试旧版本API
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        print("XGBoost training complete.")

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model.load_model(path)