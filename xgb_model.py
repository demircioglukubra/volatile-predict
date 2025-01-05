import optuna
from xgboost import XGBRegressor, callback
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class ModelOptimizer:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        """
        Initializes the optimizer, splits the data into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.study = None

    def objective(self, trial):
        """
        Defines the objective function for hyperparameter optimization.
        """
        # Suggest hyperparameters to tune
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,  # Ensures reproducibility
            'eval_metric': 'rmse',  # Add eval_metric here
        }

        # Train model
        model = XGBRegressor(**params)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_test, self.y_test)]
        )

        # Predict and calculate RMSE
        y_pred = model.predict(self.X_test)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        return rmse

    def run_optimization(self, n_trials=50):
        """
        Runs the hyperparameter optimization process.
        """
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=n_trials)

        return self.study.best_params, self.study.best_value
