import sys
sys.path.append('/path/to/other-branch-folder/')
import preprocess
import xgboost as xgb
from xgboost import cv, DMatrix, train, XGBRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

X_path = r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\prepped_data\preprocessed_features.csv"
y_path = r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\prepped_data\preprocessed_labels.csv"

X,y = preprocess.load_data(X_path,y_path)
X = X.iloc[:, 1:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test and train data as CSV files (optional, if needed for persistence)
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\prepped_data\scaled_prepped\X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(r'C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\prepped_data\scaled_prepped\X_test.csv', index=False)
y_train.to_csv(r'C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\prepped_data\scaled_prepped\y_train.csv', index=False)
y_test.to_csv(r'C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\prepped_data\scaled_prepped\y_test.csv', index=False)

# Placeholder to visualize feature names if needed
feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]

# Update XGBoost cross-validation process to use only training data
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

# Define parameter distributions for random search
param_distributions = {
    'max_depth': [5, 7, 9],
    'learning_rate': np.random.uniform(0.01, 0.2, 10),
    'subsample': np.random.uniform(0.8, 1.0, 10),
    'colsample_bytree': np.random.uniform(0.8, 1.0, 10),
    'gamma': np.random.uniform(0, 1, 10),
    'alpha': np.random.uniform(0, 1, 10),
    'min_child_weight': np.random.uniform(1, 3, 10),
}

# Number of random configurations and boosting rounds to test
n_iter = 50
boost_rounds = [100, 200, 300]

# Initialize placeholders for results
best_params = None
best_score = float('inf')
best_num_boost_round = None

print("Starting Random Search...")
for i in range(n_iter):
    # Randomly sample parameters
    params = {
        'max_depth': np.random.choice(param_distributions['max_depth']),
        'learning_rate': np.random.choice(param_distributions['learning_rate']),
        'subsample': np.random.choice(param_distributions['subsample']),
        'colsample_bytree': np.random.choice(param_distributions['colsample_bytree']),
        'gamma': np.random.choice(param_distributions['gamma']),
        'alpha': np.random.choice(param_distributions['alpha']),
        'min_child_weight': np.random.choice(param_distributions['min_child_weight']),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
    }

    for num_boost_round in boost_rounds:
        print(f"Testing params: {params} with num_boost_round={num_boost_round}")
        fold_scores = []

        # Cross-validate using only the training set
        for train_index, val_index in kf.split(X_train_scaled, y_train):
            X_train_fold, X_val_fold = X_train_scaled[train_index], X_train_scaled[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model = XGBRegressor(
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                objective='reg:squarederror',
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                alpha=params['alpha'],
                min_child_weight=params['min_child_weight'],
                n_estimators=num_boost_round,
                eval_metric='rmse',  # Evaluation metric for validation
                random_state=42
            )

            model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)

            # Prediction and evaluation
            y_pred = model.predict(X_val_fold)
            rmse = mean_squared_error(y_val_fold, y_pred)
            fold_scores.append(rmse)

        # Calculate mean RMSE for the combination
        mean_rmse = np.mean(fold_scores)
        print(f"Mean RMSE: {mean_rmse} for params {params} and num_boost_round={num_boost_round}")

        # Save the best parameters and boosting rounds
        results.append((mean_rmse, params, num_boost_round))
        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params
            best_num_boost_round = num_boost_round

# Train the final model on the entire training set with the best parameters
print("\nTraining final model with best parameters...")
final_model = XGBRegressor(
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    gamma=best_params['gamma'],
    alpha=best_params['alpha'],
    min_child_weight=best_params['min_child_weight'],
    objective='reg:squarederror',
    n_estimators=best_num_boost_round,
    eval_metric='rmse',
    random_state=42
)

final_model.fit(X_train_scaled, y_train)

# Evaluate on the test set
y_pred_test = final_model.predict(X_test_scaled)
test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nBest parameters (grid search): {best_params}")
print(f"\nTest Set Performance:")
print(f"Test RMSE: {test_rmse}")
print(f"Test R-Squared: {test_r2}")