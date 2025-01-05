import preprocess
from preprocess import DataProcessor
from xgboost import cv, DMatrix, train, XGBRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# File paths
f_path = r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\features.csv"
l_path = r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\labels.csv"

# Load data and combine together the devolatile rate and fuel chars and kinetic parameters
data_processor = preprocess.DataProcessor(f_path, l_path)
features, labels = data_processor.load_data()
combined_data = data_processor.preprocess_data(features, labels)

# For mixed fuels calculate weighted fuel characteristics
feature_engineering = preprocess.FeatureEngineering(combined_data)
final_data_weighted = feature_engineering.calculate_weighted_characteristics()

# Split dataset into mixed_fuels and biomass_only datasets
mixed_fuels = feature_engineering.mixed_fuels()
biomass_fuels = feature_engineering.biomass_fuels()

mixed_fuels = preprocess.OutlierRemoval(mixed_fuels, 'devol_yield').filter_outliers()
biomass_fuels = preprocess.OutlierRemoval(biomass_fuels, 'devol_yield').filter_outliers()

y = final_data_weighted['devol_yield']
y = pd.DataFrame(y)
X = final_data_weighted.drop(columns=['devol_yield']).iloc[:, 1:]

data_prep = preprocess.DataPreparation(X, y)
X_train, X_test, y_train, y_test = data_prep.split_data()


# Convert data to DMatrix format for XGBoost
dtrain = DMatrix(X_train, label=y_train)
dtest = DMatrix(X_test, label=y_test)


# Define parameter distributions for random search
param_distributions = {
    'max_depth': [3, 5, 7],
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

# Perform random search
kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Use 3-fold cross-validation
results = []

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

        # Cross-validate for the current parameter combination
        for train_index, test_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
            y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

            # Train the model
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

# Train final model with the best parameters
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

final_model.fit(X_train, y_train)

# Make predictions and evaluate performance
y_pred = final_model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)**0.5
r_squared = r2_score(y_test, y_pred)

print(f"\nBest parameters (random search): {best_params}")
print(f"Optimal number of boosting rounds: {best_num_boost_round}")
print(f"Mean Squared Error: {rmse}")
print(f"R-Squared: {r_squared}")

