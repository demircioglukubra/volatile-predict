import re
import shap
from xgboost import cv, DMatrix, train, XGBRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import graphviz
import os
import joblib


'''
Feature Engineering Approach has been integrated (Multivariate Transformation & Feature Extraction)
'''

X_train = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\X_train.csv", delimiter=",")
X_test = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\X_test.csv" ,delimiter=",")
y_train = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\y_train.csv", delimiter=",")
y_test = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\y_test.csv", delimiter=",")


feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"Feature_{i}" for i in range(X_train.shape[1])]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []


param_distributions = {
    'max_depth': [3,5, 7],
    'learning_rate': np.random.uniform(0.001, 0.2, 10),
    'subsample': np.random.uniform(0.6, 1.0, 10),
    'colsample_bytree': np.random.uniform(0.6, 1.0, 10),
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
        for train_index, val_index in kf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
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

final_model.fit(X_train, y_train)

# Evaluate on the test set
y_pred_test = final_model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
test_r2 = r2_score(y_test, y_pred_test)

graph = xgb.to_graphviz(final_model, num_trees=0, rankdir="LR")
# Render and view the updated tree
graph.render("xgb_tree_original_values", format="png", cleanup=True)
graph.view()

final_model.get_booster().dump_model('xgboost_tree_dump.txt')
joblib.dump(final_model, 'xgb.pkl')

importances = final_model.feature_importances_
feature_names = X_train.columns

# Sort feature importance
sorted_idx = importances.argsort()[::-1]
sorted_features = [feature_names[i] for i in sorted_idx]

# Plot
plt.figure(figsize=(10, 6))
plt.barh(sorted_features, importances[sorted_idx], color='royalblue')
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in XGBoost Model")
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.show()

explainer = shap.Explainer(final_model)
shap_values = explainer(X_train)

# Visualize the SHAP summary plot
shap.summary_plot(shap_values, X_train, feature_names=feature_names)

print(f"\nBest parameters (grid search): {best_params}")
print(f"\nTest Set Performance:")
print(f"Test RMSE: {test_rmse}")
print(f"Test R-Squared: {test_r2}")


# Determine the number of features
num_features = X_test.shape[1]

# Define the grid dimensions for subplots (e.g., 3 rows x 4 columns for 12 features)
n_cols = 4
n_rows = int(np.ceil(num_features / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
axes = axes.flatten()

# Loop over each feature and plot actual vs. predicted values
for i, feature in enumerate(X_test.columns):
    ax = axes[i]
    ax.scatter(X_test[feature], y_test, color='blue', alpha=0.5, label='Actual')
    ax.scatter(X_test[feature], y_pred_test, color='red', alpha=0.5, label='Predicted')
    ax.set_xlabel(feature)
    ax.set_ylabel('Target')
    ax.legend()
    ax.set_title(f"Feature: {feature}")

# Hide any unused subplots if the grid is larger than the number of features
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.6, label='Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Parity Plot)')
plt.legend()
plt.grid(True)
plt.show()

y_pred_test = y_pred_test.flatten()

residuals = y_test - y_pred_test

plt.figure(figsize=(7, 5))
plt.scatter(y_pred_test, residuals, color='red', alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(X_test['hc'], y_pred_test, color='red', alpha=0.6)
plt.scatter(X_test['hc'], y_test, color='blue', alpha=0.6)
plt.axhline(y=0, color='black', linestyle='--', lw=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True)
plt.show()

