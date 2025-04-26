import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import shap
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib


'''
Feature Engineering Approach has been integrated (Multivariate Transformation & Feature Extraction)
'''

X_train = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\X_train.csv", delimiter=",")
X_test = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\X_test.csv" ,delimiter=",")
y_train = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\y_train.csv", delimiter=",")
y_test = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\y_test.csv", delimiter=",")


def remove_outliers(X, y, threshold=3):
    """
    Remove outliers from features and target using z-score method.

    Parameters:
    -----------
    X : DataFrame or array
        Feature matrix
    y : Series or array
        Target variable
    threshold : float, default=3
        Z-score threshold for outlier detection

    Returns:
    --------
    X_clean : DataFrame or array
        Feature matrix with outliers removed
    y_clean : Series or array
        Target variable with outliers removed
    """
    # Convert inputs to DataFrame/Series if they aren't already
    X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
    y_s = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()

    # Calculate z-scores for each feature
    z_scores = pd.DataFrame()
    for col in X_df.columns:
        z_scores[col] = (X_df[col] - X_df[col].mean()) / X_df[col].std()

    # Create a mask for rows where all features have z-scores within threshold
    mask = (z_scores.abs() < threshold).all(axis=1)

    # Also check for outliers in the target variable
    y_zscore = (y_s - y_s.mean()) / y_s.std()
    target_mask = y_zscore.abs() < threshold

    # Combine masks
    final_mask = mask & target_mask

    # Apply mask to both X and y
    X_clean = X_df[final_mask]
    y_clean = y_s[final_mask]

    removed_count = len(X_df) - len(X_clean)
    print(f"Removed {removed_count} outliers ({removed_count / len(X_df) * 100:.2f}% of data)")

    return X_clean, y_clean

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

print("Original data shape:", X_train.shape, y_train.shape)
if hasattr(y_train, 'shape') and len(y_train.shape) > 1 and y_train.shape[1] == 1:
    y_train = y_train.squeeze()

X_train, y_train = remove_outliers(X_train, y_train, threshold=3)
print("After outlier removal:", X_train.shape, y_train.shape)


print("Original data shape:", X_test.shape, y_test.shape)
if hasattr(y_test, 'shape') and len(y_test.shape) > 1 and y_test.shape[1] == 1:
    y_test = y_test.squeeze()

X_test, y_test = remove_outliers(X_test, y_test, threshold=3)
print("After outlier removal:", X_test.shape, y_test.shape)


feature_names = X_train.columns if isinstance(X_train, pd.DataFrame) else [f"Feature_{i}" for i in range(X_train.shape[1])]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

n_iter = 50
best_params = None
best_score = float('inf')

print("Starting Random Search...")
for i in range(n_iter):
    params = {
        'hidden_layer_sizes': (np.random.choice([50, 100, 150]),),
        'activation': np.random.choice(['relu', 'tanh', 'logistic']),
        'solver': np.random.choice(['adam', 'sgd']),
        'learning_rate_init': np.random.uniform(0.0001, 0.005),  # Correct assignment of a float value
    }

    fold_scores = []

    # Cross-validate using only the training set
    for train_index, val_index in kf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        model = MLPRegressor(
            hidden_layer_sizes=params['hidden_layer_sizes'],
            activation=params['activation'],
            solver=params['solver'],
            learning_rate_init=params['learning_rate_init'],
            alpha=0.05,
            random_state=42,
            max_iter= 5000
        )

        model.fit(X_train_fold, y_train_fold)

        # Prediction and evaluation
        y_pred = model.predict(X_val_fold)
        rmse = mean_squared_error(y_val_fold, y_pred)
        fold_scores.append(rmse)

    # Calculate mean RMSE for the combination
    mean_rmse = np.mean(fold_scores)
    print(f"Mean RMSE: {mean_rmse} for params {params}")

    # Save the best parameters and boosting rounds
    results.append((mean_rmse, params))
    if mean_rmse < best_score:
        best_score = mean_rmse
        best_params = params

# Train the final model on the entire training set with the best parameters
print("\nTraining final model with best parameters...")
final_model = MLPRegressor(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    activation=best_params['activation'],
    solver=best_params['solver'],
    learning_rate_init=best_params['learning_rate_init'],
    random_state=42,
    warm_start = True
)

final_model.fit(X_train, y_train)

# Evaluate on the test set
y_pred_test = final_model.predict(X_test)
test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nBest parameters (grid search): {best_params}")
print(f"\nTest Set Performance:")
print(f"Test RMSE: {test_rmse}")
print(f"Test R-Squared: {test_r2}")

joblib.dump(final_model, 'mlp_model.pkl')

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

