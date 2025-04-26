from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import pandas as pd
from sklearn.compose import TransformedTargetRegressor

# Load data
X_train = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\X_train.csv", delimiter=",")
X_test = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\X_test.csv", delimiter=",")
y_train = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\y_train.csv", delimiter=",").values.ravel()
y_test = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\xgb_best_performers\bio+mix_R81\y_test.csv", delimiter=",").values.ravel()

X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

xgb_model = joblib.load('xgb.pkl')
mlp_model = joblib.load('mlp_model.pkl')

stacking_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_model),  # Burada TransformedTargetRegressor kullanmaya gerek yok
        ('mlp', mlp_model)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

# Create a pipeline with preprocessing + stacking
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('stacking', stacking_model)
])

# Fit and evaluate the model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Stacking Model Performance:", pipeline.score(X_test, y_test))
