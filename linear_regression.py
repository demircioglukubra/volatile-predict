import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from statsmodels.stats.outliers_influence import variance_inflation_factor



X_train = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\prepped_data\scaled_prepped\X_train.csv")
X_test = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\prepped_data\scaled_prepped\X_test.csv")
y_train = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\prepped_data\scaled_prepped\y_train.csv")
y_test = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\prepped_data\scaled_prepped\y_test.csv")

"""
"""
X_train = X_train.drop(columns=['vm', 'h', 'o', 'c', 'lhv'])
X_test = X_test.drop(columns=['vm', 'h', 'o', 'c', 'lhv'])


vif_data = pd.DataFrame()
vif_data["Feature"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
sorted_idx = vif_data['VIF'].argsort()[::-1]
for i in sorted_idx:
    print(f"{X_train.columns[i]}: {vif_data['VIF'][i]:.4f}")

# Handles multicollinearrity
ridge = Ridge(alpha=0.5)
ridge.fit(X_train, y_train)
print("Ridge Score:", ridge.score(X_test, y_test))

lasso =Lasso(alpha=0.5)
lasso.fit(X_train, y_train)
print("Lasso Score:", lasso.score(X_test, y_test))

elastic =ElasticNet(alpha=0.5)
elastic.fit(X_train, y_train)
print("ElasticNet Score:", elastic.score(X_test, y_test))