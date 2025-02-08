import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import preprocess
from preprocess import DataProcessor, FeatureEngineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\feature_eng\distributed_fuels.csv", delimiter=';')
data = data.loc[data['devol_yield'] > 0]

data.loc[data['sample'].str.contains('gm|gumm', na=False), 'o'] = 0.1


h_o = data['h']/data['o']
T_time = data['residence_time']*data['temperature']
T_rate = data['temperature']/data['heat_rate']
vm_fc = data['vm']/data['fc']
ac_fc = data['ac']/data['fc']

data = data.drop(['h', 'o', 'c', 'wc', 'lhv', 'vm', 'ac', 'fc', 'heat_rate'], axis=1)

data = pd.concat([data, h_o.rename('H/O'), T_time.rename('T_time'), T_rate.rename('T_rate'), vm_fc.rename('VM/FC'), ac_fc.rename('AC/FC')], axis=1)

X_eng = data.drop(['sample','devol_yield'], axis=1)
y_eng = data['devol_yield']

X_eng_train, X_eng_test, y_eng_train, y_eng_test = train_test_split(X_eng, y_eng, test_size=0.2)

print(f"Train set: X_train={X_eng_train.shape}, y_train={y_eng_train.shape}")
print(f"Test set: X_test={X_eng_test.shape}, y_test={y_eng_test.shape}")

# Apply StandardScaler only if needed
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_eng_train)
X_test_scaled = scaler.transform(X_eng_test)

X_eng_train.to_csv(r'C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\feature_eng\X_train_engineered.csv', index=False)
X_eng_test.to_csv(r'C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\feature_eng\X_test_engineered.csv', index=False)
y_eng_train.to_csv(r'C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\feature_eng\y_train_engineered.csv', index=False)
y_eng_test.to_csv(r'C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\feature_eng\y_test_engineered.csv', index=False)