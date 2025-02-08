import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import preprocess
from preprocess import DataProcessor, FeatureEngineering

data = pd.read_csv(r"C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\feature_eng\distributed_fuels.csv", delimiter=';')
data = data.loc[data['devol_yield'] > 0]

data.loc[data['sample'].str.contains('gm|gumm', na=False), 'o'] = 0.1


h_o = data['h']/data['o']
T_time = data['residence_time']*data['temperature']
T_rate = data['temperature']/data['heat_rate']
vm_fc = data['vm']/data['fc']
ac_fc = data['ac']/data['fc']

data = data.drop(['h', 'o', 'c', 'wc', 'lhv', 'vm', 'ac', 'fc', 'heat_rate'], axis=1)

# Add new features
data = pd.concat([data, h_o.rename('H/O'), T_time.rename('T_time'), T_rate.rename('T_rate'), vm_fc.rename('VM/FC'), ac_fc.rename('AC/FC')], axis=1)
print(data.columns)

data.to_csv(r'C:\Users\demir\OneDrive\Desktop\MSc Thesis\Data\feature_eng\feature_engineered.csv', index=False)