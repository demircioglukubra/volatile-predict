import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class DataProcessor:
    def __init__(self, features_path, labels_path):
        self.features_path = features_path
        self.labels_path = labels_path

    def load_data(self):
        features = pd.read_csv(self.features_path, delimiter=";", header=0)
        labels = pd.read_csv(self.labels_path, delimiter=";", header=0)
        return features, labels

    def preprocess_data(self, features, labels):
        # Combine features and labels into a single DataFrame
        raw_data = pd.concat([features, labels['devol_yield']], axis=1)
        raw_data = raw_data.drop(columns=['atmosphere'])
        return raw_data


class FeatureEngineering:
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def calculate_weighted_characteristics(self):
        char_1_columns = ['wc_1', 'vm_1', 'fc_1', 'ac_1', 'c_1', 'h_1', 'o_1', 'n_1', 's_1', 'cl_1', 'hc_1', 'oc_1',
                          'lhv_1']
        char_2_columns = ['wc_2', 'vm_2', 'fc_2', 'ac_2', 'c_2', 'h_2', 'o_2', 'n_2', 's_2', 'cl_2', 'hc_2', 'oc_2',
                          'lhv_2']

        fuel_char_1 = self.raw_data[char_1_columns]
        fuel_char_2 = self.raw_data[char_2_columns]

        # Weighted sums
        fuel_char_1_weighted = self.raw_data['x_fuel1'].values[:, None] * fuel_char_1.values
        fuel_char_2_weighted = self.raw_data['x_fuel2'].values[:, None] * fuel_char_2.values

        # Total characteristics
        fuel_char_total = pd.DataFrame(
            fuel_char_1_weighted + fuel_char_2_weighted,
            columns=['wc', 'vm', 'fc', 'ac', 'c', 'h', 'o', 'n', 's', 'cl', 'hc', 'oc', 'lhv']
        )
        return fuel_char_total

    def generate_final_dataset(self, fuel_char_total):
        # Drop unnecessary columns
        columns_to_drop = [
            'wc_1', 'vm_1', 'fc_1', 'ac_1', 'c_1', 'h_1', 'o_1', 'n_1', 's_1', 'cl_1', 'hc_1', 'oc_1', 'lhv_1',
            'wc_2', 'vm_2', 'fc_2', 'ac_2', 'c_2', 'h_2', 'o_2', 'n_2', 's_2', 'cl_2', 'hc_2', 'oc_2', 'lhv_2',
            'x_fuel1', 'x_fuel2'
        ]
        self.raw_data = self.raw_data.drop(columns=columns_to_drop)

        # Add the weighted fuel characteristics
        final_data = pd.concat([self.raw_data, fuel_char_total], axis=1)
        return final_data
    def mixed_fuels(self):
        mixed_fuels = self.raw_data[self.raw_data.iloc[:, 1] != self.raw_data.iloc[:, 14]].reset_index(drop=True)
        return mixed_fuels
    def biomass_fuels(self):
        biomass_fuels = self.raw_data[self.raw_data.iloc[:, 1] == self.raw_data.iloc[:, 14]].reset_index(drop=True)
        return biomass_fuels

class OutlierRemoval:
    def __init__(self, data):
        self.data = data
    def iqr(self):
        q1 = self.data.quantile(0.25)
        q3 = self.data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return lower_bound, upper_bound
    def outlier_removal(self):
        lower_bound, upper_bound = self.iqr()
        filtered_data = self.data[(self.data >= lower_bound) & (self.data <= upper_bound)]
        return filtered_data


class DataVisualizer:
    @staticmethod
    def plot_distribution(data, z_scores, title, color):
        plt.figure(figsize=(10, 10))
        plt.hist(z_scores, color=color, bins=30, alpha=0.7)
        plt.title(title, fontsize=20)
        plt.show()

    @staticmethod
    def plot_correlation_heatmaps(data, methods=["pearson", "kendall", "spearman"], title_prefix=""):
        for method in methods:
            correlation = data.corr(method=method)
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
            plt.title(f"{title_prefix} {method.capitalize()} Correlation Heatmap", fontsize=14)
            plt.show()

    @staticmethod
    def plot_feature_importance(scores, feature_names):
        scores /= scores.max()  # Normalize scores
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(scores)), scores, alpha=0.7)
        plt.xticks(np.arange(len(scores)), feature_names, rotation=90)
        plt.title("Feature Univariate Scores")
        plt.ylabel("-Log(P-Value)")
        plt.tight_layout()
        plt.show()
