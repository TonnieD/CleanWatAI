import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def load_water_data(filepath='../data/processed/gems.csv') -> pd.DataFrame:
    return pd.read_csv(filepath)

def split_features(dataframe: pd.DataFrame) -> tuple:
    sensor_features = ['pH', 'TEMP', 'EC', 'station_encoded']
    chemical_features = ['NO2N', 'NO3N', 'O2-Dis', 'NH4N']
    return sensor_features, chemical_features

def visualize_data_overview(dataframe, sensor_features, chemical_features):
    """Create essential data overview plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Correlation 
    all_features = sensor_features + chemical_features
    corr_matrix = dataframe[all_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=axes[0,0])
    axes[0,0].set_title('Feature Correlations')
    
    # Sensor features
    dataframe[sensor_features].hist(bins=30, ax=axes[0,1])
    axes[0,1].set_title('Sensor Distributions')
    
    # Chemical features  
    dataframe[chemical_features].hist(bins=30, ax=axes[1,0])
    axes[1,0].set_title('Chemical Distributions')
    plt.tight_layout()
    plt.show()
    
def remove_outliers_and_skewness(dataframe, chemical_features, method='iqr'):
    """Remove outliers and handle extreme skewness"""
    df_clean = dataframe.copy()
    outlier_stats = {}
    
    for chemical in chemical_features:
        original_count = len(df_clean)
        
        Q1 = df_clean[chemical].quantile(0.25)
        Q3 = df_clean[chemical].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (df_clean[chemical] >= lower_bound) & (df_clean[chemical] <= upper_bound)
        df_clean = df_clean[mask]
        
        outlier_stats[chemical] = {
            'removed': original_count - len(df_clean),
            'original_skew': dataframe[chemical].skew(),
            'cleaned_skew': df_clean[chemical].skew()
        }
    
    return df_clean, outlier_stats

def apply_transforms(dataframe, chemical_features):
    """Apply stronger transformations for extreme skewness"""
    df_transformed = dataframe.copy()
    
    for chemical in chemical_features:
        if df_transformed[chemical].skew() > 5:
            from scipy.stats import boxcox
            df_transformed[chemical] = np.sqrt(df_transformed[chemical] + 1)
            
    return df_transformed

def preprocess_features(dataframe, all_features) -> tuple:
    """Handle skewness and scale features"""
    X = dataframe[all_features].copy()
    
    chemical_cols = ['NO2N', 'NO3N', 'O2-Dis', 'NH4N', 'EC']
    for col in chemical_cols:
        X[col] = np.log1p(X[col])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

def train_full_isolation_forest(X_processed):
    """full isolation forest on all features"""
    iso_forest = IsolationForest(contamination=0.04, random_state=42)
    iso_forest.fit(X_processed)
    return iso_forest

def convert_to_risk_scores(anomaly_scores):
    min_score = anomaly_scores.min()
    max_score = anomaly_scores.max()
    risk_scores = (max_score - anomaly_scores) / (max_score - min_score)
    return risk_scores

def add_station_encoding(dataframe):
    df_enhanced = dataframe.copy()
    station_date_parts = df_enhanced['GEMS.Station.Number_Sample.Date'].str.split('_', expand=True)
    df_enhanced['station_id'] = station_date_parts[0]
    label_encoder = LabelEncoder()
    df_enhanced['station_encoded'] = label_encoder.fit_transform(df_enhanced['station_id'].fillna('UNKNOWN'))
    os.makedirs('models', exist_ok=True)
    encoder_path = '../models/gems_station_label_encoder.pkl'
    joblib.dump(label_encoder, encoder_path)
    return df_enhanced, label_encoder

def train_inference_isolation_forest(dataframe, sensor_features):
    """inference isolation forest on sensor-only + station_encoded features"""
    X_sensor = dataframe[sensor_features]
    scaler = StandardScaler()
    X_sensor_scaled = scaler.fit_transform(X_sensor)
    iso_forest_inference = IsolationForest(contamination=0.04, random_state=42)
    iso_forest_inference.fit(X_sensor_scaled)
    return iso_forest_inference, scaler

class WaterQualityPipeline:
    def __init__(self, model, scaler, min_score, max_score):
        self.model = model
        self.scaler = scaler
        self.min_score = min_score
        self.max_score = max_score

    def predict_risk(self, data: pd.DataFrame):
        data_scaled = self.scaler.transform(data)
        anomaly_scores = self.model.decision_function(data_scaled)
        risk_scores = self.convert_to_risk_scores(anomaly_scores)
        return risk_scores
    

    def convert_to_risk_scores(self, anomaly_scores):
        denominator = self.max_score - self.min_score
        if denominator == 0:
            return np.zeros_like(anomaly_scores)
            
        risk_scores = (self.max_score - anomaly_scores) / denominator
        return np.clip(risk_scores, 0, 1)
        
    def save(self, filepath="../models/water_quality_pipeline.pkl"):
        """Save the entire pipeline object."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"✅ Pipeline saved to {filepath}")

    @staticmethod
    def load(filepath="../models/water_quality_pipeline.pkl"):
        return joblib.load(filepath)
