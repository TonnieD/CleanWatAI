"""
CleanWater AI: Production-Quality XGBoost Training Module

This module implements a 6-phase production-quality approach for water quality prediction
using WHO guidelines and ML best practices. All functions are modular and can be imported
into notebooks or used standalone.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# WHO Guidelines Definition
GUIDELINES = {
    'pH': {'type': 'range', 'min': 6.5, 'max': 8.5},
    'NO3N': {'type': 'max', 'limit': 50},
    'NO2N': {'type': 'max', 'limit': 3},
    'NH4N': {'type': 'max', 'limit': 1.5},
    'O2-Dis': {'type': 'min', 'limit': 5},
    'TP': {'type': 'max', 'limit': 0.05}
}

def _calculate_excursions(sample):
    """Calculates the excursion for each parameter in a single sample."""
    excursions = {}
    for param, guideline in GUIDELINES.items():
        if param not in sample or pd.isna(sample[param]):
            continue
            
        value = sample[param]
        
        failed = False
        if guideline['type'] == 'range':
            if not (guideline['min'] <= value <= guideline['max']):
                failed = True
                excursion = (value - guideline['max']) / (guideline['max'] - guideline['min']) if value > guideline['max'] else (guideline['min'] - value) / (guideline['max'] - guideline['min'])
        elif guideline['type'] == 'max':
            if value > guideline['limit']:
                failed = True
                excursion = (value / guideline['limit']) - 1
        elif guideline['type'] == 'min':
            if value < guideline['limit']:
                failed = True
                excursion = (guideline['limit'] / value) - 1
        
        if failed:
            excursions[param] = excursion
            
    return excursions

def _calculate_wqi_for_sample(sample):
    """Calculates the final WQI score for a single sample."""
    excursions = _calculate_excursions(sample)
    
    num_failed_params = len(excursions)
    if num_failed_params == 0:
        return 100.0
    
    f1_scope = (num_failed_params / len(GUIDELINES)) * 100
    f2_frequency = f1_scope
    
    sum_of_excursions = sum(excursions.values())
    nse = sum_of_excursions / len(GUIDELINES)
    f3_amplitude = (nse) / (0.01 * nse + 0.01)
    
    wqi_score = 100 - (np.sqrt(f1_scope**2 + f2_frequency**2 + f3_amplitude**2) / 1.732)
    
    return max(0, wqi_score)

def _categorize_wqi(score):
    """Assigns a category based on the WQI score."""
    if 95 <= score <= 100:
        return 'Excellent'
    elif 80 <= score < 95:
        return 'Good'
    elif 65 <= score < 80:
        return 'Fair'
    elif 45 <= score < 65:
        return 'Marginal'
    else:
        return 'Poor'

def create_wqi_labels(df):
    """
    Main function to add WQI score and category labels to the DataFrame.
    """
    df_copy = df.copy()
    print("💧 Calculating WQI scores for each sample...")
    df_copy['wqi_score'] = df_copy.apply(_calculate_wqi_for_sample, axis=1)
    
    print("🏷️ Assigning water quality categories...")
    df_copy['water_quality'] = df_copy['wqi_score'].apply(_categorize_wqi)
    
    print("✅ Phase 1 complete. WQI labels created.")
    return df_copy

def load_and_preprocess_gems_data(file_path='../data/processed/gems.csv'):
    """
    Load GEMS data and create necessary features for modeling.
    """
    print("📊 Loading GEMS data...")
    try:
        gems_df = pd.read_csv(file_path, index_col=0)
        print(f"✅ Loaded {len(gems_df)} samples with {gems_df.shape[1]} features")
        
        # Extract temporal and station features from index
        if gems_df.index.dtype == 'object':
            try:
                index_parts = gems_df.index.str.split('_', expand=True)
                if index_parts.shape[1] >= 2:
                    date_part = index_parts[1]
                    gems_df['date'] = pd.to_datetime(date_part, errors='coerce')
                    gems_df['year'] = gems_df['date'].dt.year
                    gems_df['month'] = gems_df['date'].dt.month
                    gems_df['station'] = index_parts[0]
                    
                    le = LabelEncoder()
                    gems_df['station_encoded'] = le.fit_transform(gems_df['station'])
                    
                    print(f"✅ Extracted temporal features: year range {gems_df['year'].min()}-{gems_df['year'].max()}")
                else:
                    print("⚠️ Could not extract date from index, creating dummy temporal features")
                    gems_df['year'] = 2020
                    gems_df['month'] = 6
                    gems_df['station_encoded'] = 0
            except Exception as e:
                print(f"⚠️ Error processing index: {e}, creating dummy temporal features")
                gems_df['year'] = 2020
                gems_df['month'] = 6
                gems_df['station_encoded'] = 0
        else:
            print("⚠️ Index is not string type, creating dummy temporal features")
            gems_df['year'] = 2020
            gems_df['month'] = 6
            gems_df['station_encoded'] = 0
        
        # Handle missing WHO parameters
        required_params = ['pH', 'NO2N', 'NO3N', 'TP', 'O2-Dis', 'NH4N']
        missing_params = [col for col in required_params if col not in gems_df.columns]
        
        if missing_params:
            print(f"⚠️ Missing WHO parameters: {missing_params}")
            print("🔧 Creating synthetic data for missing parameters...")
            
            np.random.seed(42)
            for param in missing_params:
                if param == 'pH':
                    gems_df[param] = np.random.normal(7.2, 0.5, len(gems_df))
                elif param == 'NO2N':
                    gems_df[param] = np.random.exponential(0.5, len(gems_df))
                elif param == 'NO3N':
                    gems_df[param] = np.random.exponential(2, len(gems_df))
                elif param == 'TP':
                    gems_df[param] = np.random.exponential(0.02, len(gems_df))
                elif param == 'O2-Dis':
                    gems_df[param] = np.random.normal(8, 2, len(gems_df))
                elif param == 'NH4N':
                    gems_df[param] = np.random.exponential(0.3, len(gems_df))
        
        print(f"📊 Dataset shape: {gems_df.shape}")
        return gems_df
        
    except FileNotFoundError:
        print("❌ GEMS data file not found. Please ensure gems.csv exists in data/processed/")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise

def perform_stratified_split(gems_labeled, feature_cols, test_set_path='../data/processed/gems_test_set.csv'):
    """
    Perform stratified data splitting into train/val/test sets.
    """
    print("🔀 Performing stratified data splitting...")
    
    X = gems_labeled[feature_cols]
    y = gems_labeled['water_quality']
    
    print(f"📊 Feature matrix shape: {X.shape}")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    
    # First split: 70% train, 30% temp
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(splitter1.split(X, y))
    
    # Second split: Split the 30% temp into 15% val + 15% test
    X_temp = X.iloc[temp_idx]
    y_temp = y.iloc[temp_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_temp, test_idx_temp = next(splitter2.split(X_temp, y_temp))
    
    # Convert temp indices back to original indices
    val_idx = temp_idx[val_idx_temp]
    test_idx = temp_idx[test_idx_temp]
    
    # Create data splits
    train_df = gems_labeled.iloc[train_idx].copy()
    val_df = gems_labeled.iloc[val_idx].copy()
    test_df = gems_labeled.iloc[test_idx].copy()
    
    print(f"✅ Data splitting completed:")
    print(f"   📚 Training set: {len(train_df)} samples ({len(train_df)/len(gems_labeled)*100:.1f}%)")
    print(f"   🔍 Validation set: {len(val_df)} samples ({len(val_df)/len(gems_labeled)*100:.1f}%)")
    print(f"   🧪 Test set: {len(test_df)} samples ({len(test_df)/len(gems_labeled)*100:.1f}%)")
    
    # Verify stratification
    print(f"\n🎯 Target distribution verification:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = split_df['water_quality'].value_counts(normalize=True).round(3)
        print(f"   {split_name}: {dist.to_dict()}")
    
    # Save test set
    test_df.to_csv(test_set_path)
    print(f"💾 Test set saved to {test_set_path}")
    
    print("✅ Phase 2 completed successfully")
    return train_df, val_df, test_df

def create_preprocessing_pipeline(feature_cols, preprocessor_path='../models/preprocessor.pkl'):
    """
    Create and return preprocessing pipeline for features.
    """
    print("🔧 Creating preprocessing pipeline...")
    
    numerical_features = ['pH', 'NO2N', 'NO3N', 'TP', 'O2-Dis', 'NH4N', 'year', 'month']
    categorical_features = ['station_encoded']
    
    print(f"📊 Numerical features: {numerical_features}")
    print(f"🏷️ Categorical features: {categorical_features}")
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def preprocess_data(train_df, val_df, test_df, feature_cols, preprocessor, preprocessor_path='../models/preprocessor.pkl'):
    """
    Apply preprocessing pipeline to train/val/test data.
    """
    print("🔄 Applying preprocessing pipeline...")
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df['is_safe']
    X_val = val_df[feature_cols]
    y_val = val_df['is_safe']
    X_test = test_df[feature_cols]
    y_test = test_df['is_safe']
    
    print(f"📊 Training features shape: {X_train.shape}")
    print(f"🎯 Training target distribution: {y_train.value_counts().to_dict()}")
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"✅ Preprocessing completed:")
    print(f"   📚 X_train_processed shape: {X_train_processed.shape}")
    print(f"   🔍 X_val_processed shape: {X_val_processed.shape}")
    print(f"   🧪 X_test_processed shape: {X_test_processed.shape}")
    
    # Get feature names
    numerical_features = ['pH', 'NO2N', 'NO3N', 'TP', 'O2-Dis', 'NH4N', 'year', 'month']
    categorical_features = ['station_encoded']
    feature_names = (numerical_features + 
                    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))
    print(f"🏷️ Final feature names: {feature_names}")
    
    # Save preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    print(f"💾 Preprocessor saved to {preprocessor_path}")
    
    print("✅ Phase 3 completed successfully")
    return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, feature_names

def train_contender_models(X_train_processed, y_train, X_val_processed, y_val):
    """
    Train and evaluate RandomForest vs XGBoost models.
    """
    print("🏆 Training contender models...")
    
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        model.fit(X_train_processed, y_train)
        
        y_val_pred = model.predict(X_val_processed)
        y_val_proba = model.predict_proba(X_val_processed)[:, 1]
        
        f1 = f1_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, y_val_proba)
        
        results[name] = {
            'model': model,
            'f1_score': f1,
            'auc_score': auc,
            'predictions': y_val_pred,
            'probabilities': y_val_proba
        }
        
        print(f"✅ {name} - F1: {f1:.4f}, AUC: {auc:.4f}")
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = results[best_model_name]['model']
    
    print(f"\n🏆 Best performing model: {best_model_name}")
    print(f"   F1 Score: {results[best_model_name]['f1_score']:.4f}")
    print(f"   AUC Score: {results[best_model_name]['auc_score']:.4f}")
    
    print("✅ Phase 4 completed successfully")
    return results, best_model_name, best_model

def tune_hyperparameters(best_model_name, X_train_processed, y_train, X_val_processed, y_val):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    print("🎛️ Performing hyperparameter tuning...")
    
    if best_model_name == 'RandomForest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestClassifier(random_state=42)
    elif best_model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    print(f"🔍 Tuning {best_model_name} with parameters: {param_grid}")
    
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    print("🔄 Running GridSearchCV...")
    grid_search.fit(X_train_processed, y_train)
    
    final_model = grid_search.best_estimator_
    print(f"✅ Best parameters found: {grid_search.best_params_}")
    print(f"✅ Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # Evaluate tuned model
    y_val_pred_tuned = final_model.predict(X_val_processed)
    y_val_proba_tuned = final_model.predict_proba(X_val_processed)[:, 1]
    
    f1_tuned = f1_score(y_val, y_val_pred_tuned)
    auc_tuned = roc_auc_score(y_val, y_val_proba_tuned)
    
    print(f"\n📈 Tuned model validation performance:")
    print(f"   F1 Score: {f1_tuned:.4f}")
    print(f"   AUC Score: {auc_tuned:.4f}")
    
    print("✅ Phase 5 completed successfully")
    return final_model, grid_search, f1_tuned, auc_tuned

def final_evaluation(final_model, preprocessor, feature_cols, best_model_name, 
                    test_set_path='../data/processed/gems_test_set.csv'):
    """
    Perform final evaluation on unseen test data.
    """
    print("🧪 Final evaluation on unseen test data...")
    
    test_df_final = pd.read_csv(test_set_path, index_col=0)
    X_test_final = test_df_final[feature_cols]
    y_test_final = test_df_final['is_safe']
    
    X_test_final_processed = preprocessor.transform(X_test_final)
    
    y_test_pred_final = final_model.predict(X_test_final_processed)
    y_test_proba_final = final_model.predict_proba(X_test_final_processed)[:, 1]
    
    final_f1 = f1_score(y_test_final, y_test_pred_final)
    final_auc = roc_auc_score(y_test_final, y_test_proba_final)
    
    print(f"🎯 FINAL TEST RESULTS:")
    print(f"   Model: {best_model_name}")
    print(f"   F1 Score: {final_f1:.4f}")
    print(f"   AUC Score: {final_auc:.4f}")
    
    print(f"\n📊 Final Classification Report:")
    print(classification_report(y_test_final, y_test_pred_final, 
                              target_names=['Unsafe', 'Safe']))
    
    print("✅ Phase 6 completed successfully")
    return final_f1, final_auc, y_test_final, y_test_pred_final

def save_model_artifacts(final_model, best_model_name, grid_search, final_f1, final_auc, 
                        feature_names, train_df, val_df, test_df_final):
    """
    Save all model artifacts and metadata.
    """
    # Save final model
    model_path = f'../models/{best_model_name.lower()}_final.pkl'
    joblib.dump(final_model, model_path)
    print(f"💾 Final model saved to {model_path}")
    
    # Create model info
    model_info = {
        'model_type': best_model_name,
        'best_params': grid_search.best_params_,
        'final_f1_score': final_f1,
        'final_auc_score': final_auc,
        'feature_names': feature_names,
        'training_samples': len(train_df),
        'validation_samples': len(val_df),
        'test_samples': len(test_df_final),
        'who_guidelines_used': GUIDELINES
    }
    
    # Save model info
    with open('../models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    print(f"📋 Model information saved to models/model_info.json")
    print(f"\n💾 Model Artifacts Created:")
    print(f"   • Preprocessor: models/preprocessor.pkl")
    print(f"   • Final Model: models/{best_model_name.lower()}_final.pkl")
    print(f"   • Test Set: data/processed/gems_test_set.csv")
    print(f"   • Model Info: models/model_info.json")

def visualize_wqi_distribution(gems_labeled):
    """
    Create visualizations for WQI score and category distributions.
    """
    category_counts = gems_labeled['water_quality'].value_counts()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(gems_labeled['wqi_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('WQI Score Distribution')
    plt.xlabel('WQI Score')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    category_counts.plot(kind='bar', color='lightcoral', alpha=0.7)
    plt.title('Water Quality Categories')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 3, 3)
    safe_counts = gems_labeled['is_safe'].value_counts()
    safe_counts.index = ['Unsafe', 'Safe']
    safe_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
    plt.title('Binary Safety Classification')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.show()

def visualize_confusion_matrix(y_true, y_pred, model_name, title_suffix=""):
    """
    Create confusion matrix visualization.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Unsafe', 'Safe'], yticklabels=['Unsafe', 'Safe'])
    plt.title(f'{model_name} - Confusion Matrix{title_suffix}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def visualize_feature_importance(final_model, feature_names, model_name):
    """
    Create feature importance visualization.
    """
    if hasattr(final_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Feature Importance (Top 10):")
        print(feature_importance.head(10))
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title(f'{model_name} - Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    return None

def visualize_performance_evolution(results, best_model_name, f1_tuned, auc_tuned, final_f1, final_auc):
    """
    Create performance evolution visualization across phases.
    """
    performance_summary = pd.DataFrame({
        'Dataset': ['Validation (Pre-tuning)', 'Validation (Post-tuning)', 'Test (Final)'],
        'F1_Score': [results[best_model_name]['f1_score'], f1_tuned, final_f1],
        'AUC_Score': [results[best_model_name]['auc_score'], auc_tuned, final_auc]
    })
    
    print(f"\n📈 Performance Summary:")
    print(performance_summary.round(4))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(performance_summary.index, performance_summary['F1_Score'], 
             marker='o', linewidth=2, markersize=8)
    plt.title('F1 Score Evolution')
    plt.ylabel('F1 Score')
    plt.xlabel('Phase')
    plt.xticks(performance_summary.index, performance_summary['Dataset'], rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(performance_summary.index, performance_summary['AUC_Score'], 
             marker='s', linewidth=2, markersize=8, color='orange')
    plt.title('AUC Score Evolution')
    plt.ylabel('AUC Score')
    plt.xlabel('Phase')
    plt.xticks(performance_summary.index, performance_summary['Dataset'], rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return performance_summary

def run_complete_pipeline(data_path='../data/processed/gems.csv'):
    """
    Execute the complete 6-phase water quality prediction pipeline.
    """
    print("🚀 Starting Complete Water Quality Prediction Pipeline")
    print("="*60)
    
    # Phase 1: Load data and create WQI labels
    gems_df = load_and_preprocess_gems_data(data_path)
    gems_labeled = create_wqi_labels(gems_df)
    
    # Display WQI distribution
    print(f"\n📊 WQI Score Distribution:")
    print(f"Mean WQI: {gems_labeled['wqi_score'].mean():.2f}")
    print(f"Std WQI: {gems_labeled['wqi_score'].std():.2f}")
    
    # Create binary target
    gems_labeled['is_safe'] = (gems_labeled['wqi_score'] >= 65).astype(int)
    print(f"\n🎯 Binary classification target:")
    print(f"Safe water samples: {gems_labeled['is_safe'].sum()} ({gems_labeled['is_safe'].mean()*100:.1f}%)")
    
    # Phase 2: Data splitting
    feature_cols = ['pH', 'NO2N', 'NO3N', 'TP', 'O2-Dis', 'NH4N', 'year', 'month', 'station_encoded']
    train_df, val_df, test_df = perform_stratified_split(gems_labeled, feature_cols)
    
    # Phase 3: Preprocessing
    preprocessor = create_preprocessing_pipeline(feature_cols)
    X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test, feature_names = preprocess_data(
        train_df, val_df, test_df, feature_cols, preprocessor
    )
    
    # Phase 4: Model training
    results, best_model_name, best_model = train_contender_models(
        X_train_processed, y_train, X_val_processed, y_val
    )
    
    # Phase 5: Hyperparameter tuning
    final_model, grid_search, f1_tuned, auc_tuned = tune_hyperparameters(
        best_model_name, X_train_processed, y_train, X_val_processed, y_val
    )
    
    # Phase 6: Final evaluation
    final_f1, final_auc, y_test_final, y_test_pred_final = final_evaluation(
        final_model, preprocessor, feature_cols, best_model_name
    )
    
    # Save artifacts
    test_df_final = pd.read_csv('../data/processed/gems_test_set.csv', index_col=0)
    save_model_artifacts(final_model, best_model_name, grid_search, final_f1, final_auc, 
                        feature_names, train_df, val_df, test_df_final)
    
    print("\n" + "="*60)
    print("🎉 PRODUCTION-QUALITY MODEL TRAINING COMPLETED!")
    print("="*60)
    print(f"✅ Final Model: {best_model_name}")
    print(f"✅ Test F1 Score: {final_f1:.4f}")
    print(f"✅ Test AUC Score: {final_auc:.4f}")
    print(f"✅ All 6 phases completed successfully")
    print("="*60)
    
    return {
        'final_model': final_model,
        'preprocessor': preprocessor,
        'feature_names': feature_names,
        'results': results,
        'best_model_name': best_model_name,
        'final_metrics': {'f1': final_f1, 'auc': final_auc}
    }

if __name__ == "__main__":
    pipeline_results = run_complete_pipeline()
