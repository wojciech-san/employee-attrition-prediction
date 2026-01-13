"""
Employee Attrition Prediction - Model Training Script

This script trains the final Random Forest model for employee attrition prediction
based on the exploratory data analysis notebook.

The script:
1. Loads and merges raw data from multiple sources
2. Performs feature engineering (working hours, overtime, absences)
3. Handles missing values and drops weak predictors
4. Prepares train/validation/test datasets
5. Trains the final Random Forest model
6. Saves the trained model and feature vectorizer to a pickle file
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def load_raw_data(data_dir='./data'):
    """Load all raw data files"""
    print("Loading raw data...")
    
    employee_survey_data = pd.read_csv(f'{data_dir}/employee_survey_data.csv')
    general_data = pd.read_csv(f'{data_dir}/general_data.csv')
    manager_survey_data = pd.read_csv(f'{data_dir}/manager_survey_data.csv')
    in_time = pd.read_csv(f'{data_dir}/in_time.csv').rename(columns={'Unnamed: 0': 'EmployeeID'}).set_index('EmployeeID')
    out_time = pd.read_csv(f'{data_dir}/out_time.csv').rename(columns={'Unnamed: 0': 'EmployeeID'}).set_index('EmployeeID')
    
    return employee_survey_data, general_data, manager_survey_data, in_time, out_time


def engineer_time_features(in_time, out_time):
    """Engineer features from in_time and out_time data"""
    print("Engineering time-based features...")
    
    in_time = in_time.apply(pd.to_datetime)
    out_time = out_time.apply(pd.to_datetime)
    
    # Calculate actual working hours
    actual_working_hours = out_time - in_time
    working_hours_float = actual_working_hours.apply(lambda x: x.dt.total_seconds() / 3600)
    
    # Extract features
    mean_hours = working_hours_float.mean(axis=1)
    overtime_days = (working_hours_float > 8).sum(axis=1)
    absent_days = in_time.isnull().sum(axis=1)
    
    time_features = pd.DataFrame({
        'EmployeeID': mean_hours.index,
        'Mean_Work_Hours': mean_hours.values,
        'Overtime_Days_Count': overtime_days.values,
        'Absent_Days_Count': absent_days.values
    })
    
    return time_features


def merge_data(general_data, employee_survey_data, manager_survey_data, time_features):
    """Merge all data sources"""
    print("Merging datasets...")
    
    df_final = general_data.merge(employee_survey_data, on='EmployeeID', how='left')
    df_final = df_final.merge(manager_survey_data, on='EmployeeID', how='left')
    df_final = df_final.merge(time_features, on='EmployeeID', how='left')
    
    return df_final


def prepare_data(df_final):
    """Prepare data: drop weak predictors, handle nulls, encode target"""
    print("Preparing data for modeling...")
    
    # Weak predictors (low correlation with attrition)
    weak_predictors = [
        'EmployeeID', 'Over18', 'StandardHours', 'EmployeeCount',
        'Absent_Days_Count'
    ]
    
    cols_to_drop = [c for c in weak_predictors if c in df_final.columns]
    df_model = df_final.drop(columns=cols_to_drop)
    
    print(f"Dropped {len(cols_to_drop)} weak predictors; df_model shape: {df_model.shape}")
    
    # Identify categorical and numerical columns
    categorical_columns = [
        col for col in list(df_model.dtypes[df_model.dtypes == "object"].index)
        if col != 'Attrition' and col != 'Target'
    ]
    
    numerical_columns = [
        col for col in list(df_model.dtypes[df_model.dtypes != "object"].index)
    ]
    
    # Handle missing values in numerical columns
    for col in numerical_columns:
        df_model[col] = df_model[col].fillna(df_model[col].median())
    
    # Encode target variable
    df_model['Target'] = df_model['Attrition'].map({'Yes': 1, 'No': 0})
    df_model = df_model.drop(columns=['Attrition'])
    
    return df_model, categorical_columns, numerical_columns


def create_train_val_test_split(df_model, test_size=0.2, val_size=0.25, random_state=1):
    """Create train, validation, and test datasets"""
    print("Creating train/validation/test splits...")
    
    df_full_train, df_test = train_test_split(
        df_model, test_size=test_size, random_state=random_state
    )
    
    df_train, df_val = train_test_split(
        df_full_train, test_size=val_size, random_state=random_state
    )
    
    # Reset indices
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_full_train = df_full_train.reset_index(drop=True)
    
    # Extract target variables
    y_train = df_train['Target'].values
    y_val = df_val['Target'].values
    y_test = df_test['Target'].values
    y_full_train = df_full_train['Target'].values
    
    # Drop target from features
    df_train = df_train.drop(columns=['Target'])
    df_val = df_val.drop(columns=['Target'])
    df_test = df_test.drop(columns=['Target'])
    df_full_train = df_full_train.drop(columns=['Target'])
    
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}, Full Train: {len(df_full_train)}")
    
    return df_train, df_val, df_test, df_full_train, y_train, y_val, y_test, y_full_train


def train_final_model(df_full_train, y_full_train, categorical_columns, numerical_columns):
    """Train the final Random Forest model on full training data"""
    print("Training final Random Forest model...")
    
    # Vectorize categorical and numerical features
    dv = DictVectorizer(sparse=False)
    dicts_full_train = df_full_train[categorical_columns + numerical_columns].to_dict(orient='records')
    X_full_train = dv.fit_transform(dicts_full_train)
    
    # Train Random Forest with tuned hyperparameters
    rf = RandomForestClassifier(
        class_weight='balanced',
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=5,
        random_state=1,
        n_jobs=-1
    )
    
    rf.fit(X_full_train, y_full_train)
    
    print("✓ Model training completed")
    
    return dv, rf


def save_model(dv, model, output_path='./models/random_forest_employee_attrition_v1.bin'):
    """Save the trained model and vectorizer to a pickle file"""
    print(f"Saving model to {output_path}...")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save both the vectorizer and model
    with open(output_path, 'wb') as f_out:
        pickle.dump((dv, model), f_out)
    
    print(f"✓ Model saved successfully")


def evaluate_model(dv, model, df_test, y_test, categorical_columns, numerical_columns):
    """Evaluate model on test set"""
    print("Evaluating model on test set...")
    
    dicts_test = df_test[categorical_columns + numerical_columns].to_dict(orient='records')
    X_test = dv.transform(dicts_test)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✓ Test ROC-AUC Score: {test_auc:.4f}")
    
    return test_auc


def main():
    """Main training pipeline"""
    print("="*80)
    print("EMPLOYEE ATTRITION PREDICTION - MODEL TRAINING")
    print("="*80)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    employee_survey_data, general_data, manager_survey_data, in_time, out_time = load_raw_data()
    
    # Engineer features
    time_features = engineer_time_features(in_time, out_time)
    
    # Merge all data
    df_final = merge_data(general_data, employee_survey_data, manager_survey_data, time_features)
    
    # Prepare data
    df_model, categorical_columns, numerical_columns = prepare_data(df_final)
    
    # Create train/val/test splits
    df_train, df_val, df_test, df_full_train, y_train, y_val, y_test, y_full_train = \
        create_train_val_test_split(df_model)
    
    # Train final model
    dv, rf = train_final_model(df_full_train, y_full_train, categorical_columns, numerical_columns)
    
    # Evaluate on test set
    test_auc = evaluate_model(dv, rf, df_test, y_test, categorical_columns, numerical_columns)
    
    # Save model
    save_model(dv, rf)
    
    print()
    print("="*80)
    print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"Final Model Performance (Test Set ROC-AUC): {test_auc:.4f}")
    print(f"Model saved to: ./models/random_forest_employee_attrition_v1.bin")
    print("="*80)


if __name__ == "__main__":
    main()
