#!/usr/bin/env python3
"""
Main script for XGBoost Imbalanced Dataset Classification
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

from src.data_preprocessing import DataPreprocessor
from src.model_training import XGBoostTrainer
from src.evaluation import ModelEvaluator

def create_sample_imbalanced_dataset():
    """Create a sample imbalanced dataset for demonstration"""
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Save dataset
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/imbalanced_dataset.csv', index=False)
    print("Sample imbalanced dataset created and saved!")
    
    return df

def main():
    """Main execution pipeline"""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Create or load dataset
    print("=== Step 1: Data Loading ===")
    df = create_sample_imbalanced_dataset()
    
    # Step 2: Data preprocessing and EDA
    print("\n=== Step 2: Data Preprocessing ===")
    preprocessor = DataPreprocessor()
    
    # Exploratory analysis
    class_counts = preprocessor.exploratory_analysis(df, 'target')
    
    # Preprocess features
    X, y = preprocessor.preprocess_features(df, 'target')
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Step 3: Model Training with different approaches
    print("\n=== Step 3: Model Training ===")
    trainer = XGBoostTrainer()
    evaluator = ModelEvaluator()
    
    # Approach 1: XGBoost with scale_pos_weight
    print("\n--- Training XGBoost with scale_pos_weight ---")
    model_weighted = trainer.train_model(X_train, y_train, use_scale_pos_weight=True)
    results_weighted = evaluator.evaluate_model(model_weighted, X_test, y_test, "XGBoost_Weighted")
    
    # Approach 2: XGBoost with SMOTE
    print("\n--- Training XGBoost with SMOTE ---")
    X_train_smote, y_train_smote = preprocessor.apply_smote(X_train, y_train)
    trainer_smote = XGBoostTrainer()
    model_smote = trainer_smote.train_model(X_train_smote, y_train_smote, use_scale_pos_weight=False)
    results_smote = evaluator.evaluate_model(model_smote, X_test, y_test, "XGBoost_SMOTE")
    
    # Approach 3: Hyperparameter tuning
    print("\n--- Hyperparameter Tuning ---")
    trainer_tuned = XGBoostTrainer()
    model_tuned = trainer_tuned.hyperparameter_tuning(X_train, y_train)
    results_tuned = evaluator.evaluate_model(model_tuned, X_test, y_test, "XGBoost_Tuned")
    
    # Step 4: Model Evaluation and Visualization
    print("\n=== Step 4: Model Evaluation ===")
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrix(y_test, results_weighted['y_pred'], "XGBoost_Weighted")
    evaluator.plot_confusion_matrix(y_test, results_smote['y_pred'], "XGBoost_SMOTE")
    
    # Plot ROC curves
    evaluator.plot_roc_curve(y_test, results_weighted['y_pred_proba'], "XGBoost_Weighted")
    evaluator.plot_roc_curve(y_test, results_smote['y_pred_proba'], "XGBoost_SMOTE")
    
    # Plot Precision-Recall curves
    evaluator.plot_precision_recall_curve(y_test, results_weighted['y_pred_proba'], "XGBoost_Weighted")
    evaluator.plot_precision_recall_curve(y_test, results_smote['y_pred_proba'], "XGBoost_SMOTE")
    
    # Compare all models
    all_results = {
        'XGBoost_Weighted': results_weighted,
        'XGBoost_SMOTE': results_smote,
        'XGBoost_Tuned': results_tuned
    }
    evaluator.compare_models(all_results)
    
    # Step 5: Save best model
    print("\n=== Step 5: Model Saving ===")
    best_model_name = max(all_results.keys(), key=lambda k: all_results[k]['f1_score'])
    print(f"Best model based on F1-score: {best_model_name}")
    
    # Save the best model
    if best_model_name == 'XGBoost_Weighted':
        trainer.save_model('results/best_xgboost_model.pkl')
    elif best_model_name == 'XGBoost_SMOTE':
        trainer_smote.save_model('results/best_xgboost_model.pkl')
    else:
        trainer_tuned.save_model('results/best_xgboost_model.pkl')
    
    print("\\nProject completed successfully!")
    print("Check the 'results' folder for visualizations and saved model.")

if __name__ == "__main__":
    main()