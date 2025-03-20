import os
import time
from src.data_preparation import DataPreparation
from src.models import BaseModels, MetaModel
from src.evaluation import ModelEvaluator
from src.analysis import DatasetAnalyzer
from src.attack_analysis import AttackAnalyzer
import argparse
from typing import Tuple
import numpy as np

def clean_attack_name(name: str) -> str:
    """Clean attack name by removing special characters"""
    return name.replace('', '').strip()

def train_base_models(X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, base_models: BaseModels) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train base models and get predictions
    """
    print("\nTraining base models...")
    start_time = time.time()
    
    # Train base models
    base_models.train(X_train, y_train)
    
    # Get predictions for training and validation sets
    print("Generating predictions from base models...")
    train_pred = base_models.predict(X_train)
    val_pred = base_models.predict(X_val)
    
    print(f"Base models training completed in {time.time() - start_time:.2f} seconds")
    return train_pred, val_pred

def train_meta_model(meta_model: MetaModel, base_train_pred: np.ndarray, 
                    base_val_pred: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """
    Train meta model and get predictions
    """
    print("\nTraining meta model...")
    start_time = time.time()
    
    # Train meta model
    meta_model.train(base_train_pred, y_train)
    
    # Get final predictions
    print("Generating final predictions...")
    final_pred = meta_model.predict(base_val_pred)
    
    print(f"Meta model training completed in {time.time() - start_time:.2f} seconds")
    return final_pred

def run_analysis(data_dir: str, output_dir: str, random_state: int = 42):
    """
    Run complete analysis pipeline
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Record start time
    total_start_time = time.time()
    
    # Step 1: General Dataset Analysis
    print("\n=== Step 1: General Dataset Analysis ===")
    analyzer = DatasetAnalyzer(data_dir)
    results = analyzer.analyze_datasets()
    analyzer.generate_visualizations(results, analysis_dir)
    
    # Step 2: Specific Attack Analysis
    print("\n=== Step 2: Specific Attack Analysis ===")
    attack_analyzer = AttackAnalyzer(data_dir)
    attacks = [
        'DDoS', 'DoS Hulk', 'DoS GoldenEye', 'PortScan',
        'Web Attack Brute Force', 'Web Attack XSS',
        'Web Attack Sql Injection', 'FTP-Patator',
        'SSH-Patator', 'Bot', 'Infiltration', 'Heartbleed'
    ]
    
    for attack in attacks:
        attack_analyzer.analyze_specific_attack(attack, os.path.join(analysis_dir, "specific_attacks"))
    
    # Step 3: Data Preparation
    print("\n=== Step 3: Data Preparation ===")
    data_prep = DataPreparation(data_dir)
    df = data_prep.load_datasets()
    
    # Clean attack names
    df[' Label'] = df[' Label'].apply(clean_attack_name)
    
    df = data_prep.preprocess_data(df)
    X, y = data_prep.prepare_features(df)
    X_train, X_test, y_train, y_test = data_prep.split_data(X, y, random_state=random_state)
    
    # Save scaler and feature columns
    data_prep.save_scaler(output_dir)
    data_prep.save_feature_columns(output_dir)
    
    # Step 4: Model Training
    print("\n=== Step 4: Model Training ===")
    base_models = BaseModels(random_state=random_state)
    meta_model = MetaModel(random_state=random_state)
    
    # Train base models and get predictions
    base_train_pred, base_test_pred = train_base_models(X_train, y_train, X_test, base_models)
    
    # Train meta model and get final predictions
    final_pred = train_meta_model(meta_model, base_train_pred, base_test_pred, y_train)
    
    # Save models
    base_models.save_models(output_dir)
    meta_model.save_model(output_dir)
    
    # Step 5: Model Evaluation
    print("\n=== Step 5: Model Evaluation ===")
    evaluator = ModelEvaluator()
    metrics = evaluator.compute_metrics(y_test, final_pred)
    
    # Print, plot and save metrics
    evaluator.print_metrics()
    evaluator.plot_metrics(output_dir)
    evaluator.save_metrics(output_dir)
    
    # Print total execution time
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

def main():
    parser = argparse.ArgumentParser(description="Network Traffic Analysis and Intrusion Detection")
    parser.add_argument("--data_dir", type=str, default="dataset",
                      help="Directory containing the dataset files")
    parser.add_argument("--output_dir", type=str, default="output",
                      help="Directory to save results")
    parser.add_argument("--random_state", type=int, default=42,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    run_analysis(args.data_dir, args.output_dir, args.random_state)

if __name__ == "__main__":
    main() 