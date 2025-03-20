import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from typing import Tuple, List, Dict
import joblib
from tqdm import tqdm

class DataPreparation:
    def __init__(self, data_dir: str):
        """
        Initialize DataPreparation class
        Args:
            data_dir (str): Directory containing the dataset files
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_datasets(self) -> pd.DataFrame:
        """
        Load all CSV files from the data directory
        Returns:
            pd.DataFrame: Combined dataset
        """
        all_data = []
        total_size = sum(os.path.getsize(os.path.join(self.data_dir, f)) 
                        for f in os.listdir(self.data_dir) if f.endswith('.csv'))
        
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Loading datasets") as pbar:
            for file in os.listdir(self.data_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(self.data_dir, file)
                    file_size = os.path.getsize(file_path)
                    
                    print(f"\nProcessing {file}...")
                    df = pd.read_csv(file_path)
                    
                    # Print basic statistics for each file
                    print(f"Rows: {len(df):,}")
                    print(f"Attack types: {df[' Label'].value_counts().to_dict()}")
                    
                    all_data.append(df)
                    pbar.update(file_size)
        
        print("\nCombining all datasets...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Store feature columns for consistency
        self.feature_columns = [col for col in combined_df.columns if col != ' Label']
        
        return combined_df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset
        Args:
            df (pd.DataFrame): Input dataframe
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("\nPreprocessing data...")
        print(f"Initial shape: {df.shape}")
        
        # Handle missing values
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        print(f"Removed {missing_before} missing values")
        
        # Handle infinite values
        inf_mask = np.isinf(df.select_dtypes(include=np.number))
        inf_count = inf_mask.sum().sum()
        if inf_count > 0:
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()
            print(f"Removed {inf_count} infinite values")
        
        # Convert ' Label' column to binary
        label_counts_before = df[' Label'].value_counts()
        df[' Label'] = df[' Label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
        print("\nLabel distribution:")
        print(df[' Label'].value_counts())
        
        print(f"Final shape: {df.shape}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target variables
        Args:
            df (pd.DataFrame): Input dataframe
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features) and y (target)
        """
        print("\nPreparing features...")
        
        # Separate features and target
        X = df[self.feature_columns]
        y = df[' Label']
        
        # Convert all features to numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Drop any remaining non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        dropped_columns = set(X.columns) - set(numeric_columns)
        if dropped_columns:
            print(f"Dropping non-numeric columns: {dropped_columns}")
        X = X[numeric_columns]
        
        # Scale features
        print("Scaling features...")
        X = self.scaler.fit_transform(X)
        
        print(f"Final feature matrix shape: {X.shape}")
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data into train and test sets
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
            test_size (float): Proportion of test set
            random_state (int): Random seed
        Returns:
            Tuple: Train and test sets
        """
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Print class distribution
        print("\nClass distribution:")
        print("Training set:")
        print(pd.Series(y_train).value_counts(normalize=True))
        print("\nTest set:")
        print(pd.Series(y_test).value_counts(normalize=True))
        
        return X_train, X_test, y_train, y_test
    
    def save_scaler(self, output_dir: str):
        """
        Save the fitted scaler
        Args:
            output_dir (str): Directory to save the scaler
        """
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        
    def save_feature_columns(self, output_dir: str):
        """
        Save the feature columns list
        Args:
            output_dir (str): Directory to save the feature columns
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'feature_columns.txt'), 'w') as f:
            f.write('\n'.join(self.feature_columns)) 