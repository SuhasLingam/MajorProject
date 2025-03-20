import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
from sklearn.preprocessing import StandardScaler

class AttackAnalyzer:
    def __init__(self, data_dir: str):
        """
        Initialize AttackAnalyzer
        Args:
            data_dir (str): Directory containing dataset files
        """
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        
    def clean_attack_name(self, name: str) -> str:
        """Clean attack name by removing special characters"""
        return name.replace('', '').strip()
        
    def analyze_specific_attack(self, attack_type: str, output_dir: str):
        """
        Analyze specific attack type
        Args:
            attack_type: Type of attack to analyze
            output_dir: Directory to save results
        """
        print(f"\nAnalyzing {attack_type} attack...")
        
        # Load and combine all datasets
        attack_data = []
        benign_data = []
        
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(self.data_dir, file))
                
                # Clean data - replace infinite values with NaN
                df = df.replace([np.inf, -np.inf], np.nan)
                
                # Clean attack names
                df[' Label'] = df[' Label'].apply(self.clean_attack_name)
                
                # Extract attack and benign samples
                attack_samples = df[df[' Label'] == attack_type]
                if len(attack_samples) > 0:
                    benign_samples = df[df[' Label'] == 'BENIGN'].sample(
                        n=min(len(attack_samples), len(df[df[' Label'] == 'BENIGN'])),
                        replace=True
                    )
                    
                    # Clean samples
                    attack_samples = self._clean_data(attack_samples)
                    benign_samples = self._clean_data(benign_samples)
                    
                    attack_data.append(attack_samples)
                    benign_data.append(benign_samples)
        
        if not attack_data:
            print(f"No samples found for attack type: {attack_type}")
            return
        
        # Combine data
        attack_df = pd.concat(attack_data)
        benign_df = pd.concat(benign_data)
        
        # Create output directory
        attack_dir = os.path.join(output_dir, attack_type.replace(' ', '_'))
        os.makedirs(attack_dir, exist_ok=True)
        
        try:
            # Generate visualizations
            self._analyze_traffic_patterns(attack_df, benign_df, attack_dir)
            self._analyze_feature_importance(attack_df, benign_df, attack_dir)
            self._generate_statistics_report(attack_df, benign_df, attack_dir)
        except Exception as e:
            print(f"Error analyzing {attack_type}: {str(e)}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by handling missing and infinite values"""
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # For numeric columns, fill NaN with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _analyze_traffic_patterns(self, attack_df: pd.DataFrame, benign_df: pd.DataFrame, output_dir: str):
        """Analyze and visualize traffic patterns"""
        try:
            # 1. Packet size distribution
            plt.figure(figsize=(12, 6))
            plt.hist(attack_df[' Packet Length Mean'].clip(-1e6, 1e6), bins=50, alpha=0.5, label='Attack')
            plt.hist(benign_df[' Packet Length Mean'].clip(-1e6, 1e6), bins=50, alpha=0.5, label='Benign')
            plt.title('Packet Length Distribution')
            plt.xlabel('Packet Length')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'packet_length_distribution.png'))
            plt.close()
            
            # 2. Flow duration analysis
            plt.figure(figsize=(12, 6))
            duration_data = [
                attack_df[' Flow Duration'].clip(0, attack_df[' Flow Duration'].quantile(0.99)),
                benign_df[' Flow Duration'].clip(0, benign_df[' Flow Duration'].quantile(0.99))
            ]
            plt.boxplot(duration_data, labels=['Attack', 'Benign'])
            plt.title('Flow Duration Comparison')
            plt.ylabel('Duration')
            plt.yscale('log')
            plt.savefig(os.path.join(output_dir, 'flow_duration_comparison.png'))
            plt.close()
            
            # 3. Packet rate analysis
            plt.figure(figsize=(12, 6))
            plt.scatter(
                attack_df[' Flow Packets/s'].clip(-1e6, 1e6),
                attack_df['Flow Bytes/s'].clip(-1e6, 1e6),
                alpha=0.5, label='Attack'
            )
            plt.scatter(
                benign_df[' Flow Packets/s'].clip(-1e6, 1e6),
                benign_df['Flow Bytes/s'].clip(-1e6, 1e6),
                alpha=0.5, label='Benign'
            )
            plt.title('Packet Rate vs Byte Rate')
            plt.xlabel('Packets/s')
            plt.ylabel('Bytes/s')
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'packet_rate_analysis.png'))
            plt.close()
        except Exception as e:
            print(f"Error in traffic pattern analysis: {str(e)}")
    
    def _analyze_feature_importance(self, attack_df: pd.DataFrame, benign_df: pd.DataFrame, output_dir: str):
        """Analyze feature importance"""
        try:
            # Prepare data
            features = attack_df.select_dtypes(include=[np.number]).columns
            attack_data = attack_df[features]
            benign_data = benign_df[features]
            
            # Calculate feature differences
            feature_diff = {}
            for feature in features:
                if feature != ' Label':  # Skip the label column
                    attack_mean = attack_data[feature].mean()
                    benign_mean = benign_data[feature].mean()
                    if not (np.isnan(attack_mean) or np.isnan(benign_mean)):
                        feature_diff[feature] = abs(float(attack_mean - benign_mean))
            
            if feature_diff:
                # Plot top discriminating features
                plt.figure(figsize=(15, 8))
                top_features = dict(sorted(feature_diff.items(), key=lambda x: x[1], reverse=True)[:10])
                plt.bar(top_features.keys(), top_features.values())
                plt.title('Top Discriminating Features')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
                plt.close()
            else:
                print("No valid features found for importance analysis")
        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")
    
    def _generate_statistics_report(self, attack_df: pd.DataFrame, benign_df: pd.DataFrame, output_dir: str):
        """Generate statistical report"""
        try:
            report_path = os.path.join(output_dir, 'statistics_report.md')
            
            with open(report_path, 'w') as f:
                f.write(f"# Attack Analysis Report\n\n")
                
                f.write("## Basic Statistics\n")
                f.write(f"- Number of attack samples: {len(attack_df)}\n")
                f.write(f"- Number of benign samples: {len(benign_df)}\n\n")
                
                f.write("## Key Features Statistics\n")
                key_features = [
                    ' Flow Duration', ' Flow Packets/s', 'Flow Bytes/s',
                    ' Packet Length Mean', ' Packet Length Std'
                ]
                
                for feature in key_features:
                    if feature in attack_df.columns:  # Check if feature exists
                        f.write(f"\n### {feature.strip()}\n")
                        
                        # Attack traffic statistics
                        attack_stats = attack_df[feature].describe()
                        f.write("Attack Traffic:\n")
                        f.write(f"- Mean: {attack_stats['mean']:.2f}\n")
                        f.write(f"- Std: {attack_stats['std']:.2f}\n")
                        f.write(f"- Min: {attack_stats['min']:.2f}\n")
                        f.write(f"- Max: {attack_stats['max']:.2f}\n")
                        f.write(f"- 25th Percentile: {attack_stats['25%']:.2f}\n")
                        f.write(f"- Median: {attack_stats['50%']:.2f}\n")
                        f.write(f"- 75th Percentile: {attack_stats['75%']:.2f}\n\n")
                        
                        # Benign traffic statistics
                        benign_stats = benign_df[feature].describe()
                        f.write("Benign Traffic:\n")
                        f.write(f"- Mean: {benign_stats['mean']:.2f}\n")
                        f.write(f"- Std: {benign_stats['std']:.2f}\n")
                        f.write(f"- Min: {benign_stats['min']:.2f}\n")
                        f.write(f"- Max: {benign_stats['max']:.2f}\n")
                        f.write(f"- 25th Percentile: {benign_stats['25%']:.2f}\n")
                        f.write(f"- Median: {benign_stats['50%']:.2f}\n")
                        f.write(f"- 75th Percentile: {benign_stats['75%']:.2f}\n")
        except Exception as e:
            print(f"Error in generating statistics report: {str(e)}")

    def analyze_attack(self, df: pd.DataFrame, attack_type: str):
        """
        Analyze a specific attack type
        Args:
            df (pd.DataFrame): Input dataframe
            attack_type (str): Type of attack to analyze
        """
        print(f"\nAnalyzing {attack_type} attack...")
        
        # Extract attack samples
        attack_samples = df[df[' Label'] == attack_type]
        
        # Skip if no samples found (except for specific web attacks)
        if len(attack_samples) == 0:
            if attack_type not in ['Web Attack Brute Force', 'Web Attack XSS', 'Web Attack Sql Injection']:
                print(f"No samples found for attack type: {attack_type}")
            return
        
        # Extract benign samples for comparison
        benign_samples = df[df[' Label'] == 'BENIGN']
        
        # Analyze each feature
        for feature in df.columns:
            if feature != ' Label':
                # Calculate statistics for attack samples
                attack_stats = attack_samples[feature].describe()
                
                # Calculate statistics for benign samples
                benign_stats = benign_samples[feature].describe()
                
                # Calculate feature importance
                importance = abs(attack_stats['mean'] - benign_stats['mean'])
                
                # Store results
                self.feature_importance[feature][attack_type] = importance
                self.attack_stats[attack_type][feature] = attack_stats
                self.benign_stats[attack_type][feature] = benign_stats 