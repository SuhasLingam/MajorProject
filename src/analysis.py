import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os
from collections import defaultdict

class DatasetAnalyzer:
    def __init__(self, data_dir: str):
        """
        Initialize DatasetAnalyzer
        Args:
            data_dir (str): Directory containing dataset files
        """
        self.data_dir = data_dir
        self.attack_categories = {
            'DoS/DDoS': ['DDoS', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'],
            'Port Scanning': ['PortScan'],
            'Brute Force': ['FTP-Patator', 'SSH-Patator', 'Web Attack Brute Force'],
            'Web Attacks': ['Web Attack XSS', 'Web Attack Sql Injection'],
            'Infiltration': ['Infiltration', 'Bot'],
            'Heartbleed': ['Heartbleed'],
            'Benign': ['BENIGN']
        }
        
    def clean_attack_name(self, name: str) -> str:
        """Clean attack name by removing special characters"""
        return name.replace('', '').strip()
        
    def analyze_datasets(self) -> Dict:
        """
        Analyze all datasets and categorize attacks
        Returns:
            Dict: Analysis results
        """
        results = {
            'total_records': 0,
            'attack_distribution': defaultdict(int),
            'daily_distribution': {},
            'category_distribution': defaultdict(int),
            'feature_stats': None
        }
        
        all_data = []
        print("\nAnalyzing datasets...")
        
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                print(f"\nProcessing {file}...")
                df = pd.read_csv(os.path.join(self.data_dir, file))
                
                # Clean attack names
                df[' Label'] = df[' Label'].apply(self.clean_attack_name)
                
                day = file.split('-')[0]
                
                # Count attack types
                attack_counts = df[' Label'].value_counts()
                results['daily_distribution'][day] = dict(attack_counts)
                
                # Update total counts
                results['total_records'] += len(df)
                for attack, count in attack_counts.items():
                    results['attack_distribution'][attack] += count
                
                # Store data for feature analysis
                all_data.append(df)
        
        # Combine all data for feature analysis
        combined_df = pd.concat(all_data)
        results['feature_stats'] = self._analyze_features(combined_df)
        
        # Calculate category distribution
        for attack, count in results['attack_distribution'].items():
            category = self._get_attack_category(attack)
            results['category_distribution'][category] += count
        
        return results
    
    def _get_attack_category(self, attack_name: str) -> str:
        """Get category for an attack type"""
        for category, attacks in self.attack_categories.items():
            if attack_name in attacks:
                return category
        return "Unknown"
    
    def _analyze_features(self, df: pd.DataFrame) -> Dict:
        """Analyze features statistics"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return {
            'feature_stats': df[numeric_cols].describe(),
            'correlations': df[numeric_cols].corr()
        }
    
    def generate_visualizations(self, results: Dict, output_dir: str):
        """
        Generate visualizations from analysis results
        Args:
            results: Analysis results
            output_dir: Directory to save visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Attack Category Distribution
        plt.figure(figsize=(12, 6))
        categories = results['category_distribution']
        plt.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%')
        plt.title('Distribution of Attack Categories')
        plt.savefig(os.path.join(output_dir, 'attack_categories_pie.png'))
        plt.close()
        
        # 2. Daily Attack Distribution
        plt.figure(figsize=(15, 8))
        daily_data = pd.DataFrame(results['daily_distribution']).fillna(0)
        daily_data.plot(kind='bar', stacked=True)
        plt.title('Daily Distribution of Attacks')
        plt.xlabel('Day')
        plt.ylabel('Number of Records')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'daily_attacks.png'))
        plt.close()
        
        # 3. Attack Type Distribution
        plt.figure(figsize=(15, 8))
        attacks = results['attack_distribution']
        plt.bar(attacks.keys(), attacks.values())
        plt.title('Distribution of Attack Types')
        plt.xlabel('Attack Type')
        plt.ylabel('Number of Records')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attack_distribution.png'))
        plt.close()
        
        # 4. Feature Correlation Heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(results['feature_stats']['correlations'], 
                   cmap='coolwarm', center=0, annot=False)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlations.png'))
        plt.close()
        
        # Save analysis results as text
        self._save_analysis_report(results, output_dir)
    
    def _save_analysis_report(self, results: Dict, output_dir: str):
        """Save analysis results as a text report"""
        report_path = os.path.join(output_dir, 'analysis_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Network Traffic Analysis Report\n\n")
            
            f.write("## Dataset Overview\n")
            f.write(f"Total Records: {results['total_records']:,}\n\n")
            
            f.write("## Attack Categories\n")
            for category, count in results['category_distribution'].items():
                percentage = (count / results['total_records']) * 100
                f.write(f"- {category}: {count:,} ({percentage:.2f}%)\n")
            
            f.write("\n## Daily Distribution\n")
            for day, attacks in results['daily_distribution'].items():
                f.write(f"\n### {day}\n")
                for attack, count in attacks.items():
                    f.write(f"- {attack}: {count:,}\n")
            
            f.write("\n## Attack Types\n")
            for attack, count in results['attack_distribution'].items():
                percentage = (count / results['total_records']) * 100
                f.write(f"- {attack}: {count:,} ({percentage:.2f}%)\n") 