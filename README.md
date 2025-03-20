# Network Traffic Analysis and Intrusion Detection System

A comprehensive system for analyzing network traffic and detecting various types of cyber attacks using machine learning techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Dataset Information](#dataset-information)
- [Attack Types](#attack-types)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Contributing](#contributing)

## Overview

This project implements a machine learning-based intrusion detection system that analyzes network traffic patterns to identify various types of cyber attacks. The system uses multiple machine learning models and ensemble techniques to achieve high accuracy in attack detection.

## Architecture

### System Overview

[![](https://mermaid.ink/img/pako:eNqtV11r40YU_SuDlg0t2G0ifyQruoFYWi8LyTax01Ba92EkXcWD5RkxM6rjzQZKaaEs_c5S6FIohdJ97ENf2pf-mP0D7U_oSDOSpThbCJGfZOnMmat7z7l3dG4FLATLsU45Tqbo2JtQpH5376IDTChy2TxhFKgUaEHkFD0KGBUaIlJfr_GwxGgfL4GjDycW-vfny1_0PZfFMQSSMIo20CFnAQhB6OnE-kgzZL-9Lb3my2_RY5ALxmfomOMoIgFSONRu76I929D-hg5xMAOJXJzIlAOqM9ka3dHo51-hIeAc9uBMcqzjqK_o6BVdveL7v_OwhdrgIVDgeLVitUYlJn-3oxTHRC6RO4VgJupRvJWRHm2dK1YDe8fnb-_mUMV3UY9Ao-0MfaLAYbHtRXVbIZcxqGShiMSxcwe2fIDtlpCczcC5swN2F-NWwGLGHT9WSWpFjMr2AsjpVDo-i8M1Krs5qk5zVN3mqI4aTNbRbbMFNCzKWfpm5YiKe169-PGfP7_RGnuNZQbGMi__MjaLAdOMpTDMwBjm-cuVBegpoQDcwCpk2jUD45rLZ5rzMeNzJcYneN01A-2aQe6aVz99moWrLEto-xiERONECf4a1xSRjMH0hPUYhjqGr39fB18JYWiCXrfIoKi674cQ-WWltu5t90P7RlUf2M1RdZqj6jZHNbwl1XWqPlDjJK6Ng19_MDdzlWQK3EAPPsZxmourVli3mAZ_oBGmIZujIeOZqjJpbyDXCPvFJ-j9hwPGzINMC27R9j9TWhcw9-Mr48HVsnW7pcgOQAldR_bG_ujNazSrH6668jrbiTbB58jlTIh2rYFXNz8xm68L1i0EG-xAP7hXFqGzswOd4Eb1dO3mqDrNUXWbozq5JdV1gn03lUkq6weYZ2gEIo3VqWdDXSWMy6tN2DNKvfwCHQKPsnZJA8jnvNIVJ4EopekZyV1-h4aE4tgw1hXi2eWxZU9K9QY51Z6CLwWpclWWrHr2CRFp2a7zlR4WU59hHtaW1tQ9Ahy3JZmD0jklkmXDocLeKw1VAvXrleDXUOtaeYWuowg2fbusVdTbDjY3b1R2z26OqtMcVbc5qt4tqSq6VoV9RCXwdpwL2mWU6hFqDqnqiHX__i56qoo7wgs97SfWUzU49fPB6rk5f0C4QqlulbVhDXWrUAiJ3ibHeRnOsytBDQFCX70B2mcsMbGoFObH34yhfJ4HY68DdDN-L1H9FfQmbqfCr_27j32IDXsQYyE8iFCeiWMiVabzNFP1RVPkOL_-vwTrG4I8AWern5xVdlylFo1VGUv3qKvZOK-rmqpYdRFU1LPf75t92wsSyqljJ2fFjVAZFnOOlw7qoZ7VsuagegoJ1afZecY7seQU5jCxHHUZYj6bWBN6oXA4lWy8pIHlSJ5Cy-IsPZ1aToRjof6leb48glWzmxeQBNMPGJsb0MV_Smh9oA)](https://mermaid.live/edit#pako:eNqtV11r40YU_SuDlg0t2G0ifyQruoFYWi8LyTax01Ba92EkXcWD5RkxM6rjzQZKaaEs_c5S6FIohdJ97ENf2pf-mP0D7U_oSDOSpThbCJGfZOnMmat7z7l3dG4FLATLsU45Tqbo2JtQpH5376IDTChy2TxhFKgUaEHkFD0KGBUaIlJfr_GwxGgfL4GjDycW-vfny1_0PZfFMQSSMIo20CFnAQhB6OnE-kgzZL-9Lb3my2_RY5ALxmfomOMoIgFSONRu76I929D-hg5xMAOJXJzIlAOqM9ka3dHo51-hIeAc9uBMcqzjqK_o6BVdveL7v_OwhdrgIVDgeLVitUYlJn-3oxTHRC6RO4VgJupRvJWRHm2dK1YDe8fnb-_mUMV3UY9Ao-0MfaLAYbHtRXVbIZcxqGShiMSxcwe2fIDtlpCczcC5swN2F-NWwGLGHT9WSWpFjMr2AsjpVDo-i8M1Krs5qk5zVN3mqI4aTNbRbbMFNCzKWfpm5YiKe169-PGfP7_RGnuNZQbGMi__MjaLAdOMpTDMwBjm-cuVBegpoQDcwCpk2jUD45rLZ5rzMeNzJcYneN01A-2aQe6aVz99moWrLEto-xiERONECf4a1xSRjMH0hPUYhjqGr39fB18JYWiCXrfIoKi674cQ-WWltu5t90P7RlUf2M1RdZqj6jZHNbwl1XWqPlDjJK6Ng19_MDdzlWQK3EAPPsZxmourVli3mAZ_oBGmIZujIeOZqjJpbyDXCPvFJ-j9hwPGzINMC27R9j9TWhcw9-Mr48HVsnW7pcgOQAldR_bG_ujNazSrH6668jrbiTbB58jlTIh2rYFXNz8xm68L1i0EG-xAP7hXFqGzswOd4Eb1dO3mqDrNUXWbozq5JdV1gn03lUkq6weYZ2gEIo3VqWdDXSWMy6tN2DNKvfwCHQKPsnZJA8jnvNIVJ4EopekZyV1-h4aE4tgw1hXi2eWxZU9K9QY51Z6CLwWpclWWrHr2CRFp2a7zlR4WU59hHtaW1tQ9Ahy3JZmD0jklkmXDocLeKw1VAvXrleDXUOtaeYWuowg2fbusVdTbDjY3b1R2z26OqtMcVbc5qt4tqSq6VoV9RCXwdpwL2mWU6hFqDqnqiHX__i56qoo7wgs97SfWUzU49fPB6rk5f0C4QqlulbVhDXWrUAiJ3ibHeRnOsytBDQFCX70B2mcsMbGoFObH34yhfJ4HY68DdDN-L1H9FfQmbqfCr_27j32IDXsQYyE8iFCeiWMiVabzNFP1RVPkOL_-vwTrG4I8AWern5xVdlylFo1VGUv3qKvZOK-rmqpYdRFU1LPf75t92wsSyqljJ2fFjVAZFnOOlw7qoZ7VsuagegoJ1afZecY7seQU5jCxHHUZYj6bWBN6oXA4lWy8pIHlSJ5Cy-IsPZ1aToRjof6leb48glWzmxeQBNMPGJsb0MV_Smh9oA)

The architecture diagram above illustrates our system's comprehensive design with four main layers:

1. **Data Collection & Processing Layer**:

   - Network traffic capture and packet analysis
   - Feature extraction with quality checks
   - Dataset generation with validation
   - Real-time data processing capabilities

2. **Data Processing Layer**:

   - Advanced data cleaning and preprocessing
   - Feature engineering and selection
   - Data normalization and splitting
   - Automated validation checks

3. **Model Training & Evaluation Layer**:

   - Multiple base models (Random Forest, XGBoost)
   - Ensemble learning with cross-validation
   - Meta-model optimization (Logistic Regression)
   - Continuous model updates

4. **Results & Reporting Layer**:
   - Performance metrics and analysis
   - Attack pattern visualization
   - Real-time monitoring dashboard
   - Automated report generation

**Key Features**:

- Comprehensive data validation at each stage
- Advanced feature engineering pipeline
- Multi-model ensemble approach
- Real-time monitoring capabilities
- Automated feedback loops
- Scalable architecture design

### System Flow Chart

[![](https://mermaid.ink/img/pako:eNqtVNtO20AQ_RVrERKVNipJgCRWhRTfAiEhAaqqat2Hxd4kK9Zea3dNuP579-LYBomHquQpM-fMnDmzkzyDhKUYuGBF2TbZIC6d2XWcO-qzv-_MEcmdSCE2Mz74HQPnRmpWDP58cTqdU8fTuQBJ5MwYSkm-1pDle4bga8KS44KzBAvxhuEbRvCs2xaUSNsoBq8WDgz8otDh4b5Kvzihbvadq7ne9GmIvYo4MUQsZM2rXdXljbPQ1J_pmmuUpyxzIsZVcaNgGeea8XPiMdbGzmp1ZTMliSQsF2aKqeaHucDZLcVNwfnHBZYwNYQLXT3HaiNz9Uj02y3_ejpja6JMJc41XnO9Tpa_sxfeI1oi3bJl8MI0nJlxGrweaGJhG8xMMDfLwKKkUrxTWJSyKKUN54Z8WU3KSSKarhZbaOwHESWi5AntrL6hLK1WwbhsYZcGuzqwO0zNyVlkYREbLHdBPd-NfKTqfW2cUCREgFeO0Her-6wIpe7ekT-Ojg-hkJzdYXevPxyGfR8mjDLubjdE4grqbEkqN26_eIArlsvOFpP1Rrq3jKY2IcgTdrtHxcM7vereK7led3QS9Wu57mhwEvQ-luv9s5wwvx8rNvJ7A6_xNvC60fhTxTJ9j5VYFI2Gh41YdDzwVfiJYsycW6UWjrrhSbNHv9cdHnv_qdbSc8bwqr6Udt6D_u5F2-nA7r2dCmEEJ_AMnsMpvIAzu6s2YQ4v4QIuK18AggzzDJFU_Qs_a14M5AZnOAau-poifheDOH9VPFRKdvOYJ8CVvMQQcFauN8BdISpUVBYpkjggaM1RVmcLlP9ibBe__gWMdthq?type=png)](https://mermaid.live/edit#pako:eNqtVNtO20AQ_RVrERKVNipJgCRWhRTfAiEhAaqqat2Hxd4kK9Zea3dNuP579-LYBomHquQpM-fMnDmzkzyDhKUYuGBF2TbZIC6d2XWcO-qzv-_MEcmdSCE2Mz74HQPnRmpWDP58cTqdU8fTuQBJ5MwYSkm-1pDle4bga8KS44KzBAvxhuEbRvCs2xaUSNsoBq8WDgz8otDh4b5Kvzihbvadq7ne9GmIvYo4MUQsZM2rXdXljbPQ1J_pmmuUpyxzIsZVcaNgGeea8XPiMdbGzmp1ZTMliSQsF2aKqeaHucDZLcVNwfnHBZYwNYQLXT3HaiNz9Uj02y3_ejpja6JMJc41XnO9Tpa_sxfeI1oi3bJl8MI0nJlxGrweaGJhG8xMMDfLwKKkUrxTWJSyKKUN54Z8WU3KSSKarhZbaOwHESWi5AntrL6hLK1WwbhsYZcGuzqwO0zNyVlkYREbLHdBPd-NfKTqfW2cUCREgFeO0Her-6wIpe7ekT-Ojg-hkJzdYXevPxyGfR8mjDLubjdE4grqbEkqN26_eIArlsvOFpP1Rrq3jKY2IcgTdrtHxcM7vereK7led3QS9Wu57mhwEvQ-luv9s5wwvx8rNvJ7A6_xNvC60fhTxTJ9j5VYFI2Gh41YdDzwVfiJYsycW6UWjrrhSbNHv9cdHnv_qdbSc8bwqr6Udt6D_u5F2-nA7r2dCmEEJ_AMnsMpvIAzu6s2YQ4v4QIuK18AggzzDJFU_Qs_a14M5AZnOAau-poifheDOH9VPFRKdvOYJ8CVvMQQcFauN8BdISpUVBYpkjggaM1RVmcLlP9ibBe__gWMdthq)

The flow chart above illustrates the system's workflow:

1. **Data Pipeline**:

   - Data Loading: Import network traffic data
   - Preprocessing: Clean and normalize data
   - Split Data: Divide into train (80%) and test (20%) sets

2. **Model Pipeline**:

   - Train base models:
     - Random Forest: For complex pattern recognition
     - XGBoost: For gradient boosted decision trees
   - Create ensemble: Combine predictions from base models
   - Meta Model: Logistic Regression for final classification
   - Final evaluation

3. **Output Generation**:
   - Performance metrics
   - Visualizations
   - Analysis reports

Color coding:

- 🟢 Green: Start/End points
- 🔵 Blue: Data processing
- 🟣 Purple: Data splitting
- 🟡 Orange: Model operations
- 🔴 Red: Output generation

## Features

- Multi-model ensemble approach
- Comprehensive attack type detection
- Feature importance analysis
- Detailed performance metrics
- Visualization of results
- Attack pattern analysis
- Automated data preprocessing
- Cross-validation support
- Model persistence

## Dataset Information

The project uses the CICIDS2017 dataset, which contains various types of network traffic data including:

### Dataset Download Instructions

1. Download the CICIDS2017 dataset from the official source:

   - Visit: https://www.unb.ca/cic/datasets/ids-2017.html
   - Download the dataset files
   - Extract the files to the `dataset/` directory in this project

2. Required files:
   - Monday-WorkingHours.pcap_ISCX.csv
   - Tuesday-WorkingHours.pcap_ISCX.csv
   - Wednesday-workingHours.pcap_ISCX.csv
   - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
   - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
   - Friday-WorkingHours-Morning.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
   - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv

Note: The dataset files are not included in this repository due to their large size. Please download them separately from the official source.

### Features

| Category                     | Feature Name                | Description                                  |
| ---------------------------- | --------------------------- | -------------------------------------------- |
| **Flow Metrics**             | Flow Duration               | Total duration of the flow                   |
|                              | Total Fwd Packets           | Number of packets in forward direction       |
|                              | Total Backward Packets      | Number of packets in backward direction      |
|                              | Flow Bytes/s                | Rate of bytes per second                     |
|                              | Flow Packets/s              | Rate of packets per second                   |
| **Packet Length**            | Total Length of Fwd Packets | Total size of forward packets                |
|                              | Total Length of Bwd Packets | Total size of backward packets               |
|                              | Fwd Packet Length Max       | Maximum forward packet length                |
|                              | Fwd Packet Length Min       | Minimum forward packet length                |
|                              | Fwd Packet Length Mean      | Average forward packet length                |
|                              | Fwd Packet Length Std       | Standard deviation of forward packet length  |
|                              | Bwd Packet Length Max       | Maximum backward packet length               |
|                              | Bwd Packet Length Min       | Minimum backward packet length               |
|                              | Bwd Packet Length Mean      | Average backward packet length               |
|                              | Bwd Packet Length Std       | Standard deviation of backward packet length |
| **IAT (Inter Arrival Time)** | Flow IAT Mean               | Mean inter-arrival time of flow              |
|                              | Flow IAT Std                | Standard deviation of flow IAT               |
|                              | Flow IAT Max                | Maximum flow IAT                             |
|                              | Flow IAT Min                | Minimum flow IAT                             |
|                              | Fwd IAT Total               | Total IAT in forward direction               |
|                              | Fwd IAT Mean                | Mean IAT in forward direction                |
|                              | Fwd IAT Std                 | Standard deviation of forward IAT            |
|                              | Fwd IAT Max                 | Maximum forward IAT                          |
|                              | Fwd IAT Min                 | Minimum forward IAT                          |
|                              | Bwd IAT Total               | Total IAT in backward direction              |
|                              | Bwd IAT Mean                | Mean IAT in backward direction               |
|                              | Bwd IAT Std                 | Standard deviation of backward IAT           |
|                              | Bwd IAT Max                 | Maximum backward IAT                         |
|                              | Bwd IAT Min                 | Minimum backward IAT                         |
| **Flags**                    | Fwd PSH Flags               | Number of PSH flags in forward direction     |
|                              | Bwd PSH Flags               | Number of PSH flags in backward direction    |
|                              | Fwd URG Flags               | Number of URG flags in forward direction     |
|                              | Bwd URG Flags               | Number of URG flags in backward direction    |
|                              | FIN Flag Count              | Number of FIN flags                          |
|                              | SYN Flag Count              | Number of SYN flags                          |
|                              | RST Flag Count              | Number of RST flags                          |
|                              | PSH Flag Count              | Number of PSH flags                          |
|                              | ACK Flag Count              | Number of ACK flags                          |
|                              | URG Flag Count              | Number of URG flags                          |
|                              | CWE Flag Count              | Number of CWE flags                          |
|                              | ECE Flag Count              | Number of ECE flags                          |
| **Header Information**       | Fwd Header Length           | Length of forward packet header              |
|                              | Bwd Header Length           | Length of backward packet header             |
| **Packet Statistics**        | Fwd Packets/s               | Forward packets per second                   |
|                              | Bwd Packets/s               | Backward packets per second                  |
|                              | Min Packet Length           | Minimum length of packet                     |
|                              | Max Packet Length           | Maximum length of packet                     |
|                              | Packet Length Mean          | Mean packet length                           |
|                              | Packet Length Std           | Standard deviation of packet length          |
|                              | Packet Length Variance      | Variance of packet length                    |
| **Ratios & Averages**        | Down/Up Ratio               | Download to upload ratio                     |
|                              | Average Packet Size         | Average size of packet                       |
|                              | Avg Fwd Segment Size        | Average size of forward segment              |
|                              | Avg Bwd Segment Size        | Average size of backward segment             |
| **Bulk Transfer**            | Fwd Avg Bytes/Bulk          | Average bytes in forward bulk                |
|                              | Fwd Avg Packets/Bulk        | Average packets in forward bulk              |
|                              | Fwd Avg Bulk Rate           | Average forward bulk rate                    |
|                              | Bwd Avg Bytes/Bulk          | Average bytes in backward bulk               |
|                              | Bwd Avg Packets/Bulk        | Average packets in backward bulk             |
|                              | Bwd Avg Bulk Rate           | Average backward bulk rate                   |
| **Subflow Information**      | Subflow Fwd Packets         | Forward packets in subflow                   |
|                              | Subflow Fwd Bytes           | Forward bytes in subflow                     |
|                              | Subflow Bwd Packets         | Backward packets in subflow                  |
|                              | Subflow Bwd Bytes           | Backward bytes in subflow                    |
| **Window Information**       | Init_Win_bytes_forward      | Initial window bytes forward                 |
|                              | Init_Win_bytes_backward     | Initial window bytes backward                |
|                              | act_data_pkt_fwd            | Actual data packets forward                  |
|                              | min_seg_size_forward        | Minimum segment size forward                 |
| **Activity Time**            | Active Mean                 | Mean time flow was active                    |
|                              | Active Std                  | Standard deviation of active time            |
|                              | Active Max                  | Maximum time flow was active                 |
|                              | Active Min                  | Minimum time flow was active                 |
| **Idle Time**                | Idle Mean                   | Mean time flow was idle                      |
|                              | Idle Std                    | Standard deviation of idle time              |
|                              | Idle Max                    | Maximum time flow was idle                   |
|                              | Idle Min                    | Minimum time flow was idle                   |


## Attack Types

| Attack Category | Attack Types                                                     | Description                                                  |
| --------------- | ---------------------------------------------------------------- | ------------------------------------------------------------ |
| DoS             | DoS Hulk, DoS GoldenEye, DoS Slowhttptest, DoS Slowloris         | Denial of Service attacks that overwhelm target systems      |
| DDoS            | DDoS LOIT, DDoS                                                  | Distributed Denial of Service attacks using multiple sources |
| Port Scan       | PortScan                                                         | Scanning network ports to identify open services             |
| Web Attacks     | Web Attack Brute Force, Web Attack XSS, Web Attack Sql Injection | Various web-based attacks targeting web applications         |
| Botnet          | Botnet ARES                                                      | Botnet-related malicious activities                          |
| Infiltration    | Infiltration                                                     | Unauthorized access attempts                                 |
| Heartbleed      | Heartbleed                                                       | Exploitation of Heartbleed vulnerability                     |
| Normal          | BENIGN                                                           | Normal network traffic                                       |

## Project Structure

```
├── main.py                 # Main script to run the analysis
├── requirements.txt        # Project dependencies
├── README.md              # Project documentation
├── PROGRESS.md            # Progress tracking
├── src/
│   ├── data_preparation.py    # Data loading and preprocessing
│   ├── analysis.py           # General dataset analysis
│   ├── attack_analysis.py    # Attack-specific analysis
│   ├── model.py             # Machine learning models
│   └── visualization.py     # Visualization utilities
├── dataset/               # Dataset files
└── output/               # Generated output files
    ├── analysis/         # Analysis reports and visualizations
    └── models/          # Trained model files
```

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset files in the `dataset/` directory.

2. Run the analysis:

```bash
python main.py --random_state 123
```

Optional arguments:

- `--random_state`: Set random seed for reproducibility (default: 123)
- `--train_size`: Training set size (default: 0.8)
- `--test_size`: Test set size (default: 0.2)

## Output

The system generates various outputs in the `output/` directory:

### Analysis Reports

- `attack_distribution.csv`: Distribution of different attack types
- `feature_importance.csv`: Ranked list of important features
- `model_performance.csv`: Detailed performance metrics for each model

### Visualizations

- `confusion_matrix.png`: Confusion matrix for model evaluation
- `roc_curves.png`: ROC curves for each attack type
- `attack_patterns.png`: Visualization of attack patterns
- `feature_correlations.png`: Feature correlation heatmap

### Model Files

- `random_forest.pkl`: Trained Random Forest model
- `xgboost.pkl`: Trained XGBoost model
- `meta_model.pkl`: Trained Logistic Regression meta-model
- `scaler.pkl`: Fitted data scaler

### Performance Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Detection Rate
- False Positive Rate
