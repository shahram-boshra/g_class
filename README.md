# Molecular Graph Classification Model with PyTorch Geometric

This repository contains a comprehensive implementation for molecular graph classification using PyTorch Geometric (PyG). It provides a modular and configurable pipeline for processing molecular data, building graph neural network models, and conducting training and evaluation.

## Table of Contents

-   [Overview](#overview)
-   [Features](#features)
-   [Installation](#installation)
-   [Project Structure](#project-structure)
-   [Configuration](#configuration)
-   [Usage](#usage)
-   [Modules](#modules)
-   [Data Processing Details](#data-processing-details)
-   [Model Architecture Details](#model-architecture-details)
-   [Training and Evaluation Details](#training-and-evaluation-details)
-   [Output](#output)
-   [Contributing](#contributing)
-   [License](#license)
-   [Acknowledgments](#acknowledgments)

## Overview

This project focuses on the development and application of graph neural networks (GNNs) for molecular property prediction. It leverages RDKit for molecular data processing and PyG for GNN implementation, providing a robust and flexible framework for handling chemical data.

## Features

-   **Modular Design**: Code organized into distinct modules for data processing, model definition, and training.
-   **Configurable Pipeline**: Utilizes YAML configuration for easy adjustment of data paths, model hyperparameters, and training parameters.
-   **RDKit Integration**: Comprehensive molecular processing including hydrogenation, sanitization, kekulization, embedding, and optimization.
-   **Flexible GNN Architecture**: Supports various graph convolution layers (GCN, GAT, SAGE, GIN, GraphConv, TransformerConv, CustomMP) with customizable layer types and hyperparameters.
-   **Efficient Data Handling**: Caching mechanisms to speed up data processing and loading.
-   **Robust Training Setup**: Includes learning rate scheduling, early stopping, and L1 regularization.
-   **Comprehensive Evaluation**: Provides metrics such as MAE, MSE, R2, and Explained Variance, along with loss and metric plotting.
-   **Data Augmentation**: includes random rotation, scale, jitter, and flips.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [repository_url]
    cd [repository_directory]
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install torch torch-geometric scikit-learn pandas rdkit-pypi pyyaml pydantic diskcache matplotlib
    ```

## Project Structure

Molecular-Graph-Classification/
├── main.py             # Main execution script
├── rdkit_utils.py      # RDKit molecular processing utilities
├── data_utils.py       # Data loading and processing utilities
├── config_loader.py    # Configuration loading and validation
├── dataset.py          # Custom PyTorch Geometric dataset
├── models.py           # Graph neural network model definition
├── training_utils.py   # Training and evaluation utilities
├── config.yaml         # Configuration file
├── README.md           # This file
└── C:/Chem_Data/        # Data directory (configurable in config.yaml)
└── Mols/           # Directory containing .mol files
└── targets_g_class.csv # Target values in CSV format


## Configuration

The `config.yaml` file allows for extensive customization of the project.

```yaml
rdkit_processing:
  steps: ["hydrogenate", "sanitize", "kekulize", "embed", "optimize"] # RDKit processing steps

data:
  root_dir: "C:/Chem_Data" # Root directory of the dataset
  target_csv: "targets_g_class.csv" # CSV file with target values
  use_cache: true # Enable/disable caching
  train_split: 0.8 # Training set split ratio
  valid_split: 0.1 # Validation set split ratio

model:
  batch_size: 32 # Batch size for training
  learning_rate: 0.001 # Learning rate
  weight_decay: 0.0001 # Weight decay
  step_size: 50 # Step size for learning rate scheduler
  gamma: 0.5 # Gamma for learning rate scheduler
  reduce_lr_factor: 0.5 # Factor for reducing learning rate on plateau
  reduce_lr_patience: 10 # Patience for reducing learning rate
  early_stopping_patience: 20 # Patience for early stopping
  early_stopping_delta: 0.001 # Minimum change for early stopping
  l1_regularization_lambda: 0.001 # L1 regularization lambda
  first_layer_type: "custom_mp" # Type of first graph convolution layer
  hidden_channels: 256 # Number of hidden channels
  second_layer_type: "custom_mp" # Type of second graph convolution layer
  dropout_rate: 0.5 # Dropout rate
Usage
Prepare your data:

Place .mol files in the C:/Chem_Data/Mols directory (or the path specified in config.yaml).
Create a targets_g_class.csv file with molecule names as the index and target values as columns.
Run the main script:

Bash

python main.py
The script will:

Load and process molecular data.
Split the dataset.
Train the model.
Evaluate the model.
Save test predictions and generate plots.
Modules
main.py: Orchestrates the entire workflow, from data loading to model evaluation.
rdkit_utils.py: Provides RDKit-based molecular processing functions.
data_utils.py: Handles molecular graph data processing, including feature extraction and graph construction.
config_loader.py: Manages configuration loading and validation using YAML and Pydantic.
dataset.py: Implements a custom PyG dataset for molecular data.
models.py: Defines the GNN model architecture.
training_utils.py: Provides training and evaluation utilities, including early stopping and plotting.
Data Processing Details
RDKit Processing: Configurable steps including hydrogenation, sanitization, kekulization, embedding, and optimization.
Feature Extraction: One-hot encoding of atomic and bond features.
Graph Construction: Creation of PyG Data objects with node features, edge indices, and edge attributes.
Caching: Caching of processed graphs to reduce processing time.
Model Architecture Details
Graph Convolution Layers: Supports GCN, GAT, SAGE, GIN, GraphConv, TransformerConv, and a custom message passing layer.
Normalization: Batch normalization after each convolution layer.
Activation: ELU activation functions.
Dropout: Dropout regularization.
Global Pooling: Global mean pooling to obtain graph-level representations.
Linear Output Layer: Final linear layer for classification.
L1 Regularization: Optional L1 regularization to prevent overfitting.
Training and Evaluation Details
Loss Function: Negative Log-Likelihood Loss (NLLLoss).
Optimizer: Adam optimizer.
Learning Rate Scheduling: StepLR and ReduceLROnPlateau.
Early Stopping: Stops training when validation loss stops improving.
Evaluation Metrics: MAE, MSE, R2, and Explained Variance.
Plotting: Plots of training/validation loss and evaluation metrics.
Output
test_targets.npy: NumPy array of test set targets.
test_predictions.npy: NumPy array of test set predictions.
Plots of training/validation loss and evaluation metrics.
Contributing
Contributions are welcome! Please submit pull requests or open issues to improve this project.

License
This project is licensed under the MIT License.

Acknowledgments
PyTorch Geometric: For providing the GNN framework.
RDKit: For molecular data processing.
scikit-learn: For evaluation metrics.
Pydantic and PyYAML: For configuration management.




