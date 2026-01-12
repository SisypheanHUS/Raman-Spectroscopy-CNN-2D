"""
Utility functions for Raman spectroscopy analysis.

Contains:
- Random seed setting
- Data loading functions
- Training utilities
- Evaluation metrics
"""

import os
import random
import numpy as np
import pandas as pd


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass

    os.environ['PYTHONHASHSEED'] = str(seed)


# =============================================================================
# Data Loading
# =============================================================================

def load_excel_data(filepath='data/Ethanol_Methanol.xlsx'):
    """
    Load Raman spectra from Excel file.

    Expected format: Each column is a spectrum, first column is wavenumber.

    Args:
        filepath: Path to Excel file

    Returns:
        Tuple of (wavenumbers, spectra_dict)
    """
    df = pd.read_excel(filepath, engine='openpyxl')

    # First column is typically wavenumber
    wavenumbers = df.iloc[:, 0].values

    spectra = {}
    for col in df.columns[1:]:
        spectra[col] = df[col].values

    return wavenumbers, spectra


def load_noise_data(filepath='data/dataset_noise_pure_182.npy',
                   labels_path='data/labels_noise_pure_182.npy'):
    """
    Load noise/baseline dataset.

    Args:
        filepath: Path to noise data .npy file
        labels_path: Path to labels .npy file

    Returns:
        Tuple of (data, labels) or just data if labels not found
    """
    data = np.load(filepath)

    if os.path.exists(labels_path):
        labels = np.load(labels_path)
        return data, labels

    return data


def load_processed_data(data_dir='data'):
    """
    Load preprocessed data (1D spectra and GADF maps).

    Args:
        data_dir: Base data directory

    Returns:
        Dictionary with 'spectra_1d', 'gadf_maps', 'labels'
    """
    result = {}

    # 1D spectra
    spectra_path = os.path.join(data_dir, 'synthetic', 'synthetic_1d.npy')
    if os.path.exists(spectra_path):
        result['spectra_1d'] = np.load(spectra_path)

    # GADF maps
    gadf_path = os.path.join(data_dir, 'maps', 'spectral_maps_gadf.npy')
    if os.path.exists(gadf_path):
        result['gadf_maps'] = np.load(gadf_path)

    # Labels
    labels_path = os.path.join(data_dir, 'labels', 'labels.csv')
    if os.path.exists(labels_path):
        result['labels'] = pd.read_csv(labels_path)

    return result


def load_txt_spectrum(filepath):
    """
    Load spectrum from .txt file.

    Args:
        filepath: Path to spectrum file

    Returns:
        Tuple of (wavenumbers, intensities)
    """
    df = pd.read_csv(filepath, sep=None, engine='python', header=None,
                    names=['Wavenumber', 'Intensity'])
    return df['Wavenumber'].values, df['Intensity'].values


def load_spectra_from_folder(folder_path):
    """
    Load all .txt spectra from a folder.

    Args:
        folder_path: Path to folder containing .txt files

    Returns:
        Dictionary mapping filename to (wavenumbers, intensities)
    """
    spectra = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            wavenumbers, intensities = load_txt_spectrum(filepath)
            spectra[filename] = (wavenumbers, intensities)
    return spectra


# =============================================================================
# Training Utilities
# =============================================================================

def train_model(model, X_train, y_train, X_val=None, y_val=None,
               epochs=100, batch_size=32, patience=10,
               checkpoint_path=None, class_weights=None):
    """
    Train a Keras model with standard callbacks.

    Args:
        model: Compiled Keras model
        X_train: Training data
        y_train: Training labels
        X_val: Validation data (optional)
        y_val: Validation labels (optional)
        epochs: Maximum training epochs
        batch_size: Batch size
        patience: Early stopping patience
        checkpoint_path: Path to save best model
        class_weights: Optional class weights dictionary

    Returns:
        Training history
    """
    import tensorflow as tf
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    callbacks = [
        EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                     patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss',
                         factor=0.5, patience=5, min_lr=1e-6)
    ]

    if checkpoint_path:
        callbacks.append(
            ModelCheckpoint(checkpoint_path, monitor='val_loss' if X_val is not None else 'loss',
                          save_best_only=True)
        )

    validation_data = (X_val, y_val) if X_val is not None else None

    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    return history


def compute_class_weights(y):
    """
    Compute class weights for imbalanced datasets.

    Args:
        y: Array of labels

    Returns:
        Dictionary mapping class index to weight
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Comprehensive model evaluation.

    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels
        class_names: Optional list of class names

    Returns:
        Dictionary with evaluation metrics
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                f1_score, classification_report)

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'predictions': y_pred
    }

    print(f"\n{'='*50}")
    print(f"Model Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']*100:.2f}%")
    print(f"Recall:    {results['recall']*100:.2f}%")
    print(f"F1 Score:  {results['f1_score']*100:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return results


def save_experiment(experiment_dir, model, history, config, results=None):
    """
    Save experiment results (model, history, config).

    Args:
        experiment_dir: Directory to save experiment
        model: Trained Keras model
        history: Training history
        config: Configuration dictionary
        results: Optional evaluation results
    """
    import json
    from datetime import datetime

    os.makedirs(experiment_dir, exist_ok=True)

    # Save model
    model.save(os.path.join(experiment_dir, 'model.keras'))

    # Save config
    config['timestamp'] = datetime.now().isoformat()
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Save history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(experiment_dir, 'history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)

    # Save results
    if results:
        results_save = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                       for k, v in results.items()}
        with open(os.path.join(experiment_dir, 'results.json'), 'w') as f:
            json.dump(results_save, f, indent=2)

    print(f"Experiment saved to: {experiment_dir}")
