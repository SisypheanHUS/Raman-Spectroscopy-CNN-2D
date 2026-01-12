"""
Visualization module for Raman spectroscopy analysis.

Contains functions for:
- Confusion matrix plotting
- Spectrum comparison plots
- Occlusion analysis visualization
- Training history plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix',
                         filename=None, class_names=None, figsize=(10, 8)):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        filename: Optional path to save figure
        class_names: Optional list of class names
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [f'{i*10}%' for i in range(n_classes)]

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.show()


def plot_normalized_confusion_matrix(y_true, y_pred, title='Normalized Confusion Matrix',
                                     filename=None, class_names=None, figsize=(10, 8)):
    """
    Plot normalized confusion matrix (percentage per class).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        filename: Optional path to save figure
        class_names: Optional list of class names
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [f'{i*10}%' for i in range(n_classes)]

    plt.figure(figsize=figsize)
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=100)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.show()


def plot_spectra_comparison(wavenumbers, raw_spectrum, corrected_spectrum,
                           ethanol_spectrum=None, methanol_spectrum=None,
                           title='Spectrum Comparison', filename=None):
    """
    Plot comparison of raw and corrected spectra.

    Args:
        wavenumbers: Wavenumber axis
        raw_spectrum: Original spectrum
        corrected_spectrum: Baseline-corrected spectrum
        ethanol_spectrum: Optional ethanol reference
        methanol_spectrum: Optional methanol reference
        title: Plot title
        filename: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Raw vs Corrected
    axes[0].plot(wavenumbers, raw_spectrum, 'b-', alpha=0.7, label='Raw')
    axes[0].plot(wavenumbers, corrected_spectrum, 'r-', label='Corrected')
    axes[0].set_xlabel('Wavenumber (cm⁻¹)')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Baseline Correction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reference comparison
    if ethanol_spectrum is not None or methanol_spectrum is not None:
        axes[1].plot(wavenumbers, corrected_spectrum, 'k-', label='Sample', linewidth=2)
        if ethanol_spectrum is not None:
            axes[1].plot(wavenumbers, ethanol_spectrum, 'g--', alpha=0.7, label='Ethanol')
        if methanol_spectrum is not None:
            axes[1].plot(wavenumbers, methanol_spectrum, 'r--', alpha=0.7, label='Methanol')
        axes[1].set_xlabel('Wavenumber (cm⁻¹)')
        axes[1].set_ylabel('Intensity (normalized)')
        axes[1].set_title('Reference Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.show()


def plot_training_history(history, title='Training History', filename=None):
    """
    Plot training and validation accuracy/loss curves.

    Args:
        history: Keras training history object
        title: Plot title
        filename: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train')
    if 'val_accuracy' in history.history:
        axes[0].plot(history.history['val_accuracy'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history:
        axes[1].plot(history.history['val_loss'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.show()


def occlusion_analysis(model, X, y, window_size=44, wavenumbers=None):
    """
    Perform occlusion sensitivity analysis to identify important spectral regions.

    Args:
        model: Trained Keras model
        X: Input spectra (n_samples, length, 1)
        y: True labels
        window_size: Size of occlusion window
        wavenumbers: Optional wavenumber axis for plotting

    Returns:
        Dictionary with accuracy drops per position
    """
    from sklearn.metrics import accuracy_score

    # Baseline accuracy
    y_pred_baseline = model.predict(X, verbose=0).argmax(axis=1)
    baseline_acc = accuracy_score(y, y_pred_baseline)

    spectrum_length = X.shape[1]
    accuracy_drops = []
    positions = []

    # Slide window and measure accuracy drop
    for start in range(0, spectrum_length - window_size, window_size // 2):
        X_occluded = X.copy()
        X_occluded[:, start:start+window_size, :] = 0

        y_pred = model.predict(X_occluded, verbose=0).argmax(axis=1)
        acc = accuracy_score(y, y_pred)
        drop = baseline_acc - acc

        accuracy_drops.append(drop * 100)
        positions.append(start + window_size // 2)

    return {
        'positions': positions,
        'accuracy_drops': accuracy_drops,
        'baseline_accuracy': baseline_acc
    }


def plot_occlusion_analysis(occlusion_results, wavenumbers=None, title='Occlusion Analysis',
                           filename=None):
    """
    Plot occlusion analysis results showing important spectral regions.

    Args:
        occlusion_results: Dictionary from occlusion_analysis()
        wavenumbers: Optional wavenumber axis
        title: Plot title
        filename: Optional path to save figure
    """
    positions = occlusion_results['positions']
    drops = occlusion_results['accuracy_drops']

    if wavenumbers is not None and len(wavenumbers) > max(positions):
        x_axis = [wavenumbers[p] for p in positions]
        xlabel = 'Wavenumber (cm⁻¹)'
    else:
        x_axis = positions
        xlabel = 'Position (index)'

    plt.figure(figsize=(14, 5))
    plt.bar(x_axis, drops, width=20 if wavenumbers is not None else 10, alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy Drop (%)')
    plt.title(f"{title}\n(Baseline accuracy: {occlusion_results['baseline_accuracy']*100:.1f}%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")

    plt.show()
