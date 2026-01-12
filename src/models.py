"""
Neural network models for Raman spectroscopy classification.

Contains:
- DenseNet architectures (1D and 2D)
- ResNet architectures (1D and 2D)
- Baseline correction model
"""

import tensorflow as tf
from keras import layers, models, regularizers


# =============================================================================
# DenseNet Architectures
# =============================================================================

def _dense_block_1d(x, num_layers, growth_rate, name):
    """1D Dense block with batch normalization."""
    for i in range(num_layers):
        bn = layers.BatchNormalization(name=f'{name}_bn_{i}')(x)
        relu = layers.ReLU(name=f'{name}_relu_{i}')(bn)
        conv = layers.Conv1D(growth_rate, 3, padding='same',
                            kernel_regularizer=regularizers.l2(1e-4),
                            name=f'{name}_conv_{i}')(relu)
        x = layers.Concatenate(name=f'{name}_concat_{i}')([x, conv])
    return x


def _transition_layer_1d(x, reduction, name):
    """1D Transition layer for dimensionality reduction."""
    filters = int(x.shape[-1] * reduction)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.ReLU(name=f'{name}_relu')(x)
    x = layers.Conv1D(filters, 1, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name=f'{name}_conv')(x)
    x = layers.AveragePooling1D(2, strides=2, name=f'{name}_pool')(x)
    return x


def build_1d_densenet(input_shape=(880, 1), num_classes=11, growth_rate=12):
    """
    Build 1D DenseNet for spectral classification.

    Args:
        input_shape: Input spectrum shape (default: 880 points, 1 channel)
        num_classes: Number of output classes
        growth_rate: Growth rate for dense blocks

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # Initial convolution
    x = layers.Conv1D(64, 7, strides=2, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same', name='pool1')(x)

    # Dense blocks with transitions
    x = _dense_block_1d(x, num_layers=6, growth_rate=growth_rate, name='dense1')
    x = _transition_layer_1d(x, reduction=0.5, name='trans1')

    x = _dense_block_1d(x, num_layers=12, growth_rate=growth_rate, name='dense2')
    x = _transition_layer_1d(x, reduction=0.5, name='trans2')

    x = _dense_block_1d(x, num_layers=24, growth_rate=growth_rate, name='dense3')
    x = _transition_layer_1d(x, reduction=0.5, name='trans3')

    x = _dense_block_1d(x, num_layers=16, growth_rate=growth_rate, name='dense4')

    # Classification head
    x = layers.BatchNormalization(name='bn_final')(x)
    x = layers.ReLU(name='relu_final')(x)
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return models.Model(inputs, outputs, name='DenseNet1D')


def _dense_block_2d(x, num_layers, growth_rate, name):
    """2D Dense block with batch normalization."""
    for i in range(num_layers):
        bn = layers.BatchNormalization(name=f'{name}_bn_{i}')(x)
        relu = layers.ReLU(name=f'{name}_relu_{i}')(bn)
        conv = layers.Conv2D(growth_rate, 3, padding='same',
                            kernel_regularizer=regularizers.l2(1e-4),
                            name=f'{name}_conv_{i}')(relu)
        x = layers.Concatenate(name=f'{name}_concat_{i}')([x, conv])
    return x


def _transition_layer_2d(x, reduction, name):
    """2D Transition layer for dimensionality reduction."""
    filters = int(x.shape[-1] * reduction)
    x = layers.BatchNormalization(name=f'{name}_bn')(x)
    x = layers.ReLU(name=f'{name}_relu')(x)
    x = layers.Conv2D(filters, 1, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name=f'{name}_conv')(x)
    x = layers.AveragePooling2D(2, strides=2, name=f'{name}_pool')(x)
    return x


def build_2d_densenet(input_shape=(64, 64, 1), num_classes=11, growth_rate=12):
    """
    Build 2D DenseNet for GADF image classification.

    Args:
        input_shape: Input image shape (default: 64x64, 1 channel)
        num_classes: Number of output classes
        growth_rate: Growth rate for dense blocks

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)

    # Dense blocks with transitions
    x = _dense_block_2d(x, num_layers=6, growth_rate=growth_rate, name='dense1')
    x = _transition_layer_2d(x, reduction=0.5, name='trans1')

    x = _dense_block_2d(x, num_layers=12, growth_rate=growth_rate, name='dense2')
    x = _transition_layer_2d(x, reduction=0.5, name='trans2')

    x = _dense_block_2d(x, num_layers=24, growth_rate=growth_rate, name='dense3')
    x = _transition_layer_2d(x, reduction=0.5, name='trans3')

    x = _dense_block_2d(x, num_layers=16, growth_rate=growth_rate, name='dense4')

    # Classification head
    x = layers.BatchNormalization(name='bn_final')(x)
    x = layers.ReLU(name='relu_final')(x)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return models.Model(inputs, outputs, name='DenseNet2D')


# =============================================================================
# ResNet Architectures
# =============================================================================

def _residual_block_1d(x, filters, kernel_size=3, stride=1, name='res'):
    """1D Residual block with skip connection."""
    shortcut = x

    # First conv
    x = layers.Conv1D(filters, kernel_size, strides=stride, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)

    # Second conv
    x = layers.Conv1D(filters, kernel_size, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)

    # Shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, strides=stride, padding='same',
                                kernel_regularizer=regularizers.l2(1e-4),
                                name=f'{name}_shortcut')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name}_relu2')(x)
    return x


def build_1d_resnet(input_shape=(880, 1), num_classes=11):
    """
    Build 1D ResNet for spectral classification.

    Args:
        input_shape: Input spectrum shape
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # Initial convolution
    x = layers.Conv1D(64, 7, strides=2, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling1D(3, strides=2, padding='same', name='pool1')(x)

    # Residual blocks
    x = _residual_block_1d(x, 64, name='res1a')
    x = _residual_block_1d(x, 64, name='res1b')

    x = _residual_block_1d(x, 128, stride=2, name='res2a')
    x = _residual_block_1d(x, 128, name='res2b')

    x = _residual_block_1d(x, 256, stride=2, name='res3a')
    x = _residual_block_1d(x, 256, name='res3b')

    x = _residual_block_1d(x, 512, stride=2, name='res4a')
    x = _residual_block_1d(x, 512, name='res4b')

    # Classification head
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return models.Model(inputs, outputs, name='ResNet1D')


def _residual_block_2d(x, filters, kernel_size=3, stride=1, name='res'):
    """2D Residual block with skip connection."""
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name=f'{name}_conv1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.ReLU(name=f'{name}_relu1')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name=f'{name}_conv2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                kernel_regularizer=regularizers.l2(1e-4),
                                name=f'{name}_shortcut')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_shortcut_bn')(shortcut)

    x = layers.Add(name=f'{name}_add')([x, shortcut])
    x = layers.ReLU(name=f'{name}_relu2')(x)
    return x


def build_2d_resnet(input_shape=(64, 64, 1), num_classes=11):
    """
    Build 2D ResNet for GADF image classification.

    Args:
        input_shape: Input image shape
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    x = layers.Conv2D(64, 7, strides=2, padding='same',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name='conv1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)

    x = _residual_block_2d(x, 64, name='res1a')
    x = _residual_block_2d(x, 64, name='res1b')

    x = _residual_block_2d(x, 128, stride=2, name='res2a')
    x = _residual_block_2d(x, 128, name='res2b')

    x = _residual_block_2d(x, 256, stride=2, name='res3a')
    x = _residual_block_2d(x, 256, name='res3b')

    x = _residual_block_2d(x, 512, stride=2, name='res4a')
    x = _residual_block_2d(x, 512, name='res4b')

    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    return models.Model(inputs, outputs, name='ResNet2D')


# =============================================================================
# Baseline Correction Model
# =============================================================================

def create_baseline_model(input_shape=880):
    """
    Create neural network for baseline correction refinement.

    Args:
        input_shape: Spectrum length

    Returns:
        Keras model for baseline estimation
    """
    inputs = layers.Input(shape=(input_shape,), name='input')

    x = layers.Dense(512, activation='relu', name='dense1')(inputs)
    x = layers.Dropout(0.3, name='dropout1')(x)
    x = layers.Dense(256, activation='relu', name='dense2')(x)
    x = layers.Dropout(0.3, name='dropout2')(x)
    x = layers.Dense(128, activation='relu', name='dense3')(x)
    outputs = layers.Dense(input_shape, activation='linear', name='output')(x)

    return models.Model(inputs, outputs, name='BaselineModel')
