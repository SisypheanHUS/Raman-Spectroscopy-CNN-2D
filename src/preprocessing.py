"""
Preprocessing module for Raman spectroscopy data.

Contains functions for:
- Baseline correction (airPLS, WhittakerSmooth)
- Spectrum normalization
- Spectrum interpolation
- Data augmentation (shift, stretch, synthetic generation)
"""

import numpy as np
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d


# =============================================================================
# Baseline Correction
# =============================================================================

def WhittakerSmooth(x, w, lambda_=1, differences=1):
    """
    Whittaker smoother for baseline estimation.

    Args:
        x: Input spectrum
        w: Weights array
        lambda_: Smoothing parameter
        differences: Order of differences

    Returns:
        Smoothed background estimate
    """
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)


def airPLS(x, lambda_=100, porder=1, itermax=15):
    """
    Adaptive Iteratively Reweighted Penalized Least Squares (airPLS) baseline correction.

    Args:
        x: Input spectrum
        lambda_: Smoothing parameter (auto-adjusted based on spectrum statistics)
        porder: Order of differences
        itermax: Maximum iterations

    Returns:
        Estimated baseline
    """
    m = x.shape[0]
    w = np.ones(m)
    lambda_ = max(50, min(500, 50 * np.std(x) / (np.mean(np.abs(x)) + 1e-6)))

    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())

        if dssn < 0.001 * np.abs(x).sum():
            return z

        if i == itermax:
            print(f'WARNING: Max iteration reached! lambda_={lambda_:.2f}, dssn={dssn:.2e}')
            return WhittakerSmooth(x, np.ones(m), lambda_=50)

        w[d >= 0] = 0
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]

    return z


def enhanced_baseline_correction(spectrum, baseline_model=None):
    """
    Enhanced baseline correction combining airPLS with optional neural network refinement.

    Args:
        spectrum: Input spectrum
        baseline_model: Optional trained baseline correction model

    Returns:
        Baseline-corrected spectrum
    """
    # Apply airPLS
    baseline = airPLS(spectrum)
    corrected = spectrum - baseline

    # Apply neural network refinement if model provided
    if baseline_model is not None:
        import tensorflow as tf
        spectrum_input = tf.expand_dims(corrected, axis=0)
        refined_baseline = baseline_model(spectrum_input, training=False)
        corrected = corrected - refined_baseline.numpy().flatten()

    return corrected


# =============================================================================
# Normalization
# =============================================================================

def normalize_spectrum(spectrum):
    """
    Min-max normalization of spectrum to [0, 1] range.

    Args:
        spectrum: Input spectrum array

    Returns:
        Normalized spectrum
    """
    spectrum = spectrum - np.min(spectrum)
    if np.max(spectrum) > 0:
        spectrum = spectrum / np.max(spectrum)
    return spectrum


# =============================================================================
# Interpolation
# =============================================================================

def interpolate_spectrum(spectrum, original_wavenumbers, target_length=880):
    """
    Interpolate spectrum to a fixed number of points.

    Args:
        spectrum: Input spectrum intensity values
        original_wavenumbers: Original wavenumber axis
        target_length: Target number of points (default: 880)

    Returns:
        Interpolated spectrum
    """
    f = interp1d(original_wavenumbers, spectrum, kind='linear', fill_value='extrapolate')
    new_wavenumbers = np.linspace(original_wavenumbers.min(), original_wavenumbers.max(), target_length)
    return f(new_wavenumbers)


def to_880(arr, target_length=880):
    """
    Resize spectrum array to target length using linear interpolation.

    Args:
        arr: Input spectrum array
        target_length: Target length (default: 880)

    Returns:
        Resized spectrum
    """
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, arr, kind='linear')
    return f(x_new)


# =============================================================================
# Data Augmentation
# =============================================================================

def shift_spectrum(spectrum, shift):
    """
    Shift spectrum along the wavenumber axis.

    Args:
        spectrum: Input spectrum
        shift: Number of points to shift (positive = right, negative = left)

    Returns:
        Shifted spectrum
    """
    shifted = np.zeros_like(spectrum)
    if shift > 0:
        shifted[shift:] = spectrum[:-shift]
    elif shift < 0:
        shifted[:shift] = spectrum[-shift:]
    else:
        shifted = spectrum.copy()
    return shifted


def stretch_spectrum(spectrum, alpha):
    """
    Stretch/compress spectrum along the wavenumber axis.

    Args:
        spectrum: Input spectrum
        alpha: Stretch factor (>1 = stretch, <1 = compress)

    Returns:
        Stretched spectrum
    """
    n = len(spectrum)
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, int(n * alpha))
    f = interp1d(x_old, spectrum, kind='linear', fill_value='extrapolate')
    stretched = f(x_new)

    # Resize back to original length
    return to_880(stretched, n)


def generate_synthetic_spectrum(input_spectrum, noise_data=None, spectrum_length=880):
    """
    Generate synthetic spectrum with augmentation (noise, shift, stretch).

    Args:
        input_spectrum: Base spectrum to augment
        noise_data: Optional noise profiles to add
        spectrum_length: Target spectrum length

    Returns:
        Augmented synthetic spectrum
    """
    spectrum = input_spectrum.copy()

    # Random shift (-5 to +5 points)
    shift = np.random.randint(-5, 6)
    spectrum = shift_spectrum(spectrum, shift)

    # Random stretch (0.98 to 1.02)
    alpha = np.random.uniform(0.98, 1.02)
    spectrum = stretch_spectrum(spectrum, alpha)

    # Add noise
    if noise_data is not None:
        noise_idx = np.random.randint(len(noise_data))
        noise = noise_data[noise_idx]
        noise_scale = np.random.uniform(0.01, 0.05)
        spectrum = spectrum + noise_scale * noise[:len(spectrum)]
    else:
        # Add random Gaussian noise
        noise_scale = np.random.uniform(0.01, 0.03)
        spectrum = spectrum + noise_scale * np.random.randn(len(spectrum))

    return normalize_spectrum(spectrum)
