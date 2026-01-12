"""
Spectral transformation module.

Contains:
- GADF (Gramian Angular Difference Field) transformation
- Converts 1D spectra to 2D images for CNN classification
"""

import numpy as np


def create_gadf_map(spectrum, image_size=64):
    """
    Convert 1D spectrum to 2D GADF (Gramian Angular Difference Field) image.

    GADF encodes temporal correlations in time series data as images,
    enabling the use of 2D CNNs for spectral classification.

    Args:
        spectrum: Normalized 1D spectrum array (values should be in [0, 1])
        image_size: Output image size (default: 64x64)

    Returns:
        2D GADF image of shape (image_size, image_size)
    """
    try:
        from pyts.transformation import GADF

        # Ensure spectrum is 2D for pyts
        if spectrum.ndim == 1:
            spectrum = spectrum.reshape(1, -1)

        # Create GADF transformer
        gadf = GADF(image_size=image_size, method='difference')

        # Transform spectrum to GADF image
        gadf_image = gadf.fit_transform(spectrum)

        return gadf_image[0]  # Return single image

    except ImportError:
        # Fallback implementation without pyts
        return _gadf_manual(spectrum.flatten(), image_size)


def _gadf_manual(spectrum, image_size=64):
    """
    Manual GADF implementation (fallback if pyts not available).

    Args:
        spectrum: 1D spectrum array
        image_size: Output image size

    Returns:
        2D GADF image
    """
    # Resize spectrum to image_size
    from scipy.interpolate import interp1d

    x_old = np.linspace(0, 1, len(spectrum))
    x_new = np.linspace(0, 1, image_size)
    f = interp1d(x_old, spectrum, kind='linear')
    resized = f(x_new)

    # Normalize to [-1, 1]
    resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
    resized = 2 * resized - 1
    resized = np.clip(resized, -1, 1)

    # Convert to angular representation
    phi = np.arccos(resized)

    # Create GADF matrix
    gadf = np.zeros((image_size, image_size))
    for i in range(image_size):
        for j in range(image_size):
            gadf[i, j] = np.sin(phi[i] - phi[j])

    return gadf


def batch_create_gadf(spectra, image_size=64, verbose=True):
    """
    Convert batch of 1D spectra to GADF images.

    Args:
        spectra: Array of shape (n_samples, spectrum_length)
        image_size: Output image size
        verbose: Print progress

    Returns:
        Array of shape (n_samples, image_size, image_size)
    """
    n_samples = len(spectra)
    gadf_images = np.zeros((n_samples, image_size, image_size))

    for i, spectrum in enumerate(spectra):
        if verbose and (i + 1) % 100 == 0:
            print(f"Processing spectrum {i + 1}/{n_samples}")
        gadf_images[i] = create_gadf_map(spectrum, image_size)

    return gadf_images


def spectra_to_1d_input(spectra):
    """
    Prepare spectra for 1D CNN input.

    Args:
        spectra: Array of shape (n_samples, spectrum_length)

    Returns:
        Array of shape (n_samples, spectrum_length, 1)
    """
    if spectra.ndim == 2:
        return spectra[..., np.newaxis]
    return spectra


def spectra_to_2d_input(gadf_images):
    """
    Prepare GADF images for 2D CNN input.

    Args:
        gadf_images: Array of shape (n_samples, height, width)

    Returns:
        Array of shape (n_samples, height, width, 1)
    """
    if gadf_images.ndim == 3:
        return gadf_images[..., np.newaxis]
    return gadf_images
