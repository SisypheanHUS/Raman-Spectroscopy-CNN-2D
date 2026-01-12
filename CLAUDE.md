# CLAUDE.md - Project Context

## Prompt Protocol
Before executing any task request:
1. First, rewrite the user's prompt to be more specific, detailed, and actionable
2. Show the improved prompt to the user
3. Ask for confirmation or adjustments
4. Only then proceed with execution

Skip this protocol for simple questions, clarifications, or when the user says "just do it" or "skip refine".

## Project Overview
Raman spectroscopy analysis project for detecting fake alcohol. Uses deep learning (DenseNet/ResNet) with GADF (Gramian Angular Difference Field) transformations to classify spectral data into 11 ethanol concentration classes.

## Quick Start
```python
from src.preprocessing import airPLS, normalize_spectrum
from src.models import build_1d_densenet
from src.transforms import create_gadf_map
from src.utils import load_excel_data, set_seed

# Load and process data
wavenumbers, spectra = load_excel_data('data/Ethanol_Methanol.xlsx')
corrected = airPLS(spectra['sample'])
normalized = normalize_spectrum(corrected)
```

## Project Structure
```
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies
├── src/                      # Source code modules
│   ├── preprocessing.py      # Baseline correction, normalization
│   ├── models.py             # DenseNet, ResNet architectures
│   ├── transforms.py         # GADF transformation
│   ├── visualization.py      # Plotting functions
│   └── utils.py              # Data loading, training utilities
├── notebooks/
│   └── main_pipeline.ipynb   # Complete training pipeline
├── data/
│   ├── Ethanol/              # Raw ethanol spectra
│   ├── Methanol/             # Raw methanol spectra
│   ├── Thu_6/                # Experimental data (new, unused)
│   ├── synthetic/            # Generated 1D spectra
│   └── maps/                 # GADF 2D maps
├── experiments/              # Training experiment results
└── models/                   # Saved trained models
```

## Key Modules

### src/preprocessing.py
- `airPLS()` - Baseline correction
- `normalize_spectrum()` - Min-max normalization
- `generate_synthetic_spectrum()` - Data augmentation

### src/models.py
- `build_1d_densenet()` - DenseNet for 1D spectra
- `build_2d_densenet()` - DenseNet for GADF images
- `build_1d_resnet()` - ResNet for 1D spectra
- `build_2d_resnet()` - ResNet for GADF images

### src/transforms.py
- `create_gadf_map()` - Convert 1D spectrum to 2D GADF

### src/visualization.py
- `plot_confusion_matrix()` - Confusion matrix plots
- `occlusion_analysis()` - Feature importance analysis

## Data Flow
1. Load raw spectra (.txt or .xlsx)
2. Apply baseline correction (airPLS)
3. Normalize to [0, 1]
4. Generate GADF 2D maps (64x64)
5. Train CNN classifier
6. Evaluate with confusion matrices

## Notes
- Spectra interpolated to 880 points
- 11 classes: 0%, 10%, ..., 100% ethanol
- Unused data in `data/Thu_6/` and `Tan HUS/` - keep for future work

## Large File Reading
When a file exceeds 25k tokens (~1200-1500 lines), use offset/limit:
```python
# Read lines 1-500
Read(file_path, offset=1, limit=500)

# Read lines 501-1000
Read(file_path, offset=501, limit=500)
```
For notebooks, extract each cell separately rather than reading all at once.
