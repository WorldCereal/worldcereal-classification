# Temporal vs Non-Temporal Predictions in WorldCereal

This document explains how to configure and use temporal vs non-temporal predictions in the WorldCereal classification pipeline.

## Overview

WorldCereal now supports two modes of feature extraction and prediction:

1. **Non-Temporal Prediction** (Default): Features are pooled across the time dimension, resulting in a single feature vector per pixel
2. **Temporal Prediction**: Features preserve the time dimension, and a specific timestep is selected based on a target date

## Configuration

### Parameters

The temporal prediction behavior is controlled by two new parameters in the `FeaturesParameters` class:

- `temporal_prediction` (bool, default=False): Enable/disable temporal prediction mode
- `target_date` (str, optional): Target date in ISO format (YYYY-MM-DD) for timestep selection

### Parameter Validation

- `target_date` can only be specified when `temporal_prediction=True`
- `target_date` must be in valid ISO format (YYYY-MM-DD) if provided
- If `temporal_prediction=True` and `target_date=None`, the middle timestep is selected
- **Runtime validation**: `target_date` must be within the temporal extent of the input features (checked during feature extraction)

## Error Handling

### Parameter Validation Errors

The following validation errors can occur during parameter configuration:

```python
# Error: target_date specified when temporal_prediction=False
FeaturesParameters(
    temporal_prediction=False,
    target_date="2024-07-01"  # ❌ Invalid
)
# Raises: ValidationError: target_date can only be specified when temporal_prediction=True

# Error: Invalid date format
FeaturesParameters(
    temporal_prediction=True,
    target_date="invalid-date"  # ❌ Invalid
)
# Raises: ValidationError: target_date must be in ISO format (YYYY-MM-DD)
```

### Runtime Validation Errors

During feature extraction, the following runtime errors can occur:

```python
# Error: target_date outside temporal extent
# If input features span 2024-01-01 to 2024-12-31, but target_date is outside this range:
FeaturesParameters(
    temporal_prediction=True,
    target_date="2025-06-01"  # ❌ Outside temporal extent
)
# During feature extraction:
# Raises: ValueError: Target date 2025-06-01 is outside the temporal extent of features.
#         Available time range: 2024-01-01 to 2024-12-31
```

## Usage Examples

### 1. Non-Temporal Prediction (Default)

```python
from worldcereal.parameters import FeaturesParameters, ClassifierParameters, CropLandParameters

# Default behavior - temporal_prediction=False
feature_params = FeaturesParameters(
    rescale_s1=False,
    presto_model_url="https://...",
    compile_presto=False,
    temporal_prediction=False,  # Default
    target_date=None
)

classifier_params = ClassifierParameters(
    classifier_url="https://..."
)

cropland_params = CropLandParameters(
    feature_parameters=feature_params,
    classifier_parameters=classifier_params
)
```

**Result**: Features are pooled across time → Single feature vector per pixel → Standard classification

### 2. Temporal Prediction with Specific Target Date

```python
# Enable temporal prediction with specific date
feature_params = FeaturesParameters(
    rescale_s1=False,
    presto_model_url="https://...",
    compile_presto=False,
    temporal_prediction=True,
    target_date="2024-07-01"  # Target date for timestep selection
)

croptype_params = CropTypeParameters(
    feature_parameters=feature_params,
    classifier_parameters=classifier_params
)
```

**Result**: Features preserve time dimension → Select timestep closest to 2024-07-01 → Classification

### 3. Temporal Prediction with Middle Timestep

```python
# Enable temporal prediction without specific date
feature_params = FeaturesParameters(
    rescale_s1=False,
    presto_model_url="https://...",
    compile_presto=False,
    temporal_prediction=True,
    target_date=None  # Use middle timestep
)

croptype_params = CropTypeParameters(
    feature_parameters=feature_params,
    classifier_parameters=classifier_params
)
```

**Result**: Features preserve time dimension → Select middle timestep (6/12) → Classification

## Implementation Details

### Feature Extractor Changes

The `feature_extractor.py` has been updated with the following changes:

1. **New Parameters**: Added support for `temporal_prediction` and `target_date` parameters
2. **Pooling Method Selection**: Automatically selects the appropriate pooling method:
   - `PoolingMethods.GLOBAL` for non-temporal prediction
   - `PoolingMethods.TIME` for temporal prediction
3. **Timestep Selection**: New function `select_timestep_from_temporal_features()` handles timestep selection

### Key Functions

```python
def select_timestep_from_temporal_features(features: xr.DataArray, target_date: str = None) -> xr.DataArray:
    """Select a specific timestep from temporal features based on target date.
    
    Parameters
    ----------
    features : xr.DataArray
        Temporal features with time dimension preserved.
    target_date : str, optional
        Target date in ISO format (YYYY-MM-DD). If None, selects middle timestep.
    
    Returns
    -------
    xr.DataArray
        Features for the selected timestep with time dimension removed.
    """
```

### Processing Flow

#### Non-Temporal Mode (Default)
```
Input Array (12 timesteps) 
→ Presto Feature Extraction (PoolingMethods.GLOBAL)
→ Features pooled across time
→ Single feature vector per pixel
→ Classification
```

#### Temporal Mode
```
Input Array (12 timesteps)
→ Presto Feature Extraction (PoolingMethods.TIME)
→ Features with time dimension preserved
→ Select timestep based on target_date
→ Features for selected timestep
→ Classification
```
