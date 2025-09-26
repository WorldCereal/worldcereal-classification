# Temporal vs Non-Temporal Predictions in WorldCereal

This document explains how to configure and use temporal vs non-temporal predictions in the WorldCereal classification pipeline.

## Overview

WorldCereal supports two modes for handling the temporal dimension during inference. In both cases the finetuning head emits per-timestep logits; the difference lies in how a single supervision timestep is chosen for map generation:

1. **Non-Temporal Prediction** (Default): Per-timestep logits are produced internally, but downstream evaluation collapses them to a single timestep (defaulting to the middle of the sequence when no explicit target is provided).
2. **Temporal Prediction**: Per-timestep logits are preserved explicitly and a labelled timestep (e.g. provided via a target date) is used for supervision and reporting.

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

**Result**: Per-timestep logits are produced, and the middle timestep is selected downstream → Standard classification

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

**Result**: The logits for each timestep remain accessible → Select timestep closest to 2024-07-01 → Classification

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

**Result**: The logits for each timestep remain accessible → Select middle timestep (6/12) → Classification

## Implementation Details

### Feature Extractor Changes

The `feature_extractor.py` has been updated with the following changes:

1. **New Parameters**: Added support for `temporal_prediction` and `target_date` parameters
2. **Temporal Embeddings**: Presto is always queried with `PoolingMethods.TIME` so that the downstream head can emit logits for every timestep.
3. **Timestep Selection**: The helper `select_timestep_from_temporal_features()` identifies the timestep used when a single prediction is required (e.g. for exporting rasters).

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
→ Presto Feature Extraction (PoolingMethods.TIME)
→ Per-timestep logits from finetuning head
→ Select middle timestep (if no target date)
→ Classification
```

#### Temporal Mode
```
Input Array (12 timesteps)
→ Presto Feature Extraction (PoolingMethods.TIME)
→ Per-timestep logits from finetuning head
→ Select timestep based on target_date or kernel centre
→ Classification + temporal diagnostics
```
