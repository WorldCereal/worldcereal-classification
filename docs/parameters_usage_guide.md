# WorldCereal Parameters Usage Guide

This guide shows how to use the WorldCereal parameters system to customize croptype mapping with `target_date` while keeping other parameters at their defaults.

## Simple Solution

Instead of complex convenience methods, you can just initialize `CropTypeParameters` with a custom `target_date`:

```python
# Default behavior (middle timestep)
croptype_params = CropTypeParameters()

# Custom target date
croptype_params = CropTypeParameters(target_date="2024-07-15")
```

That's it! No fuss, no complexity.

## Usage Examples

### Basic Usage (Default Parameters)
```python
from worldcereal.parameters import CropTypeParameters

# Use all defaults (target_date=None means middle timestep)
croptype_params = CropTypeParameters()
feature_params = croptype_params.feature_parameters.model_dump()
```

### Custom target_date
```python
# Initialize with custom target_date
croptype_params = CropTypeParameters(target_date="2024-07-15")
feature_params = croptype_params.feature_parameters.model_dump()

# Add any testing overrides
feature_params.update({
    "ignore_dependencies": True,
    "compile_presto": False,
})
```

### Complete Workflow Example
```python
from worldcereal.parameters import CropTypeParameters
from worldcereal.utils.models import load_model_lut

# Initialize with custom target_date
croptype_params = CropTypeParameters(target_date="2024-07-15")

# Feature extraction parameters
feature_params = croptype_params.feature_parameters.model_dump()
feature_params.update({"ignore_dependencies": True})  # For local testing

# Classification parameters
lookup_table = load_model_lut(croptype_params.classifier_parameters.classifier_url)
classifier_params = croptype_params.classifier_parameters.model_dump()
classifier_params.update({
    "ignore_dependencies": True,
    "lookup_table": lookup_table,
})
```
    lookup_table=lookup_table,
)
```

## Target Date Usage

For croptype mapping:
- `CropTypeParameters()` - Uses middle timestep (target_date=None)
- `CropTypeParameters(target_date="2024-07-15")` - Uses specific date
- The date must be within the temporal range of your input data
- Automatically has `temporal_prediction=True` in the defaults

For cropland mapping:
- `CropLandParameters()` - No target_date needed
- Per-timestep logits are generated internally; with `temporal_prediction=False` the workflow selects the middle timestep downstream.
