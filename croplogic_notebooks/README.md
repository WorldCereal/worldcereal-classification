# CropLogic inference notebook bundle

This small bundle runs the CropLogic openEO inference app notebook. It does not include the training walkthrough or the full training pipeline.

## Install

From this folder, in the environment you use for Jupyter:

```bash
python -m pip install -e ".[notebook]"
```

## Launch

```bash
python -m jupyter notebook notebooks/02_inference_app.ipynb
```

The notebook uses the public model registry and downloads model files when needed. Running inference requires internet access, an openEO account/configuration, and the dependencies listed in `pyproject.toml`.
