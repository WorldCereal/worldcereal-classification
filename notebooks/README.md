## Introduction
The notebook `system_v1_demo.ipynb` aims to show how to run WorldCereal system V1 for a custom area of interest. The current notebook focuses on training a custom crop type model for an area of interested and running inference on CDSE.

## Running on binder
In order to run this demo notebook smoothly, the WorldCereal consortium has set up a pre-configured Python environment for you. To access this environment and run the notebook, please follow the Binder link below.

However, BEFORE being able to make use of this service, you will need to register for the EGI notebooks service,
through [this link](https://aai.egi.eu/registry/co_petitions/start/coef:111).

Once this registration process has been completed, clicking the Binder icon below will bring the user to a Jupyterhub instance allowing to run the demo.

[![Binder](https://replay.notebooks.egi.eu/badge_logo.svg)](https://replay.notebooks.egi.eu/v2/gh/WorldCereal/worldcereal-binder/v1.0.0-beta?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252FWorldCereal%252Fworldcereal-classification%26urlpath%3Dlab%252Ftree%252Fworldcereal-classification%252Fnotebooks%252Fsystem_v1_demo.ipynb%26branch%3Dsystem-v1-beta)

## Running in your own environment
Alternatively you can build your own environment and run the notebook locally. For this you can build your environment based on `environment.yml` and then `pip install` the `worldcereal-classification` package using the `pyproject.toml` file.

Once configured, you can run the notebook.
