# ESA WorldCereal classification module <!-- omit in toc -->
[![Tests](https://github.com/WorldCereal/worldcereal-classification/actions/workflows/ci.yaml/badge.svg)](https://github.com/WorldCereal/worldcereal-classification/actions/workflows/ci.yaml) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![DOI](https://img.shields.io/badge/DOI-10.5194/essd--15--5491--2023-blue)](https://doi.org/10.5194/essd-15-5491-2023) [![Documentation](https://img.shields.io/badge/docs-WorldCereal%20Documentation-blue)](https://worldcereal.github.io/worldcereal-documentation/) [![Discuss Forum](https://img.shields.io/badge/discuss-forum-ED1965?logo=discourse&logoColor=white)](https://forum.esa-worldcereal.org/)


<p align="center">
  <a href="https://esa-worldcereal.org/en" target="_blank">
    <img src="assets/worldcereal_logo.jpg" alt="logo" width="300"/>
  </a>
</p>

## Overview

**WorldCereal** is a Python package designed for generating cropland and crop type maps at a wide range of spatial scales, leveraging satellite and auxiliary data, and state-of-the-art classification workflows. It uses [**openEO**](https://openeo.org/) to run classification tasks in the cloud, by default the [**Copernicus Data Space Ecosystem (CDSE)**](https://dataspace.copernicus.eu/). 

Users can leverage the system in a notebook environment through [Terrascope](https://terrascope.be/en) or set up the environment locally using the provided installation options.

In order to run classification jobs on CDSE, users can get started with [monthly free openEO processing credits](https://documentation.dataspace.copernicus.eu/Quotas.html) by registering on the CDSE platform. Additional credits can be purchased, or users may soon be able to request them through the **ESA Network of Resources**.

## Features

- **Scalable**: Generate maps at a wide range of spatial scales.
- **Cloud-based Processing**: Leverages openEO to run classifications in the cloud.
- **Powerful classification pipeline**: WorldCereal builds upon [Presto](https://arxiv.org/abs/2304.14065), a pretrained transformer-based model, leveraging global self-supervised learning of multimodal input timeseries, leading to better accuracies and higher generalizability in space and time of downstream crop classification models. The Presto backbone of WorldCereal classification pipelines is developed [here](https://github.com/WorldCereal/presto-worldcereal).
- **Customizable**: Users can pick any region or temporal range and apply either default models or train their own and produce custom maps, interacting with publicly available training data.
- **Easy to Use**: Integrates into Jupyter notebooks and other Python environments.

## Quick Start

One of our [demo notebooks](notebooks) can introduce to you quickly how to map crops with the Worldcereal system. There's two options to run these notebooks:

#### Option 1: Run on Terrascope

You can use a preconfigured environment on [**Terrascope**](https://terrascope.be/en) to run the workflows in a Jupyter notebook environment. Just register as a new user on Terrascope or use one of the supported EGI eduGAIN login methods to get started.

| :point_up:    | Once you are prompted with "Server Options", make sure to select the "Worldcereal" image. Did you choose "Terrascope" by accident? Then go to File > Hub Control Panel > Stop my server, and click the link below once again.  |
|---------------|:------------------------|

- Access our classification app here: <a href="https://notebooks.terrascope.be/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FWorldCereal%2Fworldcereal-classification&urlpath=lab%2Ftree%2Fworldcereal-classification%2Fnotebooks%2Fworldcereal_classification_app.ipynb&branch=main"><img src="https://img.shields.io/badge/run%20cropland%20demo-Terrascope-brightgreen" alt="Run cropland demo" valign="middle"></a>

- For a demo on how to interact with the WorldCereal Reference Data Module (RDM): <a href="https://notebooks.terrascope.be/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FWorldCereal%2Fworldcereal-classification&urlpath=lab%2Ftree%2Fworldcereal-classification%2Fnotebooks%2Fworldcereal_RDM_demo.ipynb&branch=main"><img src="https://img.shields.io/badge/run%20RDM%20demo-Terrascope-brightgreen" alt="Run RDM demo" valign="middle"></a>

#### Option 2: Install Locally

If you prefer to install the package locally, you can create the environment using **Conda** or **pip**.

- **Conda**<br>
First clone the repository:
```bash
git clone https://github.com/WorldCereal/worldcereal-classification.git
cd worldcereal-classification
```
Next, install the package locally:
`conda env create -f environment.yml`

- **Pip**<br>
`pip install "worldcereal[train,notebooks] @ git+https://github.com/worldcereal/worldcereal-classification.git"`

WorldCereal requires Python 3.11 or newer.

## Usage Example
In its most simple form, a cropland mask can be generated with just a few lines of code, triggering an openEO job on CDSE and downloading the result locally:

```python
import geopandas as gpd
from openeo_gfmap import TemporalContext
from shapely.geometry import box

from worldcereal.job import WorldCerealTask
from worldcereal.job_params import WorldCerealJobParams
from worldcereal.jobmanager import run_worldcereal_task
from worldcereal.parameters import WorldCerealProductType

# Define the area of interest as a bounding box (minx, miny, maxx, maxy, EPSG:4326)
aoi_gdf = gpd.GeoDataFrame(
    geometry=[box(44.432274, 51.317362, 44.698802, 51.428224)],
    crs="EPSG:4326",
)
aoi_gdf["id"] = "aoi" 

# Specify the temporal extent (for cropland, this has to be a full year)
temporal_extent = TemporalContext("2022-11-01", "2023-10-31")

# Build the job parameters
params: WorldCerealJobParams = {
    "aoi_gdf": aoi_gdf,
    "output_dir": "./worldcereal_output",
    "temporal_extent": temporal_extent,
    "product_type": WorldCerealProductType.CROPLAND,
}

# Submit and monitor the job; results are downloaded automatically
run_worldcereal_task(WorldCerealTask.CLASSIFICATION, params)

```

This can easily be extended to also generate a crop type map based on your custom crop type model trained with the WorldCereal system (see our classification app for a detailed walkthrough):

```python
# For crop type classification you must supply at least one season window that
# covers the growing season of interest.  Each window is a TemporalContext
# keyed by an arbitrary season identifier.
season_specifications = {
    "tc-s1": TemporalContext("2022-11-01", "2023-04-30"),
}

# You can point to your own crop-type head (a .zip artifact) trained with the
# WorldCereal training utilities.  Leave croptype_head_zip out (or set it to
# None) to use the default model.
croptype_params: WorldCerealJobParams = {
    "aoi_gdf": aoi_gdf,
    "output_dir": "./worldcereal_output_croptype",
    "temporal_extent": temporal_extent,
    "product_type": WorldCerealProductType.CROPTYPE,
    "season_specifications": season_specifications,
    "croptype_head_zip": "/path/to/your/custom_croptype_head.zip",
}

run_worldcereal_task(WorldCerealTask.CLASSIFICATION, croptype_params)

```

## Documentation

Comprehensive documentation of the classification module is available at the following link: https://worldcereal.github.io/worldcereal-documentation/processing/overview.html

## Support
Questions, suggestions, feedback? Use [our forum](https://forum.esa-worldcereal.org/) to get in touch with us and the rest of the community!

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

The WorldCereal project is funded by the [European Space Agency (ESA)](https://www.esa.int/) under grant no. 4000130569/20/I-NB.

WorldCereal's classification backbone makes use of the Presto model, originally implemented [here](https://github.com/nasaharvest/presto/). Without the groundbreaking work being done by Gabriel Tseng and the rest of the [NASA Harvest](https://www.nasaharvest.org/) team, both in the original Presto implementation as well as its adaptation for WorldCereal, this package would simply not exist in its present form 🙏.

The pre-configured Jupyter notebook environment in which users can train custom models and launch WorldCereal jobs is provided by [Terrascope](https://terrascope.be/en), the Belgian Earth observation data space, managed by [VITO Remote Sensing](https://remotesensing.vito.be/) on behalf of the [Belgian Science Policy Office](https://www.belspo.be/belspo/index_en.stm)

## How to cite

If you use WorldCereal resources in your work, please cite it as follows:

```bibtex

@article{van_tricht_worldcereal_2023,
	title = {{WorldCereal}: a dynamic open-source system for global-scale, seasonal, and reproducible crop and irrigation mapping},
	volume = {15},
	issn = {1866-3516},
	shorttitle = {{WorldCereal}},
	url = {https://essd.copernicus.org/articles/15/5491/2023/},
	doi = {10.5194/essd-15-5491-2023},
	number = {12},
	urldate = {2024-03-01},
	journal = {Earth System Science Data},
	author = {Van Tricht, Kristof and Degerickx, Jeroen and Gilliams, Sven and Zanaga, Daniele and Battude, Marjorie and Grosu, Alex and Brombacher, Joost and Lesiv, Myroslava and Bayas, Juan Carlos Laso and Karanam, Santosh and Fritz, Steffen and Becker-Reshef, Inbal and Franch, Belén and Mollà-Bononad, Bertran and Boogaard, Hendrik and Pratihast, Arun Kumar and Koetz, Benjamin and Szantoi, Zoltan},
	month = dec,
	year = {2023},
	pages = {5491--5515},
}
```
