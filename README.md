# WorldCereal classification module <!-- omit in toc -->
[![Tests](https://github.com/WorldCereal/worldcereal-classification/actions/workflows/ci.yaml/badge.svg)](https://github.com/WorldCereal/worldcereal-classification/actions/workflows/ci.yaml) [![DOI](https://zenodo.org/badge/621251443.svg)](https://zenodo.org/badge/latestdoi/621251443) [![Terrascope Environment](https://img.shields.io/badge/try%20on-Terrascope-blue)](https://terrascope-url) [![Documentation](https://img.shields.io/badge/docs-WorldCereal%20Documentation-blue)](https://worldcereal.github.io/worldcereal-documentation/)

<p align="center">
  <a href="https://esa-worldcereal.org/en" target="_blank">
    <img src="assets/worldcereal_logo.jpg" alt="logo" width="300"/>
  </a>
</p>

## Overview

**WorldCereal** is a Python package designed for generating cropland and crop type maps at a wide range of spatial scales, leveraging satellite and auxiliary data, and state-of-the-art classification workflows. It uses [openEO](https://openeo.org/) to run classification tasks in the cloud, by default the [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/). 

Users can leverage the system in a notebook environment through [Terrascope](https://terrascope.be/en) or set up the environment locally using the provided installation options.

In order to run classification jobs on CDSE, users can get started with **monthly free processing credits** by registering on the CDSE platform. Additional credits can be purchased, or users may soon be able to request them through the **ESA Network of Resources**.

## Features

- **Scalable**: Generate maps at a wide range of spatial scales.
- **Cloud-based Processing**: Leverages openEO to run classifications in the cloud.
- **Customizable**: Users can pick any region or temporal range and apply either default models or train their own and produce custom maps, interacting with publicly available training data.
- **Easy to Use**: Integrates into Jupyter notebooks and other Python environments.

## Quick Start

#### Option 1: Run on Terrascope

You can use a preconfigured environment on [**Terrascope**](https://terrascope.be/en) to run the workflows in a Jupyter notebook environment.

[![Terrascope Environment](https://img.shields.io/badge/try%20on-Terrascope-blue)](https://terrascope-url)

#### Option 2: Install Locally

If you prefer to install the package locally, you can create the environment using **Conda** or **pip**.

First clone the repository:
```bash
git clone https://github.com/WorldCereal/worldcereal-classification.git
cd worldcereal-classification
```
Next, install the package locally:
- for Conda: `conda env create -f environment.yml`
- for Pip: `pip install .`

## Usage Example

```
simple code snippet
```

## Documentation

Comprehensive documentation is available at the following link: https://worldcereal.github.io/worldcereal-documentation/

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](/home/kristofvt/git/worldcereal-classification/LICENSE) file for details.

## Acknowledgments

...


## How to cite

If you use the WorldCereal classification package in your work, please cite it as follows:

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

