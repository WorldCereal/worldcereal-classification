#!/bin/bash

# Install git
dnf install git -y

# # Install openeo-gfmap and presto-worldcereal
# dir=$(pwd)
# GFMAP_URL="https://github.com/Open-EO/openeo-gfmap.git"
# PRESTO_URL="https://github.com/WorldCereal/presto-worldcereal.git"

# su - jenkins -c "cd $dir && \
#   source venv310/bin/activate && \
#   git clone $GFMAP_URL && \
#   cd openeo-gfmap || { echo 'Directory not found! Exiting...'; exit 1; } && \
#   pip install . && \
#   cd ..
#   git clone -b croptype $PRESTO_URL && \
#   cd presto-worldcereal || { echo 'Directory not found! Exiting...'; exit 1; } && \
#   pip install .
# "

# For now only presto-worldcereal as gfmap is up to date on pypi
dir=$(pwd)

su - jenkins -c "cd $dir && \
  source venv310/bin/activate && \
  pip install git+https://github.com/WorldCereal/presto-worldcereal.git@croptype
"