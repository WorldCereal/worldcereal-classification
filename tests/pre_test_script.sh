#!/bin/bash

# Install git
dnf install git -y

# Install openeo-gfmap and presto-worldcereal
dir=$(pwd)
GFMAP_URL="https://github.com/Open-EO/openeo-gfmap.git"
PRESTO_URL="https://github.com/WorldCereal/presto-worldcereal.git"

su - jenkins -c "cd $dir && \
  source venv310/bin/activate && \
  git clone $GFMAP_URL && \
  cd openeo-gfmap || { echo 'Directory not found! Exiting...'; exit 1; } && \
  pip install . && \
  cd ..
  git clone $PRESTO_URL && \
  cd presto-worldcereal || { echo 'Directory not found! Exiting...'; exit 1; } && \
  pip install .
"