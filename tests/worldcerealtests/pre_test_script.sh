#!/bin/bash
echo "Starting pre_test_script.sh."

su - jenkins -c

# Install openeo-gfmap
GFMAP_URL="https://github.com/Open-EO/openeo-gfmap.git"
echo "Cloning the openeo-gfmap repository from $GFMAP_URL ..."
git clone $GFMAP_URL
cd openeo-gfmap || { echo "Directory not found! Exiting..."; exit 1; }
echo "Installing the openeo-gfmap ..."
pip install .
cd ..

# Install
PRESTO_URL="https://github.com/WorldCereal/presto-worldcereal.git"
echo "Cloning the presto-worldcereal repository from $PRESTO_URL ..."
git clone --branch updated-inferencedatasets $PRESTO_URL
cd presto-worldcereal || { echo "Directory not found! Exiting..."; exit 1; }
pip install .
cd ..

echo "pre_test_script.sh finished."

