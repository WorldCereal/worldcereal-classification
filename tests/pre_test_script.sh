#!/bin/bash

# Install git
su - jenkins -c "dnf install git -y"

# Install openeo-gfmap
GFMAP_URL="https://github.com/Open-EO/openeo-gfmap.git"
su - jenkins -c "git clone $GFMAP_URL"
cd openeo-gfmap || { echo "Directory not found! Exiting..."; exit 1; }
su - jenkins -c "pip install ."
cd ..

# Install
PRESTO_URL="https://github.com/WorldCereal/presto-worldcereal.git"
su - jenkins -c "git clone --branch updated-inferencedatasets $PRESTO_URL"
cd presto-worldcereal || { echo "Directory not found! Exiting..."; exit 1; }
su - jenkins -c "pip install ."
cd ..
