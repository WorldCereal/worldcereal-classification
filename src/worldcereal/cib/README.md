# Common input baseline (CIB) creation

Each CIB gets an experiment ID such as `CIB_V1` and will hold all extracted input/output patches and metadata describing the samples. These samples originate from many different reference datafiles, which all need to undergo a processing pipeline to make them ready to use in the CIB benchmarking experiment. For each reference datafiles, the following steps need to be followed in order.

## 0. Extract a random sample from the available samples
In case the original reference file is too large for CIB extraction (in case of LPIS for example), a stratified random sample (based on crop type) can be selected.
As some of the files are too big to process on a small VM, this step should be executed on a machine with enough RAM, currently it's a Windows machine.

`scripts/cib/ref/randomsample_shp.py`

## 1. Prepare input shapefiles for further processing
In this step, we start from a polygon or point shapefile and automatically add all required attributes for further processing such as:
- We find the appropriate S2 TILE ID and the EPSG code for the corresponding UTM Zone
- We calculate the shifted center coordinates to coincide with closest 20m pixel
- Based on the shifted coordinates and the provided kernel size (320m on each side) the bounds for a patch are calculated in the correct coordinates
- All available attributes are copied to the destination json files

The basic requirement is that the original sample file is formatted as shapefile with following naming convention:
`/data/worldcereal/data/ref/VITO_processed/{contenttype}/{labeltype}/{year}/{ref_id}_{labeltype}_{contenttype}/{ref_id}_{labeltype}_{contenttype}.shp)`
And optionally accompanied by a CSV describing the sampleIDs to extract a subset:
`/data/worldcereal/data/ref/VITO_processed/{contenttype}/{labeltype}/{year}/{ref_id}_{labeltype}_{contenttype}/{ref_id}_{labeltype}_{contenttype}_samples.csv)`

To run the preprocessing using spark (set the proper parameters in the script itself):
`bash scripts/cib/labels/preprocess_trainingpoints.sh`

## 2. Export output labels to NetCDF
In this step, the NetCDF output files are created for each sample. This includes automatic rasterization of point/polygon features. The required input for this step is the path to the `*_samples.json` file created in the previous step.

- In case of sequential running: `scripts/cib/labels/labels_to_cib.py`
- To run in parallel on spark: `scripts/cib/labels/labels_to_cib.sh`

## 3. Create AgERA5 CIB inputs
In this step, the NetCDF input files for AgERA5 data are created for each sample.

To run in parallel on spark: `scripts/cib/inputs/create_agera5_cib.sh`

## 4. Create crop calendar CIB inputs

NOTE: this step is no longer necessary!

In this step, the NetCDF input files for crop calendar data are created for each sample.

To run in parallel on spark: `scripts/cib/inputs/create_cropcalendars_cib.sh`

## 5. Do Google Earth Engine extractions for samples
Here, the locations of the samples, the kernel size and the temporal range are input to a script that acquires input imagery stacks from Google Earth Engine as TFrecord files, which are subsequently decoded into NetCDF files. The scripts are part of the external gee-export repository.

The extractions are started with:
`gee-exporter/scripts/worldcereal/gee_submit.sh`

!!! In case you need data prior to dec 2018 (outside Europe) or 2017 (Europe), you should perform the extractions through Sentinel-hub, with:
`gee-exporter/scripts/worldcereal/shub_submit.sh`

The decoding (only needed when extractions were done using GEE) is started with:
`gee-exporter/scripts/worldcereal/decode_data.sh`

## 6. Setup a CIB master database
Once all NetCDF files have been created and the CIB is complete, the final step is to create one database that holds all the individual training samples, no matter their type (POINT/POLYGON/MAP) or content (LC/CT/IRR). This CIB database is stored as GeoJSON in the root folder of the CIB experiment and serves as the entry point for CIB analyses. From this database, one can easily filter the samples for specific characteristics and immediately get a pointer to the location of inputs and output NetCDF files for each sample.

It is build with spark with: 
`scripts/cib/database/build_cib_database.sh`

Note that during building of the database, all NetCDF files are opened to check for consistency. If a sample does not meet all requirements, it will be added to a different database as well (`issues.json`), and dropped from the master database.
