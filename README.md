# WorldCereal classification module

This repository contains the code for running the WorldCereal classification module, making use of preprocessed WorldCereal input products.
The basic invocation is described in the next section.

Required packages can be installed from PyPI, except for the `satio` package which is hosted on VITO's Artifactory. Installing this package can be done through:

```pip install satio --extra-index-url https://artifactory.vgt.vito.be/api/pypi/python-packages/simple```

|:exclamation:  Note that the end-to-end functionality described below will not work out of the box, as this classification repository is part of a larger dockerized workflow taking care of input data gathering and preprocessing (released soon). |
|:-----------------------------------------|

## Running classification

Retrieve help (as shown at the bottom of this readme):

```python3.9 -m worldcereal --help```

Basic example invocation:

```python3.9 -m worldcereal 31UFS /path/to/config.json /path/to/outputfolder```

This command initiates the worldcereal product generation for all processing blocks of tile 31UFS, using the processing settings as described in the `config.json` file. All output products and accompanying log files will be written in the prescribed outputfolder location.

Example config files to run on data loaded directly from the WorldCereal s3 bucket are located under:

```sh
src/worldcereal/resources/exampleconfigs/example_bucketrun_annual_config.json  # For an annual cropland product run
src/worldcereal/resources/exampleconfigs/example_bucketrun_winter_config.json  # For a winter season product run
```

The optional `--blocks` argument allows to specify which processing blocks should be processed. Mind the proper invocation, e.g.:

```sh
--blocks='[0,1]'    # For a list of blocks
--blocks=100        # For a single block
```

The total number of blocks in a tile depends on the block size. Currently supported block sizes are 512 and 1024. The desired block size can be set either using an environment variable `EWOC_BLOCKSIZE`, or by parsing the optional `--block_size` flag. If the block size is not set explicitly, the default value of 512 will be taken.

The optional `--debug=True` argument can be added, which will initiate a run of only the first block in the list which is also buffered inward to reduce processing resources and processing time for debugging purposes.

The optional `--skip_processed=False` argument can be added to overwrite existing products instead of skipping those blocks entirely. The pipeline checks already processed products based on the `exitlogs` which are written after processing a block.

The list of environment variables that can be set and their meaning is:
```sh
EWOC_BLOCKSIZE              # Processing block size, defaults to 512
EWOC_AUXDATA                # Local path where auxdata is stored. If not set, simplified auxdata inside package is used (biomes)
EWOC_COLL_MAXGAP            # Maximum gap (in days) a collection can have before failing, defaults to 60
EWOC_COLL_MAXGAP_{coll_id}  # Maximum gap (in days) a specific collection ID (e.g. 'TIR') can have before failing. When set, it overrides EWOC_COLL_MAXGAP for that collection.
SATIO_MAXTHREADS            # Max. nr of concurrent threads per block loading input files, defaults to 5
SATIO_RETRIES               # Max. amount of loading attempts per file, defaults to 50
SATIO_DELAY                 # Nr. of seconds in between loading attempts, defaults to 5
SATIO_BACKOFF               # Multiplier for the delay after subsequent loading attempts, defaults to 1
SATIO_TIMEOUT               # Amount of seconds before raising TimeOutError when loading from s3 file, defaults to 180
```

Output of the `--help` command:

 ```sh
NAME
    __main__.py - Generates WorldCereal products.

SYNOPSIS
    __main__.py TILE CONFIGFILE OUTPUTFOLDER <flags>

DESCRIPTION
    Generates WorldCereal products.

POSITIONAL ARGUMENTS
    TILE
        Type: str
        MGRS tile ID to process. Example: '31UFS'
    CONFIGFILE
        Type: str
        path to config.json containing processing settings
    OUTPUTFOLDER
        Type: str
        path to use for saving products and logs

FLAGS
    --blocks=BLOCKS
        Type: Optional[typing.List]
        Default: None
        Block ids of the blocks to process from the given tile. Should be a sequence of integers between 0 and the total nr of blocks (depending on block size). If not provided, all blocks will be processed.
    --block_size=BLOCK_SIZE
        Type: Optional[int]
        Default: None
        The size of the processing blocks. If not provided, we try to take it from the environment variable "EWOC_BLOCKSIZE" or else use the default value 512.
    --skip_processed=SKIP_PROCESSED
        Type: bool
        Default: True
        Skip already processed blocks by checking the existlogs folder. Defaults to True.
    --debug=DEBUG
        Type: bool
        Default: False
        Run in debug mode, processing only one part of one block. Defaults to False.
    --process=PROCESS
        Type: bool
        Default: True
        If False, skip block processing
    --postprocess=POSTPROCESS
        Type: bool
        Default: True
        If False, skip post-processing to COG
    --raise_exceptions=RAISE_EXCEPTIONS
        Type: bool
        Default: False
        If True, immediately raise any unexpected exception instead of silently failing.
    --yearly_meteo=YEARLY_METEO
        Type: bool
        Default: True
        If True, use the AgERA5 collection suited to work with yearly composites of daily meteo data instead of daily files.
    --use_existing_features=USE_EXISTING_FEATURES
        Type: bool
        Default: False
        If True, processor will attempt to load the features from a file if they are available. Otherwise, normal feature computation will be done.
    --user=USER
        Type: str
        Default: '0000'
        User ID which will be written to STAC metadata defaults to "0000"
    --public=PUBLIC
        Type: bool
        Default: True
        Intended visibility of the of the created products. If True, STAC metadata will set public visibiliy flag to True, otherwise False.
    --aez_id=AEZ_ID
        Type: Optional[int]
        Default: None
        If provided, the AEZ ID will be enforced instead of automatically derived from the Sentinel-2 tile ID.
    --sparkcontext=SPARKCONTEXT
        Type: Optional[]
        Default: None
        Optional sparkcontext to parallellize block processing using spark.
        --force_start_date=FORCE_START_DATE
        Type: Optional[]
        Default: None
        if set, start_date will be overruled by this value
    --force_end_date=FORCE_END_DATE
        Type: Optional[]
        Default: None
        if set, end date will be overruled by this value

 ```
