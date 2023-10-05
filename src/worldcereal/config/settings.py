from copy import deepcopy
from pathlib import Path

# ---------------------------------------------------
# DYNAMIC SETTINGS

STARTMONTH = 3
ENDMONTH = 10
YEAR = 2021
S1_ORBITDIRECTION = 'DESCENDING'
MODEL_PATH = 'https://artifactory.vgt.vito.be/auxdata-public/hrlvlcc/croptype_models/20230615T144208-24ts-hrlvlcc-v200.zip'  # NOQA
CROPCLASS_LIBRARY = 'https://artifactory.vgt.vito.be:443/auxdata-public/hrlvlcc/openeo-dependencies/cropclass-1.0.2-20230623T152443.zip'  # NOQA
VITOCROPCLASSIFICATION_LIBRARY = 'https://artifactory.vgt.vito.be/auxdata-public/hrlvlcc/openeo-dependencies/vitocropclassification-1.4.0-20230619T091529.zip'  # NOQA
STATIC_LIBRARIES = 'https://artifactory.vgt.vito.be/auxdata-public/hrlvlcc/openeo-dependencies/hrl.zip'  # NOQA

# ---------------------------------------------------
# Processing options for crop type map generation

PROCESSING_OPTIONS = {
    "start_month": STARTMONTH,
    "end_month": ENDMONTH,
    "year": YEAR,
    "modeldir": "tmp/model/",
    "modeltag": Path(MODEL_PATH).stem,
    "target_crs": 3035,
    "s1_orbitdirection": S1_ORBITDIRECTION,
    "all_probabilities": True  # By default save all probabilities
}

# ---------------------------------------------------
# Job options for OpenEO inference

# Default job options
DEFAULT_JOB_OPTIONS = {
    "driver-memory": "2500m",
    "driver-memoryOverhead": "512m",
    "driver-cores": "1",
    "executor-memory": "850m",
    "executor-memoryOverhead": "2900m",
    "executor-cores": "1",
    "executor-request-cores": "600m",
    "max-executors": "22",
    "executor-threads-jvm": "7",
    "udf-dependency-archives": [
        f"{MODEL_PATH}#tmp/model",
        f"{CROPCLASS_LIBRARY}#tmp/cropclasslib",
        f"{VITOCROPCLASSIFICATION_LIBRARY}#tmp/vitocropclassification",
        f"{STATIC_LIBRARIES}#tmp/venv_static"
    ],
    "logging-threshold": "info",
    "mount_tmp": False,
    "goofys": "false",
    "node_caching": True
}

# Terrascope backend specific options
TERRASCOPE_JOB_OPTIONS = {
    "task-cpus": 1,
    "executor-cores": 1,
    "max-executors": 32,
    "queue": "default",
    # terrascope reads from geotiff instead of jp2, so no threading issue there
    "executor-threads-jvm": 12,
    "executor-memory": "3g",
    "executor-memoryOverhead": "3g"
}


# Sentinelhub layers specific options
SENTINELHUB_JOB_OPTIONS = {
    "sentinel-hub": {
        "client-alias": "vito"
    }
}

# ---------------------------------------------------
# Job options for OpenEO training data extractions

OPENEO_EXTRACT_JOB_OPTIONS = {
    "driver-memory": "4G",
    "driver-memoryOverhead": "4G",
    "driver-cores": "2",
    "executor-memory": "3G",
    "executor-memoryOverhead": "2G",
    "executor-cores": "2",
    "max-executors": "50",
    "soft-errors": "true"
}

OPENEO_EXTRACT_CREO_JOB_OPTIONS = {
    "driver-memory": "4G",
    "driver-memoryOverhead": "2G",
    "driver-cores": "1",
    "executor-memory": "2000m",
    "executor-memoryOverhead": "3500m",
    "executor-cores": "4",
    "executor-request-cores": "400m",
    "max-executors": "200"
}

# ---------------------------------------------------
# Collection options for OpenEO inference

# Collection definitions on Terrascope
_TERRASCOPE_COLLECTIONS = {
    'S2_collection': "TERRASCOPE_S2_TOC_V2",
    'WORLDCOVER_collection': "ESA_WORLDCOVER_10M_2021_V2",
    'METEO_collection': 'AGERA5',
    'S1_collection': "SENTINEL1_GRD_SIGMA0",
    'DEM_collection': "COPERNICUS_30"
}

# Collection definitions on CREO
_CREO_COLLECTIONS = {
    'S2_collection': "SENTINEL2_L2A",
    'WORLDCOVER_collection': None,
    'METEO_collection': None,
    'S1_collection': "SENTINEL1_GRD",
    'DEM_collection': "COPERNICUS_30"
}

# Collection definitions on Sentinelhub
_SENTINELHUB_COLLECTIONS = {
    'S2_collection': "SENTINEL2_L2A_SENTINELHUB",
    'WORLDCOVER_collection': "ESA_WORLDCOVER_10M_2021_V2",
    'METEO_collection': None,
    'S1_collection': "SENTINEL1_GRD",
    'DEM_collection': "COPERNICUS_30"
}


def _get_default_job_options(task: str = 'inference'):
    if task == 'inference':
        return DEFAULT_JOB_OPTIONS
    elif task == 'extractions':
        return OPENEO_EXTRACT_JOB_OPTIONS
    else:
        raise ValueError(f'Task `{task}` not known.')


def get_job_options(provider: str = None,
                    task: str = 'inference'):

    job_options = deepcopy(_get_default_job_options(task))

    if task == 'inference':
        if provider.lower() == 'terrascope':
            job_options.update(TERRASCOPE_JOB_OPTIONS)
        elif provider.lower() == 'sentinelhub' or provider.lower() == 'shub':
            job_options.update(SENTINELHUB_JOB_OPTIONS)
        elif provider.lower() == 'creodias':
            pass
        elif provider is None:
            pass
        else:
            raise ValueError(f'Provider `{provider}` not known.')

    elif task == 'extractions':
        if provider.lower() == 'creodias':
            job_options.update(OPENEO_EXTRACT_CREO_JOB_OPTIONS)

    return deepcopy(job_options)


def _get_default_processing_options():
    return deepcopy(PROCESSING_OPTIONS)


def get_processing_options(provider: str = 'terrascope'):
    processing_options = _get_default_processing_options()
    processing_options.update({'provider': provider})
    return deepcopy(processing_options)


def get_collection_options(provider: str):

    if provider.lower() == 'terrascope':
        return _TERRASCOPE_COLLECTIONS
    elif provider.lower() == 'sentinelhub' or provider.lower() == 'shub':
        return _SENTINELHUB_COLLECTIONS
    elif 'creo' in provider.lower():
        return _CREO_COLLECTIONS
    else:
        raise ValueError(f'Provider `{provider}` not known.')
