import pytest
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
import geopandas as gpd

from satio.collections import (L2ATrainingCollection,
                               AgERA5TrainingCollection,
                               SIGMA0TrainingCollection,
                               PatchLabelsTrainingCollection,
                               TerrascopeSigma0Collection,
                               TerrascopeV200Collection,
                               DEMCollection,
                               WorldCoverCollection,
                               AgERA5Collection)
from satio.timeseries import load_timeseries

from worldcereal.collections import (TerrascopeSigma0TiledCollection,
                                     L8ThermalTiledCollection,
                                     L8ThermalTrainingCollection,
                                     WorldCerealOpticalTiledCollection,
                                     WorldCerealSigma0TiledCollection,
                                     WorldCerealThermalTiledCollection,
                                     AgERA5YearlyCollection)


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def get_test_resource(relative_path):
    dir = Path(os.path.dirname(os.path.realpath(__file__)))
    return dir / 'testresources' / relative_path


worldcereal_training_dataframe = {
    'location_id':
        {
            0: '0000280849BAC91C',
            1: '000028085BF08E7E',
            2: '00002806639B424A',
            3: '2021_TZA_COPERNICUS-GEOGLAM_POLY_1101950'
        },
    'tile':
        {
            0: '31UES',
            1: '31UES',
            2: '31UDS',
            3: '37MBS'
        },
    'ref_id':
        {
            0: '2019_BE_LPIS-Flanders',
            1: '2019_BE_LPIS-Flanders',
            2: '2019_BE_LPIS-Flanders',
            3: '2021_TZA_COPERNICUS-GEOGLAM'
        },
    'epsg':
        {
            0: '32631',
            1: '32631',
            2: '32631',
            3: '32737'
        },
    'path':
        {
            0: get_test_resource("worldcerealtraining/0000280849BAC91C"),
            1: get_test_resource("worldcerealtraining/000028085BF08E7E"),
            2: get_test_resource("worldcerealtraining/00002806639B424A"),
            3: get_test_resource("worldcerealtraining/2021_TZA_COPERNICUS-GEOGLAM_POLY_1101950"),  # NOQA
        },
    'bounds':
        {
            0: (574680, 5621800, 575320, 5622440),
            1: (526140, 5669560, 526780, 5670200),
            2: (501360, 5629040, 502000, 5629680),
            3: (260900, 9595160, 261540, 9595800)
        },
    'start_date':
        {
            0: '2018-08-01',
            1: '2018-08-01',
            2: '2017-08-01',
            3: '2019-09-23'
        },
    'end_date':
        {
            0: '2019-11-30',
            1: '2019-11-30',
            2: '2018-11-30',
            3: '2021-09-22'
        },
    'split':
        {
            0: 'CAL',
            1: 'VAL',
            2: 'CAL',
            3: 'TEST',
        },
    'validityTi':
        {
            0: '2019-06-01',
            1: '2019-06-01',
            2: '2018-06-01',
            3: '2021-05-06'
        }
}

terrascope_S2v200_dataframe = {
    "product":
    {
        "0": "S2B_20190816T104029_31UFS_TOC_V200",
        "1": "S2B_20190819T105029_31UFS_TOC_V200",
        "2": "S2A_20190821T104031_31UFS_TOC_V200",
        "3": "S2A_20190824T105031_31UFS_TOC_V200",
        "4": "S2A_20200825T104031_31UFS_TOC_V200",
        "5": "S2B_20190826T104029_31UFS_TOC_V200",
        "6": "S2A_20200828T105031_31UFS_TOC_V200",
        "7": "S2B_20190829T105029_31UFS_TOC_V200",
        "8": "S2B_20200830T103629_31UFS_TOC_V200"
    },
        "date":
            {
                "0": datetime.strptime('20190816T104029', '%Y%m%dT%H%M%S'),
                "1": datetime.strptime('20190819T105029', '%Y%m%dT%H%M%S'),
                "2": datetime.strptime('20190821T104031', '%Y%m%dT%H%M%S'),
                "3": datetime.strptime('20190824T105031', '%Y%m%dT%H%M%S'),
                "4": datetime.strptime('20200825T104031', '%Y%m%dT%H%M%S'),
                "5": datetime.strptime('20190826T104029', '%Y%m%dT%H%M%S'),
                "6": datetime.strptime('20200828T105031', '%Y%m%dT%H%M%S'),
                "7": datetime.strptime('20190829T105029', '%Y%m%dT%H%M%S'),
                "8": datetime.strptime('20200830T103629', '%Y%m%dT%H%M%S')
    },
        "path":
            {
                "0": get_test_resource("terrascopeV200/"
                                       "S2B_20190816T104029_31UFS_TOC_V200"),
                "1": get_test_resource("terrascopeV200/"
                                       "S2B_20190819T105029_31UFS_TOC_V200"),
                "2": get_test_resource("terrascopeV200/"
                                       "S2A_20190821T104031_31UFS_TOC_V200"),
                "3": get_test_resource("terrascopeV200/"
                                       "S2A_20190824T105031_31UFS_TOC_V200"),
                "4": get_test_resource("terrascopeV200/"
                                       "S2A_20200825T104031_31UFS_TOC_V200"),
                "5": get_test_resource("terrascopeV200/"
                                       "S2B_20190826T104029_31UFS_TOC_V200"),
                "6": get_test_resource("terrascopeV200/"
                                       "S2A_20200828T105031_31UFS_TOC_V200"),
                "7": get_test_resource("terrascopeV200/"
                                       "S2B_20190829T105029_31UFS_TOC_V200"),
                "8": get_test_resource("terrascopeV200/"
                                       "S2B_20200830T103629_31UFS_TOC_V200")
    },
        "tile":
            {
                "0": "31UFS",
                "1": "31UFS",
                "2": "31UFS",
                "3": "31UFS",
                "4": "31UFS",
                "5": "31UFS",
                "6": "31UFS",
                "7": "31UFS",
                "8": "31UFS"
    },
        "epsg":
            {
                "0": "32631",
                "1": "32631",
                "2": "32631",
                "3": "32631",
                "4": "32631",
                "5": "32631",
                "6": "32631",
                "7": "32631",
                "8": "32631"
    }
}

terrascope_sigma0_dataframe = {
    "product":
    {
        "0": "S1A_IW_GRDH_SIGMA0_DV_20190815T173249_ASCENDING_161_91E0_V110",
        "1": "S1B_IW_GRDH_SIGMA0_DV_20190816T172354_ASCENDING_88_5251_V110",
        "2": "S1B_IW_GRDH_SIGMA0_DV_20190821T173224_ASCENDING_161_AB15_V110",
        "3": "S1A_IW_GRDH_SIGMA0_DV_20190822T172443_ASCENDING_88_8302_V110",
        "4": "S1A_IW_GRDH_SIGMA0_DV_20190827T173250_ASCENDING_161_22A0_V110",
        "5": "S1B_IW_GRDH_SIGMA0_DV_20190828T172354_ASCENDING_88_188F_V110",
        "6": "S1B_IW_GRDH_SIGMA0_DV_20190828T172354_ASCENDING_88_188F_V110",
        "7": "S1A_IW_GRDH_SIGMA0_DV_20200828T172450_ASCENDING_88_77A9_V110"
    },
        "date":
            {
                "0": datetime.strptime('20190815T173249', '%Y%m%dT%H%M%S'),
                "1": datetime.strptime('20190816T172354', '%Y%m%dT%H%M%S'),
                "2": datetime.strptime('20190821T173224', '%Y%m%dT%H%M%S'),
                "3": datetime.strptime('20190822T172443', '%Y%m%dT%H%M%S'),
                "4": datetime.strptime('20190827T173250', '%Y%m%dT%H%M%S'),
                "5": datetime.strptime('20190828T172354', '%Y%m%dT%H%M%S'),
                "6": datetime.strptime('20190828T172354', '%Y%m%dT%H%M%S'),
                "7": datetime.strptime('20200828T172450', '%Y%m%dT%H%M%S')
    },
        "path":
            {
                "0": get_test_resource(
                    "terrascopeSigma0/S1A_IW_GRDH_SIGMA0_DV_"
                    "20190815T173249_ASCENDING_161_91E0_V110"),
                "1": get_test_resource(
                    "terrascopeSigma0/S1B_IW_GRDH_SIGMA0_DV_"
                    "20190816T172354_ASCENDING_88_5251_V110"),
                "2": get_test_resource(
                    "terrascopeSigma0/S1B_IW_GRDH_SIGMA0_DV_"
                    "20190821T173224_ASCENDING_161_AB15_V110"),
                "3": get_test_resource(
                    "terrascopeSigma0/S1A_IW_GRDH_SIGMA0_DV_"
                    "20190822T172443_ASCENDING_88_8302_V110"),
                "4": get_test_resource(
                    "terrascopeSigma0/S1A_IW_GRDH_SIGMA0_DV_"
                    "20190827T173250_ASCENDING_161_22A0_V110"),
                "5": get_test_resource(
                    "terrascopeSigma0/S1B_IW_GRDH_SIGMA0_DV_"
                    "20190828T172354_ASCENDING_88_188F_V110"),
                "6": get_test_resource(
                    "terrascopeSigma0/S1B_IW_GRDH_SIGMA0_DV_"
                    "20190828T172354_ASCENDING_88_188F_V110"),
                "7": get_test_resource(
                    "terrascopeSigma0/S1A_IW_GRDH_SIGMA0_DV_"
                    "20200828T172450_ASCENDING_88_77A9_V110")
    },
        "tile":
            {
                "0": "31UFS",
                "1": "31UFS",
                "2": "31UFS",
                "3": "31UFS",
                "4": "31UFS",
                "5": "31UFS",
                "6": "31UFS",
                "7": "31UFS"
    },
        "epsg":
            {
                "0": "32631",
                "1": "32631",
                "2": "32631",
                "3": "32631",
                "4": "32631",
                "5": "32631",
                "6": "32631",
                "7": "32631"
    }
}

terrascope_sigma0_tiled_dataframe = {
    "product":
    {
        "0": "S1A_IW_GRDH_SIGMA0_DV_20180815T172437_ASCENDING_88_8C73_V110",
        "1": "S1A_IW_GRDH_SIGMA0_DV_20180820T173243_ASCENDING_161_D90C_V110",
        "2": "S1A_IW_GRDH_SIGMA0_DV_20180820T173308_ASCENDING_161_6BF2_V110",
        "3": "S1A_IW_GRDH_SIGMA0_DV_20180827T172438_ASCENDING_88_4373_V110",
        "4": "S1A_IW_GRDH_SIGMA0_DV_20190810T172443_ASCENDING_88_775D_V110",
        "5": "S1A_IW_GRDH_SIGMA0_DV_20190815T173249_ASCENDING_161_91E0_V110",
        "6": "S1A_IW_GRDH_SIGMA0_DV_20190815T173314_ASCENDING_161_6843_V110",
        "7": "S1B_IW_GRDH_SIGMA0_DV_20190816T172354_ASCENDING_88_5251_V110",
        "8": "S1B_IW_GRDH_SIGMA0_DV_20190816T172419_ASCENDING_88_301F_V110",
        "9": "S1B_IW_GRDH_SIGMA0_DV_20190821T173159_ASCENDING_161_E37A_V110",
        "10": "S1A_IW_GRDH_SIGMA0_DV_20190822T172443_ASCENDING_88_8302_V110",
        "11": "S1A_IW_GRDH_SIGMA0_DV_20190827T173250_ASCENDING_161_22A0_V110",
        "12": "S1A_IW_GRDH_SIGMA0_DV_20190827T173315_ASCENDING_161_138D_V110",
        "13": "S1A_IW_GRDH_SIGMA0_DV_20190903T172444_ASCENDING_88_81F3_V110"
    },
        "date":
            {
                "0": datetime.strptime('20180815T172437', '%Y%m%dT%H%M%S'),
                "1": datetime.strptime('20180820T173243', '%Y%m%dT%H%M%S'),
                "2": datetime.strptime('20180820T173308', '%Y%m%dT%H%M%S'),
                "3": datetime.strptime('20180827T172438', '%Y%m%dT%H%M%S'),
                "4": datetime.strptime('20190810T172443', '%Y%m%dT%H%M%S'),
                "5": datetime.strptime('20190815T173249', '%Y%m%dT%H%M%S'),
                "6": datetime.strptime('20190815T173314', '%Y%m%dT%H%M%S'),
                "7": datetime.strptime('20190816T172354', '%Y%m%dT%H%M%S'),
                "8": datetime.strptime('20190816T172419', '%Y%m%dT%H%M%S'),
                "9": datetime.strptime('20190821T173159', '%Y%m%dT%H%M%S'),
                "10": datetime.strptime('20190822T172443', '%Y%m%dT%H%M%S'),
                "11": datetime.strptime('20190827T173250', '%Y%m%dT%H%M%S'),
                "12": datetime.strptime('20190827T173315', '%Y%m%dT%H%M%S'),
                "13": datetime.strptime('20190903T172444', '%Y%m%dT%H%M%S'),
    },
        "path":
            {
                "0": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20180815T172437_ASCENDING_88_8C73_V110"),
                "1": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20180820T173243_ASCENDING_161_D90C_V110"),
                "2": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20180820T173308_ASCENDING_161_6BF2_V110"),
                "3": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20180827T172438_ASCENDING_88_4373_V110"),
                "4": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20190810T172443_ASCENDING_88_775D_V110"),
                "5": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20190815T173249_ASCENDING_161_91E0_V110"),
                "6": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20190815T173314_ASCENDING_161_6843_V110"),
                "7": get_test_resource(
                    "terrascopeSigma0Tiled/S1B_IW_GRDH_SIGMA0_DV_"
                    "20190816T172354_ASCENDING_88_5251_V110"),
                "8": get_test_resource(
                    "terrascopeSigma0Tiled/S1B_IW_GRDH_SIGMA0_DV_"
                    "20190816T172419_ASCENDING_88_301F_V110"),
                "9": get_test_resource(
                    "terrascopeSigma0Tiled/S1B_IW_GRDH_SIGMA0_DV_"
                    "20190821T173159_ASCENDING_161_E37A_V110"),
                "10": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20190822T172443_ASCENDING_88_8302_V110"),
                "11": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20190827T173250_ASCENDING_161_22A0_V110"),
                "12": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20190827T173315_ASCENDING_161_138D_V110"),
                "13": get_test_resource(
                    "terrascopeSigma0Tiled/S1A_IW_GRDH_SIGMA0_DV_"
                    "20190903T172444_ASCENDING_88_81F3_V110"),
    },
        "tile":
            {
                "0": "31UFS",
                "1": "31UFS",
                "2": "31UFS",
                "3": "31UFS",
                "4": "31UFS",
                "5": "31UFS",
                "6": "31UFS",
                "7": "31UFS",
                "8": "31UFS",
                "9": "31UFS",
                "10": "31UFS",
                "11": "31UFS",
                "12": "31UFS",
                "13": "31UFS",
    },
        "epsg":
            {
                "0": "32631",
                "1": "32631",
                "2": "32631",
                "3": "32631",
                "4": "32631",
                "5": "32631",
                "6": "32631",
                "7": "32631",
                "8": "32631",
                "9": "32631",
                "10": "32631",
                "11": "32631",
                "12": "32631",
                "13": "32631",
    }
}

agera5yearly_dataframe = {
    "product": {
        0: "AgERA5_2018",
        1: "AgERA5_2019",
    },
    "date": {
        0: "2018-01-01",
        1: "2019-01-01"
    },
    "path": {
        0: get_test_resource("WORLDCEREAL_PREPROC/"
                             "METEO/2018"),
        1: get_test_resource("WORLDCEREAL_PREPROC/"
                             "METEO/2019")
    },
    "tile": {
        0: "global",
        1: "global"
    },
    "epsg": {
        0: "4326",
        1: "4326"
    }
}


@ pytest.fixture
def worldcereal_training_df():
    return pd.DataFrame.from_dict(
        worldcereal_training_dataframe)


@ pytest.fixture
def AgERA5Training_collection():
    return AgERA5TrainingCollection(pd.DataFrame.from_dict(
        worldcereal_training_dataframe), dataformat='worldcereal')


@ pytest.fixture
def Sigma0Training_collection():
    return SIGMA0TrainingCollection(pd.DataFrame.from_dict(
        worldcereal_training_dataframe), dataformat='worldcereal')


@ pytest.fixture
def L2ATraining_collection():
    return L2ATrainingCollection(pd.DataFrame.from_dict(
        worldcereal_training_dataframe), dataformat='worldcereal')


@ pytest.fixture
def L8ThermalTraining_collection():
    return L8ThermalTrainingCollection(pd.DataFrame.from_dict(
        worldcereal_training_dataframe), dataformat='worldcereal')


@ pytest.fixture
def PatchLabelsTraining_collection():
    return PatchLabelsTrainingCollection(pd.DataFrame.from_dict(
        worldcereal_training_dataframe), dataformat='worldcereal')


@ pytest.fixture
def AgERA5_timeseries(AgERA5Training_collection):
    return load_timeseries(AgERA5Training_collection, "precipitation_flux")


@ pytest.fixture
def TerrascopeSigma0_collection():
    return TerrascopeSigma0Collection(pd.DataFrame.from_dict(
        terrascope_sigma0_dataframe)).filter_bounds(
            bounds=(650000, 5640200, 650060, 5640260),
            epsg=32631)


@ pytest.fixture
def TerrascopeSigma0Tiled_collection():
    return TerrascopeSigma0TiledCollection(pd.DataFrame.from_dict(
        terrascope_sigma0_tiled_dataframe)).filter_bounds(
            bounds=(650000, 5640200, 650060, 5640260),
            epsg=32631)


@ pytest.fixture
def WorldCerealSigma0Tiled_collection():
    return WorldCerealSigma0TiledCollection.from_folder(
        get_test_resource('WORLDCEREAL_PREPROC/SAR')
    ).filter_bounds(
        bounds=(661440, 9648800, 661500, 9648860),
        epsg=32736)


@ pytest.fixture
def WorldCerealOpticalTiled_collection():
    return WorldCerealOpticalTiledCollection.from_folder(
        get_test_resource('WORLDCEREAL_PREPROC/OPTICAL')
    ).filter_bounds(
        bounds=(661440, 9648800, 661500, 9648860),
        epsg=32736)


@ pytest.fixture
def WorldCerealThermalTiled_collection():
    return WorldCerealThermalTiledCollection.from_folder(
        get_test_resource('WORLDCEREAL_PREPROC/TIR')
    ).filter_bounds(
        bounds=(300000, 4189760, 300060, 4189820),
        epsg=32630)


@ pytest.fixture
def WorldCerealAgERA5Yearly_collection():
    df = pd.DataFrame.from_dict(
        agera5yearly_dataframe)
    df.date = pd.to_datetime(df.date)

    return AgERA5YearlyCollection(
        df).filter_bounds(
        bounds=(661440, 9648800, 661500, 9648860),
        epsg=32736)


@ pytest.fixture
def L8ThermalTiled_collection():
    return L8ThermalTiledCollection.from_folders(
        '/data/worldcereal/data/L8_LST_tiled').filter_bounds(
            bounds=(650000, 5640200, 650060, 5640260),
            epsg=32631)


@ pytest.fixture
def TerrascopeV200_collection():
    return TerrascopeV200Collection(pd.DataFrame.from_dict(
        terrascope_S2v200_dataframe)).filter_bounds(
            bounds=(650000, 5640200, 650060, 5640260),
            epsg=32631)


@ pytest.fixture
def dem_collection():
    return DEMCollection(folder=get_test_resource(
        "worldcerealtraining/dem")
    )


@ pytest.fixture
def worldcover_collection():
    return WorldCoverCollection(folder=get_test_resource("worldcover"))


@ pytest.fixture
def agera5_collection():
    return AgERA5Collection.from_path('/data/MTDA/AgERA5')


@ pytest.fixture
def worldcereal_cib_database():
    return gpd.read_file(get_test_resource("cib/database.json"))


@ pytest.fixture
def worldcereal_aez():
    return gpd.read_file(get_test_resource("aez/AEZ_incl_groups.shp"))


@ pytest.fixture
def worldcereal_outputs():
    return get_test_resource("WORLDCEREAL_PRODUCTS")


@ pytest.fixture
def training_df_LC():
    return get_test_resource("worldcerealtraining")


@ pytest.fixture
def temporarycrops_config():
    return get_test_resource("config/example_bucketrun_annual_config.json")


@ pytest.fixture
def selected_features():
    selected_features = get_test_resource(
        "worldcerealtraining/selected_features.txt")
    ft_selection = []
    with open(selected_features) as f:
        lines = f.readlines()
        for line in lines:
            ft_selection.append(line.strip())
    return ft_selection
