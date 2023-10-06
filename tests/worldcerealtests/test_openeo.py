import json
import os
import tempfile
from pathlib import Path

import logging

import numpy as np
import openeo
import openeo.processes
import pytest
import xarray

from worldcereal.settings import (get_collection_options, get_job_options,
                                  get_processing_options)
from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs

basedir = Path(os.path.dirname(os.path.realpath(__file__)))
logger = logging.getLogger(__name__)


@pytest.fixture
def vito_connection(capfd):
    # Note: this generic `authenticate_oidc()` call allows both:
    # - device code/refresh token based authentication for manual test
    #   suiteruns by a developer
    # - client credentials auth through env vars for automated/Jenkins CI runs
    #
    # See https://open-eo.github.io/openeo-python-client/auth.html#oidc-authentication-dynamic-method-selection  # NOQA
    # and Jenkinsfile, where Jenkins fetches the env vars from VITO TAP Vault.
    connection = openeo.connect("openeo.vito.be")
    with capfd.disabled():
        # Temporarily disable output capturing, to make sure that OIDC device
        # code instructions (if any) are shown.
        connection.authenticate_oidc()
    return connection


@pytest.fixture
def creo_connection(capfd):
    # Like `vito_connection`, but with a different OIDC provider and
    # client id/secret for Creodias/CDSE. However, because env vars
    # are a global thing, and `vito_connection` is already consuming
    # the built-in env var handling of `authenticate_oidc()` for client
    # credentials auth, we have to roll our own env-var handling logic here
    # Also see https://github.com/Open-EO/openeo-python-client/issues/435
    connection = openeo.connect("openeo.creo.vito.be")

    if os.environ.get("OPENEO_AUTH_METHOD") == "client_credentials":
        provider_id = os.environ.get("OPENEO_AUTH_CDSE_PROVIDER_ID")
        client_id = os.environ.get("OPENEO_AUTH_CDSE_CLIENT_ID")
        client_secret = os.environ.get("OPENEO_AUTH_CDSE_CLIENT_SECRET")
        connection.authenticate_oidc_client_credentials(
            provider_id=provider_id, client_id=client_id,
            client_secret=client_secret)
    else:
        with capfd.disabled():
            # Temporarily disable output capturing, to make sure that
            # OIDC device code instructions (if any) are shown.
            connection.authenticate_oidc()

    return connection


START_DATE, END_DATE = '2021-01-01', '2021-12-31'
X = 3740000.0
Y = 3020000.0
EXTENT = dict(zip(["west", "south", "east", "north"], [
    X, Y, X + 10 * 15, Y + 10 * 15]))
EXTENT["crs"] = 3035
EXTENT["srs"] = 3035


def test_preprocessing(vito_connection):

    provider = 'terrascope'

    x = 3740000.0
    y = 3020000.0
    EXTENT_20KM = dict(zip(["west", "south", "east", "north"], [
                       x, y, x + 10 * 256, y + 10 * 256]))
    EXTENT_20KM["crs"] = 3035
    EXTENT_20KM["srs"] = 3035

    collections_options = get_collection_options(provider)

    input_cube = worldcereal_preprocessed_inputs(
        vito_connection,
        EXTENT_20KM,
        START_DATE,
        END_DATE,
        target_crs=3035,
        masking="mask_scl_dilation",
        provider=provider,
        **collections_options,
    )

    # Ref file with processing graph
    ref_graph = basedir / 'testresources' / 'terrascope_graph.json'

    # uncomment to save current graph to the ref file
    with open(ref_graph, 'w') as f:
        f.write(json.dumps(input_cube.flat_graph(), indent=4))

    with open(ref_graph, 'r') as f:
        expected = json.load(f)
        assert expected == input_cube.flat_graph()


@pytest.mark.skip
def test_fetch_inputs_optical(vito_connection):

    temp_output_file = tempfile.NamedTemporaryFile()

    input_cube = worldcereal_preprocessed_inputs(
        vito_connection,
        EXTENT,
        START_DATE,
        END_DATE,
        masking='mask_scl_dilation',
        S1_collection=None,
        METEO_collection=None,
        DEM_collection=None,
        target_crs=3035
    )

    input_cube.download(temp_output_file.name, format='NetCDF')

    _ = xarray.load_dataset(temp_output_file.name)

    temp_output_file.close()


def test_fetch_inputs_all(vito_connection, tmp_path):
    '''Test to fetch all preprocessed inputs from OpenEO
    This test does a full one on one check of the inputs.
    If it fails, something in the data fetching and preprocessing
    has changed. New ref data can be written and pushed, but only
    do this if you are 100% sure that the new data is fully correct.

    Note that changing to a different backend will likely have
    slightly different results and hence a failing test.

    Test currently runs on Terrascope backend and layers.
    '''

    temp_output_file = tmp_path / 'cropclass_generated_inputs.nc'

    input_cube = worldcereal_preprocessed_inputs(
        vito_connection,
        EXTENT,
        START_DATE,
        END_DATE,
        target_crs=3035,
        s1_orbitdirection='DESCENDING',
    )

    input_cube.download(temp_output_file)

    input_data = xarray.load_dataset(temp_output_file)

    # Path to reference output file
    ref_file = basedir / 'testresources' / 'worldcereal_inputs.nc'

    # Uncomment to overwrite reference file
    # input_data.to_netcdf(ref_file)

    # Need 17 variables
    assert len(input_data.variables) == 17

    # Do 1-1 check with ref data
    ref_data = xarray.load_dataset(ref_file)
    xarray.testing.assert_allclose(input_data, ref_data)


def test_fetch_inputs_all_keep_creo(creo_connection):
    '''Test fetching preprocessed inputs from CREO
    backend.
    '''

    logger.info('Starting test test_fetch_inputs_all')

    temp_output_file = tempfile.NamedTemporaryFile()

    collections_options = get_collection_options('creo')

    input_cube = worldcereal_preprocessed_inputs(
        creo_connection,
        EXTENT,
        START_DATE,
        END_DATE,
        target_crs=3035,
        s1_orbitdirection='DESCENDING',
        provider='creo',
        **collections_options
    )

    logger.info('Starting the downloading of the inputs as a synchronous job.')
    input_cube.download(temp_output_file.name, format='NetCDF')

    logger.info('Downloading finished, loading the result file.')

    input_data = xarray.load_dataset(temp_output_file.name)

    # Path to reference output file
    ref_file = basedir / 'testresources' / 'worldcereal_inputs_creo.nc'

    # Uncomment to overwrite reference file
    input_data.to_netcdf(ref_file)

    # Need 16 variables
    assert len(input_data.variables) == 16

    # Do 1-1 check with ref data
    ref_data = xarray.load_dataset(ref_file).drop(
        'crs').to_array(dim='bands')
    input_data = input_data.drop('crs').to_array(dim='bands')

    # Do test in float32 so negative values don't overflow
    # Also test using numpy to be more verbose in case of
    # differences
    np.testing.assert_array_equal(
        ref_data.values.astype(np.float32),
        input_data.values.astype(np.float32))

    temp_output_file.close()


# def test_run_udf_openeo(vito_connection):
#     '''Test full inference run on Terrascope
#     '''

#     provider = 'terrascope'

#     # Get the appropriate job options
#     job_options = get_job_options(provider)

#     # Main croptype generation function
#     clf_results = croptype_map(EXTENT, vito_connection, provider)

#     # Finally, submit the job so the entire workflow will start
#     job = clf_results.execute_batch(
#         title="Cropclass-Classification-Workflow-Terrascope",
#         out_format="GTiff",
#         job_options=job_options)

#     # Get the results
#     results = job.get_results()

#     # Path to reference output file
#     ref_file = basedir / 'resources' / 'cropclass-terrascope-OpenEO_REF.tif'

#     # Loop over the resulting assets and download
#     for asset in results.get_assets():
#         if asset.metadata["type"].startswith("image/tiff"):
#             newfile = str(basedir / "cropclass-terrascope-") + asset.name
#             asset.download(newfile)

#             # Uncomment to overwrite reference file
#             # asset.download(ref_file)

#             # Compare new data to ref data
#             ds_new = xarray.open_dataset(newfile, engine='rasterio')
#             ds_ref = xarray.open_dataset(ref_file, engine='rasterio')

#             xarray.testing.assert_allclose(ds_new, ds_ref)

#     # Path to reference output file from local run
#     local_ref_file = basedir / 'resources' / 'cropclass_local_result.nc'

#     # Get raw crop type values from local ref file
#     local_ref_data = xarray.load_dataset(
#         local_ref_file).to_array().squeeze(drop=True).sel(
#         bands='croptype').values

#     # Get raw crop type values from new openeo run
#     openeo_data = ds_new['band_data'].sel(band=1).values.astype(np.int64)

#     # Do 1-1 check with local ref data
#     np.testing.assert_array_equal(local_ref_data, openeo_data)


# def test_run_udf_openeo_creo(creo_connection):
#     '''Test full inference run on Creodias
#     '''

#     provider = 'creodias'

#     # Get the appropriate job options
#     job_options = get_job_options(provider)

#     # For CREO we need to add METEO data manually
#     with open(os.path.join(basedir, "resources/METEO-E268N194-2021")) as meteo:
#         meteo_json = json.load(meteo)

#     # Main croptype generation function
#     clf_results = croptype_map(EXTENT, creo_connection, provider,
#                                processing_options={'METEO_data': meteo_json})

#     # Finally, submit the job so the entire workflow will start
#     job = clf_results.execute_batch(
#         title="Cropclass-Classification-Workflow-CreoDIAS",
#         out_format="GTiff",
#         job_options=job_options)

#     # Get the results
#     results = job.get_results()

#     # Path to reference output file
#     ref_file = basedir / 'resources' / 'cropclass-creo-OpenEO_REF.tif'

#     # Loop over the resulting assets and download
#     for asset in results.get_assets():
#         if asset.metadata["type"].startswith("image/tiff"):
#             newfile = str(basedir / "cropclass-creo-") + asset.name
#             asset.download(newfile)

#             # Uncomment to overwrite reference file
#             asset.download(ref_file)

#             # Compare new data to ref data
#             ds_new = xarray.open_dataset(newfile, engine='rasterio')
#             ds_ref = xarray.open_dataset(ref_file, engine='rasterio')

#             # Only test the labels
#             np.testing.assert_array_equal(
#                 ds_new['band_data'][0, :, :],
#                 ds_ref['band_data'][0, :, :])

#     # Path to reference output file from local run
#     local_ref_file = basedir / 'resources' / 'cropclass_local_result.nc'

#     # Get raw crop type values from local ref file
#     local_ref_data = xarray.load_dataset(
#         local_ref_file).to_array().squeeze(drop=True).sel(
#         bands='croptype').values

#     # Get raw crop type values from new openeo run
#     openeo_data = ds_new['band_data'].sel(band=1).values.astype(np.int64)

#     # Do 1-1 check with local ref data
#     # NOTE: we cannot do direct assert_array_equal because one
#     # pixel is not equal. Upon investigation it seems that two
#     # fields have two pixels difference on the border, likely because
#     # of all projection steps involved or catalogue differences.
#     # We do not consider it as a problematic difference because .
#     # np.testing.assert_array_equal(local_ref_data, openeo_data)
#     assert np.count_nonzero(local_ref_data != openeo_data) <= 2


def test_optical_mask(vito_connection):
    '''Test whether mask works as expected.
    Here we test the mask_scl_dilation mask
    '''
    logger.info('Starting test test_optical_mask')

    temp_output_file = tempfile.NamedTemporaryFile()

    processing_options = get_processing_options('terrascope')

    input_cube = worldcereal_preprocessed_inputs(
        vito_connection,
        EXTENT,
        START_DATE,
        END_DATE,
        S1_collection=None,
        METEO_collection=None,
        DEM_collection=None,
        WORLDCOVER_collection=None,
        preprocess=False,  # So linear interpolation is disabled
        masking='mask_scl_dilation',
        **processing_options
    )

    logger.info('Starting the downloading of the inputs as a synchronous job.')
    input_cube.download(temp_output_file.name, format='NetCDF')

    logger.info('Downloading finished, loading the result file.')

    input_data = xarray.load_dataset(temp_output_file.name)

    # We should have exactly 3116 masked pixels in case of mask_scl_dilation
    # masking
    assert (input_data['B08'].values == 65535).sum() == 4709


# @pytest.mark.skip
def test_optical_mask_creo(creo_connection):
    '''Test whether mask works as expected.
    Here we test CREO backend
    NOTE: currently skipped until bug is fixed for new baseline!
    '''
    logger.info('Starting CREO test test_optical_mask')

    temp_output_file = tempfile.NamedTemporaryFile()
    print(temp_output_file)

    processing_options = get_processing_options('creodias')
    processing_options['end_month'] = 12
    S2_collection = get_collection_options(
        provider='creodias')['S2_collection']

    input_cube = worldcereal_preprocessed_inputs(
        creo_connection,
        EXTENT,
        START_DATE,
        END_DATE,
        S2_collection=S2_collection,
        S1_collection=None,
        METEO_collection=None,
        DEM_collection=None,
        WORLDCOVER_collection=None,
        preprocess=False,  # So linear interpolation is disabled
        masking='mask_scl_dilation',
        **processing_options
    )

    logger.info('Starting the downloading of the inputs as a synchronous job.')
    input_cube.download(temp_output_file.name, format='NetCDF')

    logger.info('Downloading finished, loading the result file.')

    input_data = xarray.load_dataset(temp_output_file.name)

    # We should have exactly XXX masked pixels in case of mask_scl_dilation
    # masking
    assert (input_data['B08'].values == 65535).sum() == 5510
