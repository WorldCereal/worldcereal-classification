def test_inference_catboost_presto():
    # Load the result and ground truth
    ds = xr.open_dataset("./data/belgium_good_2020-12-01_2021-11-30.nc", engine='netcdf4')

    # Because we downloaded the data, we need to resolve
    # an issue with the CRS which has become a band. Let's get rid of it
    arr = ds.drop('crs').to_array(dim='bands')

    # Make an OpenEO datacube of this array
    udf_cube = XarrayDataCube(arr)
    result_cube = apply_datacube(udf_cube)

    # Save the result to NetCDF
    result_cube.array.to_netcdf("./data/test_result.nc")
    results = result_cube.array.values.squeeze()

    # to a numpy array
    gt_dataset = xr.open_dataset("./data/worldcereal_result.nc", engine='netcdf4')
    data_variable = gt_dataset['__xarray_dataarray_variable__']
    gt = data_variable.values[0]
    assert np.array_equal(results, gt)