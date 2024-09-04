import numpy as np
import xarray as xr
from openeo.udf import XarrayDataCube
from scipy.ndimage import distance_transform_cdt
from skimage.morphology import binary_erosion, footprints


def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    cube_array: xr.DataArray = cube.get_array()
    cube_array = cube_array.transpose("bands", "y", "x")

    clouds = np.logical_or(
        np.logical_and(cube_array < 11, cube_array >= 8), cube_array == 3
    ).isel(
        bands=0
    )  # type: ignore

    # Calculate the Distance To Cloud score
    # Erode
    er = footprints.disk(3)

    # Define a function to apply binary erosion
    def erode(image, selem):
        return ~binary_erosion(image, selem)

    # Use apply_ufunc to apply the erosion operation
    eroded = xr.apply_ufunc(
        erode,  # function to apply
        clouds,  # input DataArray
        input_core_dims=[["y", "x"]],  # dimensions over which to apply function
        output_core_dims=[["y", "x"]],  # dimensions of the output
        vectorize=True,  # vectorize the function over non-core dimensions
        dask="parallelized",  # enable dask parallelization
        output_dtypes=[np.int32],  # data type of the output
        kwargs={"selem": er},  # additional keyword arguments to pass to erode
    )

    # Distance to cloud in manhattan distance measure
    distance = xr.apply_ufunc(
        distance_transform_cdt,
        eroded,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y", "x"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32],
    )

    distance_da = xr.DataArray(
        distance,
        coords={
            "y": cube_array.coords["y"],
            "x": cube_array.coords["x"],
        },
        dims=["y", "x"],
    )

    distance_da = distance_da.expand_dims(
        dim={
            "bands": cube_array.coords["bands"],
        },
    )

    distance_da = distance_da.transpose("bands", "y", "x")

    return XarrayDataCube(distance_da)
