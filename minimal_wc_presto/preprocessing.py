from openeo.processes import array_create, if_, is_nodata, power
from openeo.rest.datacube import DataCube

COMPOSITE_WINDOW = "month"


def get_S1_bands(
    connection,
    S1_collection,
    bbox,
    start,
    end,
    other_bands=None,
    preprocess=True,
    **processing_options,
):
    """Method to add S1 bands to datacube

    Args:
        S1_collection (str): name of the S1 collection
        other_bands (DataCube): OpenEO datacube to add bands to

    Available processing_options:
        s1_orbitdirection
        provider
        target_epsg
    """
    isCreo = "creo" in processing_options.get("provider", "").lower()
    orbit_direction = processing_options.get("s1_orbitdirection", None)
    composite_window = processing_options.get("composite_window", COMPOSITE_WINDOW)

    # TODO: implement as needed
    # if isCreo:
    #     orbit_direction = catalogue_check_S1(orbit_direction, start, end, bbox)

    if orbit_direction is not None:
        properties = {
            "sat:orbit_state": lambda orbdir: orbdir == orbit_direction
        }  # NOQA
    else:
        properties = {}

    # Load collection
    S1bands = connection.load_collection(
        S1_collection,
        bands=["VH", "VV"],
        spatial_extent=bbox,
        temporal_extent=[start, end],
        properties=properties,
    )

    if S1_collection == "SENTINEL1_GRD":
        # compute backscatter if starting from raw GRD,
        # otherwise assume preprocessed backscatter
        S1bands = S1bands.sar_backscatter(
            coefficient="sigma0-ellipsoid",
            local_incidence_angle=False,
            # DO NOT USE MAPZEN
            elevation_model="COPERNICUS_30" if isCreo else None,
            options={
                "implementation_version": "2",
                "tile_size": 256,
                "otb_memory": 1024,
                "debug": False,
                "elev_geoid": "/opt/openeo-vito-aux-data/egm96.tif",
            },
        )
    else:
        pass

    # Resample to the S2 spatial resolution
    target_epsg = processing_options.get("target_epsg", None)
    if target_epsg is not None:
        S1bands = S1bands.resample_spatial(projection=target_epsg, resolution=10.0)

    if preprocess:

        # Composite to compositing window
        S1bands = S1bands.aggregate_temporal_period(
            period=composite_window, reducer="mean"
        )

        # # Linearly interpolate missing values
        # Assume Presto handles nodata natively
        # S1bands = S1bands.apply_dimension(
        #     dimension="t", process="array_interpolate_linear"
        # )

    # Scale to int16
    if isCreo:
        # for CREO, rescaling also replaces nodata introduced by orfeo
        # with a low value
        # https://github.com/Open-EO/openeo-geopyspark-driver/issues/293
        # TODO: check if nodata is correctly handled in Orfeo
        S1bands = S1bands.apply_dimension(
            dimension="bands",
            process=lambda x: array_create(
                [
                    if_(
                        is_nodata(x[0]),
                        1,
                        power(base=10, p=(10.0 * x[0].log(base=10) + 83.0) / 20.0),
                    ),
                    if_(
                        is_nodata(x[1]),
                        1,
                        power(base=10, p=(10.0 * x[1].log(base=10) + 83.0) / 20.0),
                    ),
                ]
            ),
        )
    else:
        S1bands = S1bands.apply_dimension(
            dimension="bands",
            process=lambda x: array_create(
                [
                    power(base=10, p=(10.0 * x[0].log(base=10) + 83.0) / 20.0),
                    power(base=10, p=(10.0 * x[1].log(base=10) + 83.0) / 20.0),
                ]
            ),
        )

    S1bands = S1bands.linear_scale_range(1, 65534, 1, 65534)

    # --------------------------------------------------------------------
    # Merge cubes
    # --------------------------------------------------------------------
    if other_bands is None:
        return S1bands
    else:
        merged_inputs = other_bands.resample_cube_spatial(S1bands).merge_cubes(S1bands)
        return merged_inputs


def get_S2_bands(
    connection,
    S2_collection,
    bbox,
    start,
    end,
    masking,
    preprocess=True,
    other_bands=None,
    target_epsg=None,
    **processing_options,
):
    """Method to get S2 bands and optionally merge with other bands

    Args:
        S2_collection (str): name of the S2 collection
        other_bands (DataCube): OpenEO datacube to add bands to

    Available processing_options:
        s1_orbitdirection
        provider
        target_epsg
    """

    composite_window = processing_options.get("composite_window", COMPOSITE_WINDOW)

    S2_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
    if masking not in ["satio", "mask_scl_dilation", None]:
        raise ValueError(f"Unknown masking option `{masking}`")
    if masking in ["mask_scl_dilation"]:
        # Need SCL band to mask
        S2_bands.append("SCL")
    bands = connection.load_collection(
        S2_collection,
        bands=S2_bands,
        spatial_extent=bbox,
        temporal_extent=[start, end],
        max_cloud_cover=95,
    )

    # TODO: implement as needed
    # S2URL creo only accepts request in EPSG:4326
    # isCreo = "creo" in processing_options.get("provider", "").lower()
    # if isCreo:
    #     catalogue_check_S2(start, end, bbox)

    # NOTE: For now we mask again snow/ice because clouds
    # are sometimes marked as SCL value 11!
    if masking == "mask_scl_dilation":
        # TODO: double check cloud masking parameters
        # https://github.com/Open-EO/openeo-geotrellis-extensions/blob/develop/geotrellis-common/src/main/scala/org/openeo/geotrelliscommon/CloudFilterStrategy.scala#L54  # NOQA
        bands = bands.process(
            "mask_scl_dilation",
            data=bands,
            scl_band_name="SCL",
            kernel1_size=17,
            kernel2_size=77,
            mask1_values=[2, 4, 5, 6, 7],
            mask2_values=[3, 8, 9, 10, 11],
            erosion_kernel_size=3,
        ).filter_bands(bands.metadata.band_names[:-1])
    #elif masking == "satio":
        # Apply satio-based mask
    #    mask = scl_mask_erode_dilate(
    #        connection,
    #        bbox,
    #        scl_layer_band=S2_collection + ":SCL",
    #        target_epsg=target_epsg,
    #    ).resample_cube_spatial(bands)
    #    bands = bands.mask(mask)

    if preprocess:
        # Composite to compositing window
        bands = bands.aggregate_temporal_period(
            period=composite_window, reducer="median"
        )
        # bands = max_ndvi_composite(bands, composite_window=composite_window)

        # TODO: if we would disable it here, nodata values
        # will be 65535 and we need to cope with that later
        # Linearly interpolate missing values
        # bands = bands.apply_dimension(dimension="t", process="array_interpolate_linear")

    # Force UINT16 to avoid overflow issue with S2 data
    bands = bands.linear_scale_range(0, 65534, 0, 65534)

    # --------------------------------------------------------------------
    # Merge cubes
    # --------------------------------------------------------------------
    if other_bands is None:
        return bands
    else:
        merged_inputs = other_bands.resample_cube_spatial(bands).merge_cubes(bands)
        return merged_inputs


def get_DEM(connection, DEM_collection, bbox, other_bands=None, **processing_options):
    """Method to add DEM to datacube

    Args:
        connection (_type_): _description_
        DEM_collection (str): Name of DEM collection
        other_bands (DataCube): DataCube to merge DEM into
        bbox (_type_): _description_

    Returns:
        DataCube: merged datacube
    """

    dem = connection.load_collection(
        DEM_collection,
        spatial_extent=bbox,
    )

    # Resample to the S2 spatial resolution
    target_epsg = processing_options.get("target_epsg", None)
    if target_epsg is not None:
        dem = dem.resample_spatial(
            projection=target_epsg, resolution=10.0, method="cubic"
        )

    # collection has timestamps which we need to get rid of
    dem = dem.max_time()

    # --------------------------------------------------------------------
    # Merge cubes
    # --------------------------------------------------------------------
    if other_bands is None:
        return dem
    else:
        merged_inputs = other_bands.merge_cubes(dem)
        return merged_inputs


def get_meteo(
    connection,
    METEO_collection,
    bbox,
    start,
    end,
    other_bands=None,
    target_epsg=None,
    **processing_options,
):
    # AGERA5
    composite_window = processing_options.get("composite_window", COMPOSITE_WINDOW)

    meteo = connection.load_collection(
        METEO_collection,
        spatial_extent=bbox,
        bands=["temperature-mean", "precipitation-flux"],
        temporal_extent=[start, end],
    )

    if target_epsg is not None:
        meteo = meteo.resample_spatial(
            projection=target_epsg, resolution=10.0, method="bilinear"
        )

    # Composite to desired window. we want to aggregate data with
    # different reducers. sum for precipitation within a month and
    # mean for the temperature
    meteo_temp = meteo.filter_bands(bands=["temperature-mean"])
    meteo_temp = meteo_temp.aggregate_temporal_period(
        period=composite_window, reducer="mean"
    )
    meteo_temp = meteo_temp.apply_dimension(
        dimension="t", process="array_interpolate_linear"
    )

    meteo_prec = meteo.filter_bands(bands=["precipitation-flux"])
    meteo_prec = meteo_prec.aggregate_temporal_period(
        period=composite_window, reducer="sum"
    )
    meteo_prec = meteo_prec.apply_dimension(
        dimension="t", process="array_interpolate_linear"
    )

    meteo = meteo_temp.merge_cubes(meteo_prec)

    # --------------------------------------------------------------------
    # Merge cubes
    # or return just meteo
    # --------------------------------------------------------------------
    if other_bands is None:
        return meteo
    else:
        merged_inputs = other_bands.merge_cubes(meteo)
        return merged_inputs


def add_worldcereral_labels(connection, bbox, other_bands):
    """
    ['ESA_WORLDCEREAL_ACTIVECROPLAND',
    'ESA_WORLDCEREAL_IRRIGATION',
    'ESA_WORLDCEREAL_TEMPORARYCROPS',
    'ESA_WORLDCEREAL_WINTERCEREALS',
    'ESA_WORLDCEREAL_MAIZE',
    'ESA_WORLDCEREAL_SPRINGCEREALS']
    """

    temporal = ("2020-09-01T00:00:00Z", "2021-12-31T00:00:00Z")

    # Get temporary crops layer
    temporarycrops = (
        connection.load_collection(
            "ESA_WORLDCEREAL_TEMPORARYCROPS",
            temporal_extent=temporal,
            spatial_extent=bbox,
            bands=["CLASSIFICATION"],
        )
        .rename_labels("bands", ["worldcereal_cropland"])
        .max_time()
    )
    temporarycrops = temporarycrops.resample_cube_spatial(other_bands, method="near")
    other_bands = other_bands.merge_cubes(temporarycrops)

    # Get maize layer
    maize = (
        connection.load_collection(
            "ESA_WORLDCEREAL_MAIZE",
            temporal_extent=temporal,
            spatial_extent=bbox,
            bands=["CLASSIFICATION"],
        )
        .rename_labels("bands", ["worldcereal_maize"])
        .max_time()
    )
    maize = maize.resample_cube_spatial(other_bands, method="near")
    other_bands = other_bands.merge_cubes(maize)

    # Get wintercereals layer
    wintercereals = (
        connection.load_collection(
            "ESA_WORLDCEREAL_WINTERCEREALS",
            temporal_extent=temporal,
            spatial_extent=bbox,
            bands=["CLASSIFICATION"],
        )
        .rename_labels("bands", ["worldcereal_wintercereals"])
        .max_time()
    )
    wintercereals = wintercereals.resample_cube_spatial(other_bands, method="near")
    other_bands = other_bands.merge_cubes(wintercereals)

    # # Get springcereals layer
    # springcereals = (
    #     connection.load_collection(
    #         "ESA_WORLDCEREAL_SPRINGCEREALS",
    #         temporal_extent=temporal,
    #         spatial_extent=bbox,
    #         bands=["CLASSIFICATION"],
    #     )
    #     .rename_labels("bands", ["worldcereal_springcereals"])
    #     .max_time()
    # )
    # springcereals = springcereals.resample_cube_spatial(other_bands, method="near")
    # other_bands = other_bands.merge_cubes(springcereals)

    return other_bands


def worldcereal_preprocessed_inputs(
    connection,
    bbox,
    start: str,
    end: str,
    S2_collection="SENTINEL2_L2A",
    S1_collection="SENTINEL1_GRD",
    DEM_collection="COPERNICUS_30",
    METEO_collection="AGERA5",
    preprocess=True,
    masking="mask_scl_dilation",
    worldcereal_labels=False,
    **processing_options,
) -> DataCube:
    """Main method to get preprocessed inputs from OpenEO for
    downstream crop type mapping.

    Args:
        connection: OpenEO connection instance
        bbox (_type_): _description_
        start (str): Start date for requested input data (yyyy-mm-dd)
        end (str): Start date for requested input data (yyyy-mm-dd)
        S2_collection (str, optional): Collection name for S2 data.
                        Defaults to
                        'TERRASCOPE_S2_TOC_V2'.
        S1_collection (str, optional): Collection name for S1 data.
                        Defaults to
                        'SENTINEL1_GRD'.
        DEM_collection (str, optional): Collection name for DEM data.
                        Defaults to
                        'COPERNICUS_30'.
        METEO_collection (str, optional): Collection name for
                        meteo data. Defaults to 'AGERA5'.
        preprocess (bool, optional): Apply compositing and interpolation.
                        Defaults to True.
        masking (str, optional): Masking method to be applied.
                                One of ['satio', 'mask_scl_dilation', None]
                                Defaults to 'mask_scl_dilation'.
        worldcereal_labels (bool, optional): If True, worldcereal 2021 labels
                                will be added to the datacube. Defaults to False.

    Returns:
        DataCube: OpenEO DataCube wich the requested inputs
    """

    bands = None

    # --------------------------------------------------------------------
    # Optical data
    # --------------------------------------------------------------------

    if S2_collection is not None:
        bands = get_S2_bands(
            connection,
            S2_collection,
            bbox,
            start,
            end,
            masking,
            preprocess=preprocess,
            **processing_options,
        )

    # --------------------------------------------------------------------
    # AGERA5 Meteo data
    # --------------------------------------------------------------------
    if METEO_collection is not None:
        bands = get_meteo(
            connection,
            METEO_collection,
            bbox,
            start,
            end,
            other_bands=bands,
            **processing_options,
        )

    # --------------------------------------------------------------------
    # SAR data
    # --------------------------------------------------------------------
    if S1_collection is not None:
        bands = get_S1_bands(
            connection,
            S1_collection,
            bbox,
            start,
            end,
            other_bands=bands,
            **processing_options,
        )

    bands = bands.filter_temporal(start, end)

    # --------------------------------------------------------------------
    # DEM data
    # --------------------------------------------------------------------
    if DEM_collection is not None:
        bands = get_DEM(connection, DEM_collection, bbox, bands, **processing_options)

    # --------------------------------------------------------------------
    # Worldcereal labels
    # --------------------------------------------------------------------
    if worldcereal_labels:
        bands = add_worldcereral_labels(connection, bbox, bands)

    # forcing 16bit
    bands = bands.linear_scale_range(0, 65534, 0, 65534)

    return bands


def worldcereal_raw_inputs(*args, **kwargs):
    return worldcereal_preprocessed_inputs(*args, **kwargs, preprocess=False)
