from openeo.processes import array_create, if_, is_nodata, power
from openeo.rest.datacube import DataCube

from worldcereal.openeo.masking import scl_mask_erode_dilate
from worldcereal.openeo.compositing import max_ndvi_composite


COMPOSITE_WINDOW = 'month'


def add_S1_bands(connection, S1_collection,
                 other_bands, bbox, start, end,
                 preprocess=True, **processing_options):
    """Method to add S1 bands to datacube

    Args:
        S1_collection (str): name of the S1 collection
        other_bands (DataCube): OpenEO datacube to add bands to

    Available processing_options:
        s1_orbitdirection
        provider
        target_crs
    """
    isCreo = "creo" in processing_options.get("provider", "").lower()
    orbit_direction = processing_options.get('s1_orbitdirection', None)
    composite_window = processing_options.get(
        'composite_window', COMPOSITE_WINDOW)

    # TODO: implement as needed
    # if isCreo:
    #     orbit_direction = catalogue_check_S1(orbit_direction, start, end, bbox)

    if orbit_direction is not None:
        properties = {"sat:orbit_state": lambda orbdir: orbdir == orbit_direction}  # NOQA
    else:
        properties = {}

    # Load collection
    S1bands = connection.load_collection(
        S1_collection,
        bands=['VH', 'VV'],
        spatial_extent=bbox,
        temporal_extent=[start, end],
        properties=properties
    )

    if S1_collection == "SENTINEL1_GRD":
        # compute backscatter if starting from raw GRD,
        # otherwise assume preprocessed backscatter
        S1bands = S1bands.sar_backscatter(
            coefficient='sigma0-ellipsoid',
            local_incidence_angle=False,
            # DO NOT USE MAPZEN
            elevation_model='COPERNICUS_30' if isCreo else None,
            options={"implementation_version": "2",
                     "tile_size": 256, "otb_memory": 1024, "debug": False,
                     "elev_geoid": "/opt/openeo-vito-aux-data/egm96.tif"}
        )
    else:
        pass

    # Resample to the S2 spatial resolution
    target_crs = processing_options.get("target_crs", None)
    if target_crs is not None:
        S1bands = S1bands.resample_spatial(
            projection=target_crs, resolution=10.0)

    if preprocess:

        # Composite to compositing window
        S1bands = S1bands.aggregate_temporal_period(period=composite_window,
                                                    reducer="mean")

        # Linearly interpolate missing values
        S1bands = S1bands.apply_dimension(dimension="t",
                                          process="array_interpolate_linear")

    # Scale to int16
    if isCreo:
        # for CREO, rescaling also replaces nodata introduced by orfeo
        # with a low value
        # https://github.com/Open-EO/openeo-geopyspark-driver/issues/293
        # TODO: check if nodata is correctly handled in Orfeo
        S1bands = S1bands.apply_dimension(
            dimension="bands",
            process=lambda x: array_create(
                [if_(is_nodata(x[0]), 1, power(
                    base=10, p=(10.0 * x[0].log(base=10) + 83.) / 20.)),
                    if_(is_nodata(x[1]), 1, power(
                        base=10, p=(10.0 * x[1].log(base=10) + 83.) / 20.))]))
    else:
        S1bands = S1bands.apply_dimension(
            dimension="bands",
            process=lambda x: array_create(
                [power(base=10, p=(10.0 * x[0].log(base=10) + 83.) / 20.),
                 power(base=10, p=(10.0 * x[1].log(base=10) + 83.) / 20.)]))

    S1bands = S1bands.linear_scale_range(1, 65534, 1, 65534)

    # --------------------------------------------------------------------
    # Merge cubes
    # --------------------------------------------------------------------

    merged_inputs = other_bands.resample_cube_spatial(
        S1bands).merge_cubes(S1bands)

    return merged_inputs


def add_DEM(connection, DEM_collection, other_bands, bbox,
            **processing_options):
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
    target_crs = processing_options.get("target_crs", None)
    if (target_crs is not None):
        dem = dem.resample_spatial(projection=target_crs, resolution=10.0,
                                   method='cubic')

    # collection has timestamps which we need to get rid of
    dem = dem.max_time()

    # --------------------------------------------------------------------
    # Merge cubes
    # --------------------------------------------------------------------

    merged_inputs = other_bands.merge_cubes(dem)

    return merged_inputs


def add_meteo(connection, METEO_collection, other_bands, bbox,
              start, end, target_crs=None, **processing_options):
    # AGERA5
    # TODO: add precipitation with sum compositor

    composite_window = processing_options.get(
        'composite_window', COMPOSITE_WINDOW)

    meteo = connection.load_collection(
        METEO_collection,
        spatial_extent=bbox,
        bands=['temperature-mean'],
        temporal_extent=[start, end]
    )

    if (target_crs is not None):
        meteo = meteo.resample_spatial(projection=target_crs, resolution=10.0)

    # Composite to compositing window
    meteo = meteo.aggregate_temporal_period(period=composite_window,
                                            reducer="mean")

    # Linearly interpolate missing values.
    # Shouldn't exist in this dataset but is good practice to do so
    meteo = meteo.apply_dimension(dimension="t",
                                  process="array_interpolate_linear")

    # Rename band to match Radix model requirements
    meteo = meteo.rename_labels('bands', ['temperature_mean'])

    # --------------------------------------------------------------------
    # Merge cubes
    # or return just meteo
    # --------------------------------------------------------------------
    if other_bands is None:
        return meteo

    merged_inputs = other_bands.merge_cubes(meteo)

    return merged_inputs


def worldcereal_preprocessed_inputs(
        connection, bbox, start: str, end: str,
        S2_collection='TERRASCOPE_S2_TOC_V2',
        S1_collection='SENTINEL1_GRD_SIGMA0',
        DEM_collection='COPERNICUS_30',
        METEO_collection='AGERA5',
        preprocess=True,
        masking='mask_scl_dilation',
        **processing_options) -> DataCube:
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

    Returns:
        DataCube: OpenEO DataCube wich the requested inputs
    """

    composite_window = processing_options.get('composite_window',
                                              COMPOSITE_WINDOW)

    # --------------------------------------------------------------------
    # Optical data
    # --------------------------------------------------------------------

    S2_bands = ["B02", "B03", "B04", "B05",
                "B06", "B07", "B08", "B11",
                "B12"]
    if masking not in ['satio', 'mask_scl_dilation', None]:
        raise ValueError(f'Unknown masking option `{masking}`')
    if masking in ['mask_scl_dilation']:
        # Need SCL band to mask
        S2_bands.append("SCL")
    bands = connection.load_collection(
        S2_collection,
        bands=S2_bands,
        spatial_extent=bbox,
        temporal_extent=[start, end],
        max_cloud_cover=95
    )

    # TODO: implement as needed
    # S2URL creo only accepts request in EPSG:4326
    # isCreo = "creo" in processing_options.get("provider", "").lower()
    # if isCreo:
    #     catalogue_check_S2(start, end, bbox)

    target_crs = processing_options.get("target_crs", None)
    if (target_crs is not None):
        bands = bands.resample_spatial(projection=target_crs, resolution=10.0)

    # NOTE: For now we mask again snow/ice because clouds
    # are sometimes marked as SCL value 11!
    if masking == 'mask_scl_dilation':
        # TODO: double check cloud masking parameters
        # https://github.com/Open-EO/openeo-geotrellis-extensions/blob/develop/geotrellis-common/src/main/scala/org/openeo/geotrelliscommon/CloudFilterStrategy.scala#L54  # NOQA
        bands = bands.process(
            "mask_scl_dilation",
            data=bands,
            scl_band_name="SCL",
            kernel1_size=17, kernel2_size=77,
            mask1_values=[2, 4, 5, 6, 7],
            mask2_values=[3, 8, 9, 10, 11],
            erosion_kernel_size=3).filter_bands(
            bands.metadata.band_names[:-1])
    elif masking == 'satio':
        # Apply satio-based mask
        mask = scl_mask_erode_dilate(
            connection,
            bbox,
            scl_layer_band=S2_collection + ':SCL',
            target_crs=target_crs).resample_cube_spatial(bands)
        bands = bands.mask(mask)

    if preprocess:
        # Composite to compositing window
        # bands = bands.aggregate_temporal_period(period=composite_window,
        #                                         reducer="median")
        bands = max_ndvi_composite(bands, composite_window=composite_window)

        # TODO: if we would disable it here, nodata values
        # will be 65535 and we need to cope with that later
        # Linearly interpolate missing values
        bands = bands.apply_dimension(dimension="t",
                                      process="array_interpolate_linear")

    # Force UINT16 to avoid overflow issue with S2 data
    bands = bands.linear_scale_range(0, 65534, 0, 65534)

    # --------------------------------------------------------------------
    # AGERA5 Meteo data
    # --------------------------------------------------------------------
    if METEO_collection is not None:
        bands = add_meteo(connection, METEO_collection,
                          bands, bbox, start, end,
                          target_crs=target_crs,
                          composite_window=composite_window)

    # --------------------------------------------------------------------
    # SAR data
    # --------------------------------------------------------------------
    if S1_collection is not None:
        bands = add_S1_bands(connection, S1_collection,
                             bands, bbox, start, end,
                             composite_window=composite_window,
                             **processing_options)

    bands = bands.filter_temporal(start, end)

    # --------------------------------------------------------------------
    # DEM data
    # --------------------------------------------------------------------
    if DEM_collection is not None:
        bands = add_DEM(connection, DEM_collection,
                        bands, bbox, **processing_options)

    # forcing 16bit
    bands = bands.linear_scale_range(0, 65534, 0, 65534)

    return bands


def worldcereal_raw_inputs(*args, **kwargs):
    return worldcereal_preprocessed_inputs(
        *args, **kwargs, preprocess=False)
