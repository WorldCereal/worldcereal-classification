from pathlib import Path
from typing import List, Optional

from openeo import UDF, Connection, DataCube
from openeo_gfmap import (
    Backend,
    BackendContext,
    BoundingBoxExtent,
    FetchType,
    SpatialContext,
    TemporalContext,
)
from openeo_gfmap.fetching.generic import build_generic_extractor
from openeo_gfmap.fetching.meteo import build_meteo_extractor
from openeo_gfmap.fetching.s1 import build_sentinel1_grd_extractor
from openeo_gfmap.fetching.s2 import build_sentinel2_l2a_extractor
from openeo_gfmap.preprocessing.compositing import mean_compositing, median_compositing
from openeo_gfmap.preprocessing.sar import compress_backscatter_uint16
from openeo_gfmap.utils.catalogue import UncoveredS1Exception, select_S1_orbitstate

COMPOSITE_WINDOW = "month"


def raw_datacube_S2(
    connection: Connection,
    backend_context: BackendContext,
    spatial_extent: SpatialContext,
    temporal_extent: TemporalContext,
    bands: List[str],
    fetch_type: FetchType,
    filter_tile: Optional[str] = None,
    additional_masks: bool = True,
    apply_mask: bool = False,
) -> DataCube:
    """Extract Sentinel-2 datacube from OpenEO using GFMAP routines.
    Raw data is extracted with no cloud masking applied by default (can be
    enabled by setting `apply_mask=True`). In additional to the raw band values
    a cloud-mask computed from the dilation of the SCL layer, as well as a
    rank mask from the BAP compositing are added.

    Parameters
    ----------
    connection : Connection
        OpenEO connection instance.
    backend_context : BackendContext
        GFMAP Backend context to use for extraction.
    spatial_extent : SpatialContext
        Spatial context to extract data from, can be a GFMAP BoundingBoxExtent,
        a GeoJSON dict or an URL to a publicly accessible GeoParquet file.
    temporal_extent : TemporalContext
        Temporal context to extract data from.
    bands : List[str]
        List of Sentinel-2 bands to extract.
    fetch_type : FetchType
        GFMAP Fetch type to use for extraction.
    filter_tile : Optional[str], optional
        Filter by tile ID, by default disabled. This forces the process to only
        one tile ID from the Sentinel-2 collection.
    apply_mask : bool, optional
        Apply cloud masking, by default False. Can be enabled for high
        optimization of memory usage.
    """
    # Extract the SCL collection only
    scl_cube_properties = {"eo:cloud_cover": lambda val: val <= 95.0}
    if filter_tile:
        scl_cube_properties["tileId"] = lambda val: val == filter_tile

    # Create the job to extract S2
    extraction_parameters = {
        "target_resolution": None,  # Disable target resolution
        "load_collection": {
            "eo:cloud_cover": lambda val: val <= 95.0,
        },
    }

    scl_cube = connection.load_collection(
        collection_id="SENTINEL2_L2A",
        bands=["SCL"],
        temporal_extent=[temporal_extent.start_date, temporal_extent.end_date],
        spatial_extent=dict(spatial_extent) if fetch_type == FetchType.TILE else None,
        properties=scl_cube_properties,
    )

    # Resample to 10m resolution for the SCL layer
    scl_cube = scl_cube.resample_spatial(10)

    # Compute the SCL dilation mask
    scl_dilated_mask = scl_cube.process(
        "to_scl_dilation_mask",
        data=scl_cube,
        scl_band_name="SCL",
        kernel1_size=17,  # 17px dilation on a 10m layer
        kernel2_size=77,  # 77px dilation on a 10m layer
        mask1_values=[2, 4, 5, 6, 7],
        mask2_values=[3, 8, 9, 10, 11],
        erosion_kernel_size=3,
    ).rename_labels("bands", ["S2-L2A-SCL_DILATED_MASK"])

    if additional_masks:
        # Compute the distance to cloud and add it to the cube
        distance_to_cloud = scl_cube.apply_neighborhood(
            process=UDF.from_file(Path(__file__).parent / "udf_distance_to_cloud.py"),
            size=[
                {"dimension": "x", "unit": "px", "value": 256},
                {"dimension": "y", "unit": "px", "value": 256},
                {"dimension": "t", "unit": "null", "value": "P1D"},
            ],
            overlap=[
                {"dimension": "x", "unit": "px", "value": 16},
                {"dimension": "y", "unit": "px", "value": 16},
            ],
        ).rename_labels("bands", ["S2-L2A-DISTANCE-TO-CLOUD"])

        additional_masks = scl_dilated_mask.merge_cubes(distance_to_cloud)

        # Try filtering using the geometry
        if fetch_type == FetchType.TILE:
            additional_masks = additional_masks.filter_spatial(
                spatial_extent.to_geojson()
            )

        extraction_parameters["pre_merge"] = additional_masks

    if filter_tile:
        extraction_parameters["load_collection"]["tileId"] = (
            lambda val: val == filter_tile
        )
    if apply_mask:
        extraction_parameters["pre_mask"] = scl_dilated_mask

    extractor = build_sentinel2_l2a_extractor(
        backend_context,
        bands=bands,
        fetch_type=fetch_type,
        **extraction_parameters,
    )

    return extractor.get_cube(connection, spatial_extent, temporal_extent)


def raw_datacube_S1(
    connection: Connection,
    backend_context: BackendContext,
    spatial_extent: SpatialContext,
    temporal_extent: TemporalContext,
    bands: List[str],
    fetch_type: FetchType,
    target_resolution: float = 20.0,
    orbit_direction: Optional[str] = None,
) -> DataCube:
    """Extract Sentinel-1 datacube from OpenEO using GFMAP routines.

    Parameters
    ----------
    connection : Connection
        OpenEO connection instance.
    backend_context : BackendContext
        GFMAP Backend context to use for extraction.
    spatial_extent : SpatialContext
        Spatial context to extract data from, can be a GFMAP BoundingBoxExtent,
        a GeoJSON dict or an URL to a publicly accessible GeoParquet file.
    temporal_extent : TemporalContext
        Temporal context to extract data from.
    bands : List[str]
        List of Sentinel-1 bands to extract.
    fetch_type : FetchType
        GFMAP Fetch type to use for extraction.
    target_resolution : float, optional
        Target resolution to resample the data to, by default 20.0.
    orbit_direction : Optional[str], optional
        Orbit direction to filter the data, by default None. If None and the
        backend is in CDSE, then querries the catalogue for the best orbit
        direction to use. In the case querrying is unavailable or fails, then
        uses "ASCENDING" as a last resort.
    """
    extractor_parameters = {
        "target_resolution": target_resolution,
    }
    if orbit_direction is None and backend_context.backend in [
        Backend.CDSE,
        Backend.CDSE_STAGING,
        Backend.FED,
    ]:
        try:
            orbit_direction = select_S1_orbitstate(
                backend_context, spatial_extent, temporal_extent
            )
            print(
                f"Selected orbit direction: {orbit_direction} from max "
                "accumulated area overlap between bounds and products."
            )
        except UncoveredS1Exception as exc:
            orbit_direction = "ASCENDING"
            print(
                f"Could not find any Sentinel-1 data for the given spatio-temporal context. "
                f"Using ASCENDING orbit direction as a last resort. Error: {exc}"
            )

    if orbit_direction is not None:
        extractor_parameters["load_collection"] = {
            "sat:orbit_state": lambda orbit: orbit == orbit_direction
        }

    extractor = build_sentinel1_grd_extractor(
        backend_context, bands=bands, fetch_type=fetch_type, **extractor_parameters
    )

    return extractor.get_cube(connection, spatial_extent, temporal_extent)


def raw_datacube_DEM(
    connection: Connection,
    backend_context: BackendContext,
    spatial_extent: SpatialContext,
    fetch_type: FetchType,
) -> DataCube:
    extractor = build_generic_extractor(
        backend_context=backend_context,
        bands=["COP-DEM"],
        fetch_type=fetch_type,
        collection_name="COPERNICUS_30",
    )
    return extractor.get_cube(connection, spatial_extent, None)


def raw_datacube_METEO(
    connection: Connection,
    backend_context: BackendContext,
    spatial_extent: SpatialContext,
    temporal_extent: TemporalContext,
    fetch_type: FetchType,
) -> DataCube:
    extractor = build_meteo_extractor(
        backend_context=backend_context,
        bands=["AGERA5-TMEAN", "AGERA5-PRECIP"],
        fetch_type=fetch_type,
    )
    return extractor.get_cube(connection, spatial_extent, temporal_extent)


def precomposited_datacube_METEO(
    connection: Connection,
    spatial_extent: SpatialContext,
    temporal_extent: TemporalContext,
) -> DataCube:
    """Extract the precipitation and temperature AGERA5 data from a
    pre-composited and pre-processed collection. The data is stored in the
    CloudFerro S3 stoage, allowing faster access and processing from the CDSE
    backend.

    Limitations:
        - Only monthly composited data is available.
        - Only two bands are available: precipitation-flux and temperature-mean.
        - This function do not support fetching points or polygons, but only
          tiles.
    """
    temporal_extent = [temporal_extent.start_date, temporal_extent.end_date]
    spatial_extent = dict(spatial_extent)

    # Monthly composited METEO data
    cube = connection.load_stac(
        "https://s3.waw3-1.cloudferro.com/swift/v1/agera/stac/collection.json",
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=["precipitation-flux", "temperature-mean"],
    )
    cube = cube.rename_labels(
        dimension="bands", target=["AGERA5-PRECIP", "AGERA5-TMEAN"]
    )

    return cube


def worldcereal_preprocessed_inputs_gfmap(
    connection: Connection,
    backend_context: BackendContext,
    spatial_extent: BoundingBoxExtent,
    temporal_extent: TemporalContext,
) -> DataCube:
    # Extraction of S2 from GFMAP
    s2_data = raw_datacube_S2(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=[
            "S2-L2A-B02",
            "S2-L2A-B03",
            "S2-L2A-B04",
            "S2-L2A-B05",
            "S2-L2A-B06",
            "S2-L2A-B07",
            "S2-L2A-B08",
            "S2-L2A-B11",
            "S2-L2A-B12",
        ],
        fetch_type=FetchType.TILE,
        filter_tile=False,
        additional_masks=False,
        apply_mask=True,
    )

    s2_data = median_compositing(s2_data, period="month")

    # Cast to uint16
    s2_data = s2_data.linear_scale_range(0, 65534, 0, 65534)

    # Decide on the orbit direction from the maximum overlapping area of
    # available products.

    # Extraction of the S1 data
    s1_data = raw_datacube_S1(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=[
            "S1-SIGMA0-VH",
            "S1-SIGMA0-VV",
        ],
        fetch_type=FetchType.TILE,
        target_resolution=10.0,  # Compute the backscatter at 20m resolution, then upsample nearest neighbor when merging cubes
        orbit_direction=None,  # Make the querry on the catalogue for the best orbit
    )

    s1_data = mean_compositing(s1_data, period="month")
    s1_data = compress_backscatter_uint16(backend_context, s1_data)

    dem_data = raw_datacube_DEM(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=spatial_extent,
        fetch_type=FetchType.TILE,
    )

    dem_data = dem_data.linear_scale_range(0, 65534, 0, 65534)

    meteo_data = precomposited_datacube_METEO(
        connection=connection,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
    )

    data = s2_data.merge_cubes(s1_data)
    data = data.merge_cubes(dem_data)
    data = data.merge_cubes(meteo_data)

    return data
