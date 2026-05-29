import openeo
from openeo import UDF

# ======================================================
# CONFIG
# ======================================================
SPATIAL_EXTENT = {
    "west": 4.00,
    "south": 51.08,
    "east": 4.05,
    "north": 51.10
}

TEMPORAL_EXTENT = ["2022-03-01", "2023-03-31"]

CATBOOST_URL = (
    "https://drive.google.com/uc?export=download&id=1YDjYlXF6ulS_7ZNrld_gDou4U7thv7tM"
)

# ======================================================
# CONNECT
# ======================================================
connection = openeo.connect(
    "openeo.dataspace.copernicus.eu"
).authenticate_oidc()

# ======================================================
# SENTINEL-2
# ======================================================
s2 = connection.load_collection(
    "SENTINEL2_L2A",
    spatial_extent=SPATIAL_EXTENT,
    temporal_extent=TEMPORAL_EXTENT,
    bands=[
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
        "SCL"
    ]
)

# Dekadal composite
s2 = s2.aggregate_temporal_period(
    period="dekad",
    reducer="median"
)

# Scale optical data
s2 = s2.linear_scale_range(
    0, 65534,
    0, 65534
)

# ======================================================
# SENTINEL-1
# ======================================================
s1 = connection.load_collection(
    "SENTINEL1_GRD",
    spatial_extent=SPATIAL_EXTENT,
    temporal_extent=TEMPORAL_EXTENT,
    bands=["VV", "VH"]
)

# Dekadal composite
s1 = s1.aggregate_temporal_period(
    period="dekad",
    reducer="median"
)

# Match S2 grid
s1 = s1.resample_cube_spatial(s2)

# ======================================================
# DEM
# ======================================================
dem = connection.load_collection(
    "COPERNICUS_30",
    spatial_extent=SPATIAL_EXTENT,
    bands=["DEM"]
)

# Remove temporal dimension
dem = dem.min_time()

# Rename elevation band
dem = dem.rename_labels(
    dimension="bands",
    target=["elevation"]
)

# ======================================================
# SLOPE (PRECOMPUTED STAC COLLECTION)
# ======================================================
slope = connection.load_stac(
    "https://stac.openeo.vito.be/collections/COPERNICUS30_DEM_SLOPE",
    spatial_extent=SPATIAL_EXTENT,
    bands=["Slope"]
)

# Client compatibility fix
if "t" not in slope.metadata.dimension_names():
    slope.metadata = slope.metadata.add_dimension(
        "t",
        "2020-01-01",
        "temporal"
    )

# Remove temporal dimension
slope = slope.min_time()

# Rename slope band
slope = slope.rename_labels(
    dimension="bands",
    target=["slope"]
)

# ======================================================
# TERRAIN
# ======================================================
terrain = slope.merge_cubes(dem)

# Match S2 grid
terrain = terrain.resample_cube_spatial(
    s2,
    method="bilinear"
)

# Scale terrain features
terrain = terrain.linear_scale_range(
    0, 65534,
    0, 65534
)

# ======================================================
# METEO
# ======================================================
meteo = connection.load_stac(
    url="https://stac.openeo.vito.be/collections/agera5_dekadal_composite",
    spatial_extent=SPATIAL_EXTENT,
    temporal_extent=TEMPORAL_EXTENT,
    bands=["temperature-mean", "precipitation-flux"
    ]
)

meteo = meteo.rename_labels(
    dimension="bands",
    target=[
        "AGERA5-TMEAN",
        "AGERA5-PRECIP"
    ]
)

# Match S2 grid
meteo = meteo.resample_cube_spatial(
    s2,
    method="bilinear"
)

# ======================================================
# FINAL MERGE
# ======================================================
cube = s2.merge_cubes(s1)
cube = cube.merge_cubes(terrain)
cube = cube.merge_cubes(meteo)

print("Band names:")
print(cube.metadata.band_names)

print("Inference cube ready")

# ======================================================
# UDF
# ======================================================
udf = UDF.from_file("inference_alt.py")

processed_cube = cube.apply_dimension(
    process=udf,
    dimension="t",
    context={
        "classifier_url": str(CATBOOST_URL),
        "EPSG": int(32631),
        "ignore_dependencies": True
    }
)

# processed_cube = cube.apply(
#     process=udf,
#     context={
#         "classifier_url": str(CATBOOST_URL),
#         "EPSG": int(32631),
#         "ignore_dependencies": True
#     }
# )

# processed_cube = cube.apply(
#     process=udf,
#     context={
#         "classifier_url": str(CATBOOST_URL),
#         "EPSG": int(32631),
#         "ignore_dependencies": True
#     }
# )

# ======================================================
# JOB
# ======================================================
job = processed_cube.create_job(
    title="Cropland inference corrected pipeline",
    description="S2 + S1 + DEM + slope + AGERA5 + UDF inference",
    out_format="NetCDF"
)

print("Job created:", job.job_id)

job.start_job()

print("Job started successfully")