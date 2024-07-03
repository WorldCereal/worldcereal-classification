import json
from typing import Tuple
import requests
from pathlib import Path
from typing import List, Optional
from functools import partial

import h3
import numpy as np
import pandas as pd
import geopandas as gpd
import geojson
import openeo
import pystac

from openeo_gfmap import Backend, BackendContext, FetchType, TemporalContext
from openeo_gfmap.backend import cdse_connection
from openeo_gfmap.manager.job_manager import GFMAPJobManager
from openeo_gfmap.manager.job_splitters import split_job_s2grid

from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs_gfmap

from presto.inference import process_parquet
from presto.utils import device
from shapely.geometry import Polygon, shape
from torch.utils.data import DataLoader


RDM_API = "https://ewoc-rdm-api.iiasa.ac.at"


def get_bbox_from_draw(dc, max_size=25000000):
    import geopandas as gpd

    obj = dc.last_draw
    if obj.get("geometry") is not None:
        poly = Polygon(shape(obj.get("geometry")))
        selected_area = gpd.GeoSeries(poly, crs="EPSG:4326").to_crs(epsg=3785).area[0]
        if selected_area > max_size:
            raise ValueError(
                f"Selected area is too large ({selected_area/1000000:.0f} km2). Please select an area smaller than {max_size/1000000:.0f} km2."
            )
        bbox = poly.bounds
    else:
        raise ValueError(
            "Please first draw a rectangle " "on the map before proceeding."
        )
    print(f"Your area of interest: {bbox} ({selected_area/1000000:.0f} km2)")

    return bbox, poly


def get_class_mappings():
    with open("resources/croptype_classes.json") as f:
        CLASS_MAPPINGS = json.load(f)

    return CLASS_MAPPINGS


def pick_croptypes(df: pd.DataFrame, samples_threshold: int = 100):
    import ipywidgets as widgets

    potential_classes = df["croptype_name"].value_counts().reset_index()
    potential_classes = potential_classes[
        potential_classes["count"] > samples_threshold
    ]

    checkbox_widgets = [
        widgets.Checkbox(
            value=False, description=f"{row['croptype_name']} ({row['count']} samples)"
        )
        for ii, row in potential_classes.iterrows()
    ]
    vbox = widgets.VBox(
        checkbox_widgets,
        layout=widgets.Layout(width="50%", display="inline-flex", flex_flow="row wrap"),
    )

    return vbox, checkbox_widgets


def query_worldcereal_samples(bbox_poly, buffer=250000, filter_cropland=True):
    import duckdb
    import geopandas as gpd

    print(f"Applying a buffer of {buffer/1000} km to the selected area ...")

    bbox_poly = (
        gpd.GeoSeries(bbox_poly, crs="EPSG:4326")
        .to_crs(epsg=3785)
        .buffer(buffer, cap_style="square", join_style="mitre")
        .to_crs(epsg=4326)[0]
    )

    xmin, ymin, xmax, ymax = bbox_poly.bounds
    twisted_bbox_poly = Polygon(
        [(ymin, xmin), (ymin, xmax), (ymax, xmax), (ymax, xmin)]
    )
    h3_cells_lst = []
    res = 5
    while len(h3_cells_lst) == 0:
        h3_cells_lst = list(h3.polyfill(twisted_bbox_poly.__geo_interface__, res))
        res += 1
    if res > 5:
        h3_cells_lst = tuple(np.unique([h3.h3_to_parent(xx, 5) for xx in h3_cells_lst]))

    db = duckdb.connect()
    db.sql("INSTALL spatial")
    db.load_extension("spatial")

    parquet_path = "s3://geoparquet/worldcereal_extractions_phase1/*/*.parquet"

    # only querying the croptype data here
    print("Querying WorldCereal global database ...")
    if filter_cropland:
        query = f"""
        set s3_endpoint='s3.waw3-1.cloudferro.com';
        set enable_progress_bar=false;
        select *
        from read_parquet('{parquet_path}', hive_partitioning = 1) original_data
        where original_data.h3_l5_cell in {h3_cells_lst}
        and original_data.LANDCOVER_LABEL = 11
        and original_data.CROPTYPE_LABEL not in (0, 991, 7900, 9900, 9998, 1910, 1900, 1920, 1000, 11, 9910, 6212, 7920, 9520, 3400, 3900, 4390, 4000, 4300)
        """
    else:
        query = f"""
            set s3_endpoint='s3.waw3-1.cloudferro.com';
            set enable_progress_bar=false;
            select *
            from read_parquet('{parquet_path}', hive_partitioning = 1) original_data
            where original_data.h3_l5_cell in {h3_cells_lst}
        """

    public_df_raw = db.sql(query).df()
    print("Processing selected samples ...")
    public_df = process_parquet(public_df_raw)
    public_df = map_croptypes(public_df)
    print(f"Extracted and processed {public_df.shape[0]} samples from global database.")

    return public_df


def get_inputs_outputs(
    df: pd.DataFrame, batch_size: int = 256, task_type: str = "croptype"
) -> Tuple[np.ndarray, np.ndarray]:
    from presto.dataset import WorldCerealLabelledDataset
    from presto.presto import Presto

    if task_type == "croptype":
        presto_model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ss-wc-ft-ct-30D_test.pt"
    if task_type == "cropland":
        presto_model_url = "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/PhaseII/presto-ft-cl_30D_cropland_random.pt"
        df["custom_class"] = (df["LANDCOVER_LABEL"] == 11).astype(int)
    presto_model = Presto.load_pretrained_url(presto_url=presto_model_url, strict=False)

    tds = WorldCerealLabelledDataset(df, target_function=lambda xx: xx["custom_class"])
    tdl = DataLoader(tds, batch_size=batch_size, shuffle=False)

    encoding_list, targets = [], []

    print("Computing Presto embeddings ...")
    for x, y, dw, latlons, month, variable_mask in tdl:
        x_f, dw_f, latlons_f, month_f, variable_mask_f = [
            t.to(device) for t in (x, dw, latlons, month, variable_mask)
        ]
        input_d = {
            "x": x_f,
            "dynamic_world": dw_f.long(),
            "latlons": latlons_f,
            "mask": variable_mask_f,
            "month": month_f,
        }

        presto_model.eval()
        encodings = presto_model.encoder(**input_d).detach().numpy()

        encoding_list.append(encodings)
        targets.append(y)

    encodings_np = np.concatenate(encoding_list)
    targets = np.concatenate(targets)

    print("Done.")

    return encodings_np, targets


def get_custom_labels(df, checkbox_widgets):
    selected_crops = [
        checkbox.description.split(" ")[0]
        for checkbox in checkbox_widgets
        if checkbox.value
    ]
    df["custom_class"] = "other"
    df.loc[df["croptype_name"].isin(selected_crops), "custom_class"] = df[
        "croptype_name"
    ]

    return df


def train_classifier(inputs, targets):
    import numpy as np
    from catboost import CatBoostClassifier, Pool
    from presto.utils import DEFAULT_SEED
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    print("Split train/test ...")
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(
        inputs,
        targets,
        stratify=targets,
        test_size=0.3,
        random_state=DEFAULT_SEED,
    )

    if np.unique(targets_train).shape[0] > 2:
        eval_metric = "MultiClass"
        loss_function = "MultiClass"
    else:
        eval_metric = "F1"
        loss_function = "Logloss"

    print("Computing class weights ...")
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(targets_train), y=targets_train
    )
    class_weights = {k: v for k, v in zip(np.unique(targets_train), class_weights)}
    print("Class weights:", class_weights)

    sample_weights = np.ones((len(targets_train),))
    sample_weights_val = np.ones((len(targets_val),))
    for k, v in class_weights.items():
        sample_weights[targets_train == k] = v
        sample_weights_val[targets_val == k] = v

    custom_downstream_model = CatBoostClassifier(
        iterations=8000,
        depth=8,
        learning_rate=0.05,
        early_stopping_rounds=50,
        # l2_leaf_reg=30,
        colsample_bylevel=0.9,
        l2_leaf_reg=3,
        loss_function=loss_function,
        eval_metric=eval_metric,
        random_state=DEFAULT_SEED,
        verbose=25,
        class_names=np.unique(targets_train),
    )

    print("Training CatBoost classifier ...")
    custom_downstream_model.fit(
        inputs_train,
        targets_train,
        sample_weight=sample_weights,
        eval_set=Pool(inputs_val, targets_val, weight=sample_weights_val),
    )

    pred = custom_downstream_model.predict(inputs_val).flatten()

    report = classification_report(targets_val, pred)

    return custom_downstream_model, report


def map_croptypes(
    df: pd.DataFrame,
    downstream_classes="CROPTYPE9",
) -> pd.DataFrame:
    wc2ewoc_map = pd.read_csv("resources/wc2eurocrops_map.csv")
    wc2ewoc_map["ewoc_code"] = wc2ewoc_map["ewoc_code"].str.replace("-", "").astype(int)

    ewoc_map = pd.read_csv("resources/eurocrops_map_wcr_edition.csv")
    ewoc_map = ewoc_map[ewoc_map["ewoc_code"].notna()]
    ewoc_map["ewoc_code"] = ewoc_map["ewoc_code"].str.replace("-", "").astype(int)
    ewoc_map = ewoc_map.apply(lambda x: x[: x.last_valid_index()].ffill(), axis=1)
    ewoc_map.set_index("ewoc_code", inplace=True)

    df["CROPTYPE_LABEL"].replace(0, np.nan, inplace=True)
    df["CROPTYPE_LABEL"].fillna(df["LANDCOVER_LABEL"], inplace=True)

    df["ewoc_code"] = df["CROPTYPE_LABEL"].map(
        wc2ewoc_map.set_index("croptype")["ewoc_code"]
    )
    df["landcover_name"] = df["ewoc_code"].map(ewoc_map["landcover_name"])
    df["cropgroup_name"] = df["ewoc_code"].map(ewoc_map["cropgroup_name"])
    df["croptype_name"] = df["ewoc_code"].map(ewoc_map["croptype_name"])

    df["downstream_class"] = df["ewoc_code"].map(
        {int(k): v for k, v in get_class_mappings()[downstream_classes].items()}
    )

    return df


def deploy_model(model, pattern=None):
    import datetime
    import subprocess
    import tempfile

    ARTIFACTORY_USERNAME = "worldcereal_model"

    # Generate a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if pattern is None:
        targetpath = f"{timestamp}_custommodel.onnx"
    else:
        targetpath = f"{pattern}_{timestamp}_custommodel.onnx"

    # Use a temporary file to save the ONNX model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as temp_file:
        model_file_path = temp_file.name

        model.save_model(
            f"{model_file_path}",
            format="onnx",
            export_parameters={
                "onnx_domain": "ai.catboost",
                "onnx_model_version": 1,
                "onnx_doc_string": "custom model for crop classification using CatBoost",
                "onnx_graph_name": "CatBoostModel_for_MulticlassClassification",
            },
        )

        print(f"Uploading model to `{targetpath}`")

        access_token = input("Enter your Artifactory API key: ")

        cmd = (
            f"curl -u{ARTIFACTORY_USERNAME}:{access_token} -T {model_file_path} "
            f'"https://artifactory.vgt.vito.be/artifactory/worldcereal_models/{targetpath}"'
        )

        output, _ = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, shell=True
        ).communicate()
        output = eval(output)

    uri = output["downloadUri"]

    print(f"Deployed to: {uri}")

    return uri


def terrascope_login():

    authBaseUrl = "https://sso.terrascope.be/auth/realms/terrascope"

    username = input("Enter your Terrascope username: ")
    password = input("Enter your Terrascope password: ")

    data = {
        "username": username,
        "password": password,
        "client_id": "worldcereal-rdm",
        "grant_type": "password",
    }

    # call terrascope auth server to get token
    tokenResponse = requests.post(
        f"{authBaseUrl}/protocol/openid-connect/token", data=data
    )

    token = tokenResponse.json()
    tokentype = token["token_type"]
    accessToken = token["access_token"]
    headers = {"Authorization": f"{tokentype} {accessToken}"}

    return headers


def rdm_collection_request(poly, headers=None, buffer=250000):

    if headers is None:
        headers = {}

    if buffer is not None:
        poly = (
            gpd.GeoSeries(poly, crs="EPSG:4326")
            .to_crs(epsg=3785)
            .buffer(buffer, cap_style="square", join_style="mitre")
            .to_crs(epsg=4326)[0]
        )

    bbox = poly.bounds
    bbox_str = f"Bbox={bbox[0]}&Bbox={bbox[1]}&Bbox={bbox[2]}&Bbox={bbox[3]}"

    colSearchUrl = f"{RDM_API}/collections/search?{bbox_str}"
    print(colSearchUrl)
    colSearchResponse = requests.get(colSearchUrl, headers=headers)
    test = colSearchResponse.json()

    col_ids = []
    print("The following collections intersect with your AOI:")
    for i, col in enumerate(test):
        print()
        print(
            f'Collection {i+1}: {col["collectionId"]} of type {col["type"]} containing {col["featureCount"]} samples'
        )
        col_ids.append(col["collectionId"])

    return col_ids


def rdm_features_request(
    poly, col_ids=None, headers=None, max_items=1000, buffer=250000
):

    from worldcereal.utils.refdata import _to_points

    if col_ids is None:
        col_ids = rdm_collection_request(poly, headers=headers, buffer=buffer)

    if headers is None:
        headers = {}

    if buffer is not None:
        poly = (
            gpd.GeoSeries(poly, crs="EPSG:4326")
            .to_crs(epsg=3785)
            .buffer(buffer, cap_style="square", join_style="mitre")
            .to_crs(epsg=4326)[0]
        )
    bbox = poly.bounds
    bbox_str = f"Bbox={bbox[0]}&Bbox={bbox[1]}&Bbox={bbox[2]}&Bbox={bbox[3]}"

    dfs = []
    for col_id in col_ids:
        featureSearchUrl = f"{RDM_API}/collections/{col_id}/items?{bbox_str}&MaxResultCount={max_items}"
        print(featureSearchUrl)
        featureSearchResponse = requests.get(featureSearchUrl, headers=headers)
        df = gpd.GeoDataFrame.from_features(
            featureSearchResponse.json(), crs="EPSG:4326"
        )
        dfs.append(_to_points(df))

    gdf = pd.concat(dfs, ignore_index=True)
    print(f"Got a total of {len(gdf)} reference points")

    return gdf


def create_job_dataframe(
    backend: Backend, split_jobs: List[gpd.GeoDataFrame]
) -> pd.DataFrame:
    """Create a dataframe from the split jobs, containg all the necessary information to run the job."""
    columns = [
        "backend_name",
        "out_extension",
        "start_date",
        "end_date",
        "s2_tile",
        "geometry",
    ]
    rows = []
    for job in split_jobs:
        # Compute the average in the valid date and make a buffer of 1.5 year around
        median_time = pd.to_datetime(job.valid_time).mean()
        # A bit more than 9 months
        start_date = median_time - pd.Timedelta(days=275)
        # A bit more than 9 months
        end_date = median_time + pd.Timedelta(days=275)
        s2_tile = job.tile.iloc[0]
        rows.append(
            pd.Series(
                dict(
                    zip(
                        columns,
                        [
                            backend.value,
                            ".parquet",
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                            s2_tile,
                            job.to_json(),
                        ],
                    )
                )
            )
        )

    return pd.DataFrame(rows)


def filter_extract_true(
    geometries: geojson.FeatureCollection,
) -> geojson.FeatureCollection:
    """Remove all the geometries from the Feature Collection that have the property field `extract` set to `False`"""
    return geojson.FeatureCollection(
        [f for f in geometries.features if f.properties.get("extract", 0) != 0]
    )


def get_job_nb_points(row: pd.Series) -> int:
    """Get the number of points in the geometry."""
    return len(
        list(
            filter(
                lambda feat: feat.properties.get("extract"),
                geojson.loads(row.geometry)["features"],
            )
        )
    )


def create_datacube(
    row: pd.Series,
    connection: openeo.DataCube,
    provider,
    connection_provider,
    executor_memory: str = "3G",
    executor_memory_overhead: str = "5G",
):
    """Creates an OpenEO BatchJob from the given row information."""

    # Load the temporal and spatial extent
    temporal_extent = TemporalContext(row.start_date, row.end_date)

    # Get the feature collection containing the geometry to the job
    geometry = geojson.loads(row.geometry)
    assert isinstance(geometry, geojson.FeatureCollection)

    # Filter the geometry to the rows with the extract only flag
    geometry = filter_extract_true(geometry)
    assert len(geometry.features) > 0, "No geometries with the extract flag found"

    # Backend name and fetching type
    backend = Backend(row.backend_name)
    backend_context = BackendContext(backend)

    inputs = worldcereal_preprocessed_inputs_gfmap(
        connection=connection,
        backend_context=backend_context,
        spatial_extent=geometry,
        temporal_extent=temporal_extent,
        fetch_type=FetchType.POINT,
    )

    # Finally, create a vector cube based on the Point geometries
    cube = inputs.aggregate_spatial(geometries=geometry, reducer="mean")

    # Increase the memory of the jobs depending on the number of polygons to extract
    number_points = get_job_nb_points(row)

    job_options = {
        "executor-memory": executor_memory,
        "executor-memoryOverhead": executor_memory_overhead,
        "soft-error": True,
    }
    return cube.create_job(
        out_format="Parquet",
        title=f"GFMAP_Feature_Extraction_{row.s2_tile}",
        job_options=job_options,
    )


def post_job_action(
    job_items: List[pystac.Item], row: pd.Series, parameters: dict = None
) -> list:
    for idx, item in enumerate(job_items):
        item_asset_path = Path(list(item.assets.values())[0].href)

        gdf = gpd.read_parquet(item_asset_path)

        # Convert the dates to datetime format
        gdf["date"] = pd.to_datetime(gdf["date"])

        # Convert band dtype to uint16 (temporary fix)
        # TODO: remove this step when the issue is fixed on the OpenEO backend
        bands = [
            "S2-L2A-B02",
            "S2-L2A-B03",
            "S2-L2A-B04",
            "S2-L2A-B05",
            "S2-L2A-B06",
            "S2-L2A-B07",
            "S2-L2A-B08",
            "S2-L2A-B11",
            "S2-L2A-B12",
            "S1-SIGMA0-VH",
            "S1-SIGMA0-VV",
            "COP-DEM",
            "AGERA5-PRECIP",
            "AGERA5-TMEAN",
        ]
        gdf[bands] = gdf[bands].fillna(65535).astype("uint16")

        gdf.to_parquet(item_asset_path, index=False)

    return job_items


def generate_output_path(root_folder: Path, geometry_index: int, row: pd.Series):
    features = geojson.loads(row.geometry)
    sample_id = features[geometry_index].properties.get("sample_id", None)
    if sample_id is None:
        sample_id = features[geometry_index].properties["sampleID"]

    s2_tile_id = row.s2_tile

    subfolder = root_folder / s2_tile_id
    return subfolder / f"{sample_id}{row.out_extension}"


def point_extractions(
    input_df, output_path, max_locations=500, memory="3G", memory_overhead="5G"
):

    # define path to tracking csv
    Path(output_path).mkdir(parents=True, exist_ok=True)
    tracking_df_path = Path(output_path) / "job_tracking.csv"

    # Load the input dataframe, and perform dataset splitting using the h3 tile
    # to respect the area of interest. Also filters out the jobs that have
    # no location with the extract=True flag.
    input_df = gpd.read_parquet(input_df)

    split_dfs = split_job_s2grid(input_df, max_points=max_locations)
    split_dfs = [df for df in split_dfs if df.extract.any()]

    job_df = create_job_dataframe(Backend.CDSE, split_dfs)

    # Setup the memory parameters for the job creator.
    create_cube = partial(
        create_datacube,
        executor_memory=memory,
        executor_memory_overhead=memory_overhead,
    )

    manager = GFMAPJobManager(
        output_dir=output_path,
        output_path_generator=generate_output_path,
        post_job_action=post_job_action,
        collection_id="POINT-FEATURE-EXTRACTION",
        collection_description="Worldcereal point feature extraction.",
        poll_sleep=60,
        n_threads=2,
        post_job_params={},
    )

    manager.add_backend(Backend.CDSE.value, cdse_connection, parallel_jobs=2)

    manager.run_jobs(job_df, create_cube, tracking_df_path)
