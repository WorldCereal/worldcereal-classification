import json
from typing import Tuple

import numpy as np
import pandas as pd

try:
    import presto
except ImportError:
    import sys

    presto_repo_path = "/home/cbutsko/Desktop/presto-worldcereal/"
    sys.path.append(presto_repo_path)
from presto.inference import process_parquet
from presto.utils import device
from shapely.geometry import Polygon, shape
from torch.utils.data import DataLoader


def get_bbox_from_draw(dc):
    obj = dc.last_draw
    if obj.get("geometry") is not None:
        poly = Polygon(shape(obj.get("geometry")))
        bbox = poly.bounds
    else:
        raise ValueError(
            "Please first draw a rectangle " "on the map before proceeding."
        )
    print(f"Your area of interest: {bbox}")

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
    widgets.VBox(
        checkbox_widgets,
        layout=widgets.Layout(width="50%", display="inline-flex", flex_flow="row wrap"),
    )

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


def query_worldcereal_samples(bbox_poly):
    import duckdb

    db = duckdb.connect()
    db.sql("INSTALL spatial")
    db.load_extension("spatial")

    parquet_path = "s3://geoparquet/worldcereal_extractions_phase1/*/*.parquet"

    # only querying the croptype data here
    print("Querying WorldCereal global database ...")
    public_df_raw = db.sql(
        f"""
    set s3_endpoint='s3.waw3-1.cloudferro.com';
    select *
    from read_parquet('{parquet_path}', hive_partitioning = 1) original_data
    where st_within(ST_Point(original_data.lon, original_data.lat), ST_GeomFromText('{bbox_poly.wkt}'))
    and original_data.CROPTYPE_LABEL not in (0, 991, 7900, 9900, 9998, 1910, 1900, 1920, 1000, 11, 9910, 6212, 7920, 9520, 3400, 3900, 4390, 4000, 4300)
    """
    ).df()
    print("Processing selected samples ...")
    public_df = process_parquet(public_df_raw)
    public_df = map_croptypes(public_df)
    print(f"Extracted and processed {public_df.shape[0]} samples from global database.")

    return public_df


def get_encodings_targets(
    df: pd.DataFrame, presto_model, batch_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    from presto.dataset import WorldCerealLabelledDataset

    tds = WorldCerealLabelledDataset(df, target_function=lambda xx: xx["custom_class"])
    tdl = DataLoader(tds, batch_size=batch_size, shuffle=False)

    encoding_list, targets = [], []

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

    return encodings_np, targets


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
    return df
