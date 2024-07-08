def _check_geom(row):
    try:
        result = row["geometry"].contains(row["centroid"])
    except Exception:
        result = False
    return result


def _to_points(df):
    """Convert reference dataset to points."""

    # if geometry type is point, return df
    if df["geometry"].geom_type[0] == "Point":
        return df
    else:
        # convert polygons to points
        df["centroid"] = df["geometry"].centroid
        # check whether centroid is in the original geometry
        df["centroid_in"] = df.apply(lambda x: _check_geom(x), axis=1)
        df = df[df["centroid_in"]]
        df.drop(columns=["geometry", "centroid_in"], inplace=True)
        df.rename(columns={"centroid": "geometry"}, inplace=True)
        return df
