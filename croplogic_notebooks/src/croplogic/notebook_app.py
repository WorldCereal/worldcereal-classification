"""CropLogic crop type mapping app (notebook). Pick a country on the map, a season, the crops, draw a patch, choose the year, run.

    from croplogic_app import CropLogicApp
    app = CropLogicApp.run()

Needs an environment with the WorldCereal package (worldcereal, openeo, openeo_gfmap), ipyleaflet, shapely, pyproj, rasterio and Pillow, and a free Copernicus Data Space account (openEO credits).
The models (ONNX files) and the model list (openeo_registry.json) are downloaded from the web; the openEO cloud runs the model on the patch you draw."""

import base64
import datetime
import io
import json
import math
import os
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REGISTRY = os.environ.get(
    "CROPLOGIC_REGISTRY", "https://raw.githubusercontent.com/shabarinath4/croplogic_models/main/v5/openeo_registry.json"
)
MAX_KM2 = 100.0  # largest area of ONE openEO job; a bigger area is split into tiles of at most this size
MAX_PARALLEL = 2  # tiles run at the same time (the openEO free tier allows few concurrent jobs)
CROP_COLORS = {"wheat": "#c77d2b", "rapeseed_rape": "#e0b800", "maize": "#2f8f5b"}
PRETTY = {"rapeseed_rape": "rapeseed", "wheat": "wheat", "maize": "maize"}
SEASON_LABEL = {"s1": "Season 1", "s2": "Season 2"}
PROB_STOPS = [[247, 251, 255], [158, 202, 225], [66, 146, 198], [8, 69, 148]]
DATE_STOPS = [[255, 237, 160], [254, 178, 76], [240, 59, 32], [128, 0, 38]]


# ====================================================================================================================== pure logic (no widgets, no cloud)
def load_json(src):
    """URL or local file -> dict

    Args:
        src: Input value for ``src`` used by this operation.

    Returns:
        The decoded JSON value.
    """
    if os.path.exists(str(src)):
        return json.load(open(src, encoding="utf-8"))
    req = urllib.request.Request(str(src), headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def usable(e):
    """True when a registry entry has a checked ONNX model with a URL.

    Args:
        e: Input value for ``e`` used by this operation.

    Returns:
        The value returned by ``bool``.
    """
    return bool(e.get("onnx_ok") and e.get("onnx_url"))


def catalogue(registry):
    """{country: {frame: {crop: entry}}} for entries that can be run

    Args:
        registry: Input value for ``registry`` used by this operation.

    Returns:
        The computed out.
    """
    out = {}
    for k, e in (registry.get("entries") or registry).items():
        if usable(e):
            out.setdefault(e["country"], {}).setdefault(e.get("frame") or "s1", {})[e["crop"]] = e
    return out


def nearest_month_start(d):
    """Month start closest to a date.

    Args:
        d: Input value for ``d`` used by this operation.

    Returns:
        The computed result of nearest month start.
    """
    a = d.replace(day=1)
    b = (a + datetime.timedelta(days=32)).replace(day=1)
    return a if (d - a) <= (b - d) else b


def add_months(d, n):
    """Add months to a date.

    Args:
        d: Input value for ``d`` used by this operation.
        n: Input value for ``n`` used by this operation.

    Returns:
        The value returned by ``d.replace``.
    """
    m = d.month - 1 + n
    return d.replace(year=d.year + m // 12, month=m % 12 + 1, day=min(d.day, 28))


def wanted_start(entry, year):
    """the date the 36-dekad series must start so that the area's season start sits where the model saw it in training: season start - first_step dekads.
    `year` = the year the season is centred in (harvest year for crops sown in autumn).

    Args:
        entry: Input value for ``entry`` used by this operation.
        year: Input value for ``year`` used by this operation.

    Returns:
        The computed result of wanted start.
    """
    s = entry["season"]
    sos, eos = float(s["sos_doy"]), float(s["eos_doy"])
    sy = year - 1 if sos > eos else year
    return datetime.date(sy, 1, 1) + datetime.timedelta(days=int(sos) - 1 - 10 * int(s.get("first_step") or 0))


def job_window(entries, year):
    """-> (start, end, shifts): a 12-month window from the first to the last day of a month (WorldCereal needs that), placed by the first crop; shifts = dekads each crop's
    season is moved on the 36-dekad axis compared with training (its model windows are moved by the same amount).

    Args:
        entries: Input value for ``entries`` used by this operation.
        year: Input value for ``year`` used by this operation.

    Returns:
        A tuple containing ``m``, ``end``.
    """
    wants = [wanted_start(e, year) for e in entries]
    m = nearest_month_start(wants[0])
    end = add_months(m, 12) - datetime.timedelta(days=1)
    return m, end, [int(round((w - m).days / 10.0)) for w in wants]


def shifted(cfg, k):
    """Shift the selected and valid window starts by k dekads.

    Args:
        cfg: Configuration object or mapping that controls the operation.
        k: Input value for ``k`` used by this operation.

    Returns:
        A tuple containing ``sel``, ``val``.
    """
    hi = int(cfg.get("series_len", 36)) - int(cfg["window_len"])
    sel = sorted({min(max(int(i) + k, 0), hi) for i in cfg["sel_starts"]})
    val = sorted({min(max(int(i) + k, 0), hi) for i in (cfg.get("valid_starts") or range(hi + 1))})
    return sel, val


def udf_settings(entries, shifts):
    """context 'croplogic' for the UDF: every selected crop of the model, each with its own class index, rule and windows

    Args:
        entries: Input value for ``entries`` used by this operation.
        shifts: Input value for ``shifts`` used by this operation.

    Returns:
        A dictionary containing ``bands``, ``window_len``, ``crops``.
    """
    crops = []
    for e, k in zip(entries, shifts):
        c = e["cfg"]
        sel, val = shifted(c, k)
        crops.append(
            {
                "name": e["crop"],
                "class_index": c["class_index"],
                "rule": c["rule"],
                "param": c["param"],
                "sel_starts": sel,
                "valid_starts": val,
            }
        )
    c0 = entries[0]["cfg"]
    return {"bands": c0["bands"], "window_len": c0["window_len"], "crops": crops}


def groups_for(entries):
    """crops that share one model file are mapped by one job

    Args:
        entries: Input value for ``entries`` used by this operation.

    Returns:
        A list containing the computed values.
    """
    g = {}
    for e in entries:
        g.setdefault((e["onnx_url"], e.get("frame")), []).append(e)
    return list(g.values())


def bbox_area_km2(west, south, east, north):
    """Area of a bounding box in square kilometres.

    Args:
        west: Input value for ``west`` used by this operation.
        south: Input value for ``south`` used by this operation.
        east: Input value for ``east`` used by this operation.
        north: Input value for ``north`` used by this operation.

    Returns:
        The computed result of bbox area km2.
    """
    import math

    r = 6371.0088
    return abs(math.radians(east - west)) * r * r * abs(math.sin(math.radians(north)) - math.sin(math.radians(south)))


def tiles_for(bbox, max_km2=MAX_KM2):
    """Split a bounding box into equal tiles of at most max_km2 each -> list of bbox dicts (the box itself if it is small enough).

    Args:
        bbox: Input value for ``bbox`` used by this operation.
        max_km2: Input value for ``max_km2`` used by this operation.

    Returns:
        The computed out.
    """
    import math

    w, s_, e, n = (bbox[k] for k in ("west", "south", "east", "north"))
    if bbox_area_km2(w, s_, e, n) <= max_km2 * 1.01:
        return [dict(bbox)]
    side = math.sqrt(max_km2) * 0.95
    lat = math.radians((s_ + n) / 2)
    nx = max(1, math.ceil((e - w) * 111.195 * math.cos(lat) / side))
    ny = max(1, math.ceil((n - s_) * 111.195 / side))
    out = []
    for j in range(ny):
        for i in range(nx):
            out.append(
                {
                    "west": w + (e - w) * i / nx,
                    "east": w + (e - w) * (i + 1) / nx,
                    "south": s_ + (n - s_) * j / ny,
                    "north": s_ + (n - s_) * (j + 1) / ny,
                }
            )
    return out


def merge_tifs(tifs, out_tif):
    """Mosaic the GeoTIFFs of the tiles into one file (one tile: that file itself).

    Args:
        tifs: Input value for ``tifs`` used by this operation.
        out_tif: Input value for ``out_tif`` used by this operation.

    Returns:
        The computed out tif.
    """
    if len(tifs) == 1:
        return tifs[0]
    import rasterio
    from rasterio.merge import merge

    srcs = [rasterio.open(t) for t in tifs]
    try:
        nod = srcs[0].nodata
        arr, tr = merge(srcs, nodata=nod if nod is not None else 255)
        meta = srcs[0].meta.copy()
        meta.update(height=arr.shape[1], width=arr.shape[2], transform=tr, nodata=nod if nod is not None else 255)
    finally:
        for d in srcs:
            d.close()
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(arr)
    return out_tif


def utm_epsg(lon, lat):
    """EPSG code of the UTM zone of a point.

    Args:
        lon: Input value for ``lon`` used by this operation.
        lat: Input value for ``lat`` used by this operation.

    Returns:
        The computed result of utm epsg.
    """
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def dekad_end(start, k):
    """Last day of the k-th dekad counted from a month start.

    Args:
        start: Input value for ``start`` used by this operation.
        k: Input value for ``k`` used by this operation.

    Returns:
        The computed result of dekad end.
    """
    m = start.month - 1 + k // 3
    y = start.year + m // 12
    m = m % 12 + 1
    d = k % 3
    if d < 2:
        return datetime.date(y, m, 10 * (d + 1))
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return datetime.date(ny, nm, 1) - datetime.timedelta(days=1)


def _ramp(t, stops):
    """Colours along a ramp for values between 0 and 1.

    Args:
        t: Input value for ``t`` used by this operation.
        stops: Input value for ``stops`` used by this operation.

    Returns:
        The value returned by ``(s[i] * (1 - f) + s[i + 1] * f).astype``.
    """
    t = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[..., None]
    s = np.array(stops, float)
    return (s[i] * (1 - f) + s[i + 1] * f).astype(np.uint8)


def _png(rgba):
    """Palette PNG of an RGBA array.

    Args:
        rgba: Input value for ``rgba`` used by this operation.

    Returns:
        The computed result of png.
    """
    from PIL import Image

    im = Image.fromarray(rgba, "RGBA")
    try:
        im = im.quantize(colors=48, method=Image.Quantize.FASTOCTREE)
    except Exception:
        pass
    b = io.BytesIO()
    im.save(b, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def read_result(tif, crops, start, window_len, max_px=900):
    """GeoTIFF of a job -> {'bounds': [[s, w], [n, e]], 'crops': {crop: {'map', 'prob', 'date' (data URIs), 'area_ha', 'cropland_ha', 'first', 'last'}}}

    Args:
        tif: Input value for ``tif`` used by this operation.
        crops: Input value for ``crops`` used by this operation.
        start: Input value for ``start`` used by this operation.
        window_len: Input value for ``window_len`` used by this operation.
        max_px: Input value for ``max_px`` used by this operation.

    Returns:
        A dictionary containing ``bounds``, ``crops``.
    """
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(tif) as ds:
        arr = ds.read().astype("float32")
        nod = ds.nodata
        w_, s_, e_, n_ = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        px_ha = abs(ds.transform.a * ds.transform.e) / 1e4
    h, w = arr.shape[1:]
    f = max(1, int(math.ceil(max(h, w) / max_px)))
    sl = (slice(None, None, f), slice(None, None, f))
    out = {}
    for i, crop in enumerate(crops):
        if arr.shape[0] < 3 * i + 3:
            continue
        cl, pr, dt = arr[3 * i], arr[3 * i + 1], arr[3 * i + 2]
        bad = ~np.isfinite(cl) | (cl >= 254) | ((cl == nod) if nod is not None else False)
        det = (cl >= 1) & ~bad
        b_, d_, p_, t_ = bad[sl], det[sl], pr[sl], dt[sl]
        m = np.zeros(b_.shape + (4,), np.uint8)
        m[~b_] = [120, 120, 120, 40]
        col = CROP_COLORS.get(crop, "#cc0000")
        m[d_] = [int(col[j : j + 2], 16) for j in (1, 3, 5)] + [225]
        pm = np.zeros_like(m)
        ok = ~b_ & np.isfinite(p_)
        pm[ok, :3] = _ramp(np.nan_to_num(p_[ok]) / 100.0, PROB_STOPS)
        pm[ok, 3] = 210
        dm = np.zeros_like(m)
        okd = d_ & np.isfinite(t_)
        first = last = None
        if okd.any():
            idx = np.round(t_[okd]).astype(int)
            lo, hi = int(idx.min()), int(idx.max())
            dm[okd, :3] = _ramp((idx - lo) / max(hi - lo, 1), DATE_STOPS)
            dm[okd, 3] = 225
            first, last = dekad_end(start, lo + window_len - 1), dekad_end(start, hi + window_len - 1)
        out[crop] = {
            "map": _png(m),
            "prob": _png(pm),
            "date": _png(dm) if okd.any() else None,
            "area_ha": float(det.sum() * px_ha),
            "cropland_ha": float((~bad).sum() * px_ha),
            "first": first,
            "last": last,
        }
    return {"bounds": [[s_, w_], [n_, e_]], "crops": out}


# ====================================================================================================================== the openEO job
def _busy(ex):
    """True for errors that mean the backend is busy (429, concurrent job limit).

    Args:
        ex: Input value for ``ex`` used by this operation.

    Returns:
        A boolean result.
    """
    t = f"{type(ex).__name__} {ex}"
    return "429" in t or "Too Many Requests" in t or "ConcurrentJobLimit" in t


def retrying(fn, tries=8):
    """Call a function again while the backend is busy.

    Args:
        fn: Input value for ``fn`` used by this operation.
        tries: Input value for ``tries`` used by this operation.

    Returns:
        The value returned by ``fn``.
    """
    for i in range(tries):
        try:
            return fn()
        except Exception as ex:
            if not _busy(ex) or i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def build_preprocessed_cube(connection, spatial_extent, temporal_extent_list, target_epsg):
    """the WorldCereal preprocessing chain the models were trained on (same band list and merge order as the validation runs)

    Args:
        connection: Input value for ``connection`` used by this operation.
        spatial_extent: Input value for ``spatial_extent`` used by this operation.
        temporal_extent_list: Input value for ``temporal_extent_list`` used by this operation.
        target_epsg: Input value for ``target_epsg`` used by this operation.

    Returns:
        The value returned by ``s2.merge_cubes(scl).merge_cubes(cube.filter_bands(['S1-SIGMA0-VV'])).merge_cubes(cube.filter_bands(['S1-SIGMA0-VH'])).merge_cubes(cube.filter_bands(['slope', 'elevation'])).merge_cubes(cube.filter_bands(['AGERA5-TMEAN'])).merge_cubes``.
    """
    from openeo_gfmap import Backend, BackendContext, BoundingBoxExtent, FetchType, TemporalContext
    from worldcereal.openeo.preprocessing import worldcereal_preprocessed_inputs

    bbox = BoundingBoxExtent(
        west=spatial_extent["west"],
        south=spatial_extent["south"],
        east=spatial_extent["east"],
        north=spatial_extent["north"],
        epsg=4326,
    )
    cube = worldcereal_preprocessed_inputs(
        connection=connection,
        backend_context=BackendContext(Backend.CDSE),
        spatial_extent=bbox,
        temporal_extent=TemporalContext(temporal_extent_list[0], temporal_extent_list[1]),
        fetch_type=FetchType.TILE,
        compositing_window="dekad",
        target_epsg=target_epsg,
        optical_mask_method="mask_scl_dilation",
        disable_meteo=False,
    )
    cube = cube.filter_bbox(dict(bbox))
    s2 = cube.filter_bands(
        [
            "S2-L2A-B02",
            "S2-L2A-B03",
            "S2-L2A-B04",
            "S2-L2A-B05",
            "S2-L2A-B06",
            "S2-L2A-B07",
            "S2-L2A-B08",
            "S2-L2A-B8A",
            "S2-L2A-B11",
            "S2-L2A-B12",
        ]
    )
    scl = (
        cube.filter_bands(["S2-L2A-B02"]).apply(lambda x: x * 0).rename_labels(dimension="bands", target=["S2-L2A-SCL"])
    )
    return (
        s2.merge_cubes(scl)
        .merge_cubes(cube.filter_bands(["S1-SIGMA0-VV"]))
        .merge_cubes(cube.filter_bands(["S1-SIGMA0-VH"]))
        .merge_cubes(cube.filter_bands(["slope", "elevation"]))
        .merge_cubes(cube.filter_bands(["AGERA5-TMEAN"]))
        .merge_cubes(cube.filter_bands(["AGERA5-PRECIP"]))
    )


def submit_job(conn, bbox, start, end, group, shifts, title, epsg=None):
    """-> openEO job (started). bbox = dict west/south/east/north. epsg: the same projection for all tiles of one map.

    Args:
        conn: Input value for ``conn`` used by this operation.
        bbox: Input value for ``bbox`` used by this operation.
        start: Input value for ``start`` used by this operation.
        end: Input value for ``end`` used by this operation.
        group: Input value for ``group`` used by this operation.
        shifts: Input value for ``shifts`` used by this operation.
        title: Input value for ``title`` used by this operation.
        epsg: Input value for ``epsg`` used by this operation.

    Returns:
        The computed job.
    """
    from openeo import UDF
    from openeo.processes import neq

    epsg = epsg or utm_epsg((bbox["west"] + bbox["east"]) / 2, (bbox["south"] + bbox["north"]) / 2)
    urllib.request.install_opener(_ua_opener())
    cube = retrying(lambda: build_preprocessed_cube(conn, bbox, [start.isoformat(), end.isoformat()], epsg))
    ctx = {
        "classifier_url": group[0]["onnx_url"],
        "model_json_url": None,
        "EPSG": epsg,
        "ignore_dependencies": True,
        "smooth_kernel": 1,
        "croplogic": udf_settings(group, shifts),
    }
    pc = cube.apply_neighborhood(
        process=UDF.from_file(os.path.join(HERE, "openeo", "inference_udf.py"), context={"from_parameter": "context"}),
        size=[{"dimension": "x", "value": 32, "unit": "px"}, {"dimension": "y", "value": 32, "unit": "px"}],
        context=ctx,
    )
    pc = pc.rename_labels(
        dimension="bands",
        target=[f"{l}_{e['crop']}" for e in group for l in ("classification", "probability", "detection_time")],
    )
    lc = (
        conn.load_collection("ESA_WORLDCOVER_10M_2021_V2", spatial_extent=bbox)
        .reduce_dimension(dimension="t", reducer="first")
        .resample_cube_spatial(pc)
    )
    pc = pc.mask(lc.apply(lambda x: neq(x, 40)))
    job = pc.create_job(title=title, description=title, out_format="GTiff")
    retrying(job.start_job)
    return job


def _ua_opener():
    """URL opener with a browser-style User-Agent.

    Returns:
        The computed op.
    """
    op = urllib.request.build_opener()
    op.addheaders = [("User-Agent", "Mozilla/5.0")]
    return op


# ====================================================================================================================== the app
CAL_URL = "https://raw.githubusercontent.com/WorldCereal/worldcereal-classification/main/src/worldcereal/data/cropcalendars/seasonality_lookup.geoparquet"
_CAL = {}


def load_calendar():
    """WorldCereal crop calendar points (lat, lon, season 1/2 start and end), downloaded once from GitHub.

    Returns:
        The computed result of load calendar.
    """
    if "df" not in _CAL:
        import tempfile

        import geopandas as gpd

        f = os.path.join(tempfile.gettempdir(), "worldcereal_seasonality_lookup.geoparquet")
        if not os.path.exists(f):
            urllib.request.urlretrieve(CAL_URL, f)
        d = gpd.read_parquet(f)
        _CAL["df"] = d[["lat", "lon", "s1_sos_date", "s1_eos_date", "s2_sos_date", "s2_eos_date"]].reset_index(
            drop=True
        )
    return _CAL["df"]


def calendar_text(r):
    """One line describing the seasons of a calendar point.

    Args:
        r: Input value for ``r`` used by this operation.

    Returns:
        The computed result of calendar text.
    """
    f = lambda d: d.strftime("%d %b")
    return (
        f"WorldCereal calendar at {r['lat']:.2f}, {r['lon']:.2f}: season 1 {f(r['s1_sos_date'])} to {f(r['s1_eos_date'])}; "
        f"season 2 {f(r['s2_sos_date'])} to {f(r['s2_eos_date'])}"
    )


STYLE = """<style>
.cl-btn{border-radius:8px !important;box-shadow:none !important;font-weight:500 !important}
.cl-card{border-radius:12px !important;background:#fbfdfd}
</style>"""
TEAL, PALE, LINE = "#0b5f59", "#eef5f4", "#c4dbd8"


def _label(text):
    """Small grey caption on the left of a control.

    Args:
        text: Input value for ``text`` used by this operation.

    Returns:
        The value returned by ``W.HTML``.
    """
    import ipywidgets as W

    return W.HTML(
        f"<div style='color:#6b7b7a;font-size:12px;letter-spacing:.06em;text-transform:uppercase;padding-top:9px'>{text}</div>",
        layout=W.Layout(width="90px", min_width="90px"),
    )


class _Seg:
    """Row of buttons: pick one (multi=False) or any number (multi=True); selected buttons are filled teal."""

    def __init__(self, W, multi=False, options=(), value=None) -> None:
        """Create the buttons and their container.

        Args:
            W: Input value for ``W`` used by this operation.
            multi: Input value for ``multi`` used by this operation.
            options: Input value for ``options`` used by this operation.
            value: Value associated with the key or item.
        """
        self.W = W
        self.multi = multi
        self.box = W.HBox(layout=W.Layout(flex_flow="row wrap"))
        self._b = {}
        self._cb = []
        self._val = () if multi else None
        self._dis = False
        self.options = options
        if value is not None:
            self._val = value
            self._paint()

    @property
    def options(self):
        """The (label, key) pairs of the buttons.

        Returns:
            The computed result of options.
        """
        return [(b.description, k) for k, b in self._b.items()]

    @options.setter
    def options(self, opts) -> None:
        """The (label, key) pairs of the buttons.

        Args:
            opts: Input value for ``opts`` used by this operation.
        """
        self._b = {}
        self._val = () if self.multi else None
        for label, key in opts:
            b = self.W.Button(
                description=label,
                disabled=self._dis,
                layout=self.W.Layout(width="auto", height="34px", margin="0 8px 8px 0", border="1px solid " + LINE),
            )
            b.add_class("cl-btn")
            b.on_click(lambda _b, k=key: self._click(k))
            self._b[key] = b
        self.box.children = list(self._b.values())
        self._paint()

    @property
    def value(self):
        """The selected key (one button) or tuple of keys (several buttons).

        Returns:
            The computed result of value.
        """
        return self._val

    @value.setter
    def value(self, v) -> None:
        """The selected key (one button) or tuple of keys (several buttons).

        Args:
            v: Input value for ``v`` used by this operation.
        """
        self._val = (tuple(v) if v else ()) if self.multi else v
        self._paint()
        self._fire()

    @property
    def disabled(self):
        """Whether the buttons are switched off.

        Returns:
            The computed result of disabled.
        """
        return self._dis

    @disabled.setter
    def disabled(self, flag) -> None:
        """Whether the buttons are switched off.

        Args:
            flag: Input value for ``flag`` used by this operation.
        """
        self._dis = flag
        for b in self._b.values():
            b.disabled = flag

    def observe(self, cb, names=None) -> None:
        """Register a function to call when the selection changes.

        Args:
            cb: Input value for ``cb`` used by this operation.
            names: Input value for ``names`` used by this operation.
        """
        self._cb.append(cb)

    def _click(self, k) -> None:
        """A button was pressed.

        Args:
            k: Input value for ``k`` used by this operation.
        """
        if self.multi:
            self._val = tuple(x for x in self._b if (x in self._val) != (x == k))
        else:
            self._val = k
        self._paint()
        self._fire()

    def _paint(self) -> None:
        """Colour the selected buttons."""
        for k, b in self._b.items():
            on = (k in self._val) if self.multi else (k == self._val)
            b.style.button_color = TEAL if on else PALE
            b.style.text_color = "white" if on else TEAL

    def _fire(self) -> None:
        """Call the registered functions."""
        for cb in self._cb:
            cb({"new": self._val})


class CropLogicApp:
    """Notebook app: pick a country, season and crops, draw a patch, run it on openEO and show the result."""

    def __init__(self, registry=DEFAULT_REGISTRY, countries=None) -> None:
        """Build the widgets and the map from the registry.

        Args:
            registry: Input value for ``registry`` used by this operation.
            countries: Input value for ``countries`` used by this operation.

        Raises:
            RuntimeError: If the operation cannot complete under the required conditions.
        """
        import ipywidgets as W
        from ipyleaflet import DrawControl, GeoJSON, Map, basemaps

        self.W = W
        reg = registry if isinstance(registry, dict) else load_json(registry)
        self.cat = catalogue(reg)
        if not self.cat:
            raise RuntimeError(
                "no model in the registry has a download URL (run 17_openeo_package.py with --base-url and upload openeo_registry.json)"
            )
        geo = load_json(countries or os.path.join(HERE, "data", "world_countries.geojson"))
        self.geo = {f["id"]: f for f in geo["features"]}
        self.country = self.frame = None
        self.bbox = None
        self.job_result = None
        self.overlays = []
        self.m = Map(
            center=(20, 10),
            zoom=2,
            scroll_wheel_zoom=True,
            basemap=basemaps.Esri.WorldStreetMap,
            layout=W.Layout(height="520px"),
        )
        have = {
            "type": "FeatureCollection",
            "features": [
                dict(self.geo[c], properties={"code": c, "name": self.geo[c]["properties"].get("name", c)})
                for c in sorted(self.cat)
                if c in self.geo
            ],
        }
        self.layer = GeoJSON(
            data=have,
            style={"color": "#1f6feb", "weight": 1, "fillColor": "#1f6feb", "fillOpacity": 0.35},
            hover_style={"fillOpacity": 0.65, "weight": 2},
            name="countries with a model",
        )
        self.layer.on_click(self._clicked)
        self.m.add_layer(self.layer)
        self.draw = DrawControl(
            rectangle={"shapeOptions": {"color": "#d73027"}},
            polygon={"shapeOptions": {"color": "#d73027"}},
            circle={},
            circlemarker={},
            polyline={},
            marker={},
            edit=False,
            remove=True,
        )
        self.draw.on_draw(self._drawn)
        self.m.add_control(self.draw)
        self.w_country = W.Dropdown(
            options=[("(click a blue country on the map)", None)]
            + [
                (f"{self.geo[c]['properties'].get('name', c)} ({c})" if c in self.geo else c, c)
                for c in sorted(self.cat)
            ],
            layout=W.Layout(width="360px"),
        )
        self.w_season = _Seg(W)
        self.w_crops = _Seg(W, multi=True)
        this_year = datetime.date.today().year
        self.w_year = W.Dropdown(
            options=list(range(this_year, 2017, -1)), value=this_year - 1, layout=W.Layout(width="120px")
        )
        self.w_info = W.HTML()
        self.w_area = W.HTML(
            value="<i>2. Draw a rectangle or polygon on the map, or click a grid cell (areas above %d km&sup2; are split into tiles).</i>"
            % MAX_KM2
        )
        self.w_win = W.HTML()
        self.w_grid = W.Checkbox(
            value=True,
            description="Show the 10 x 10 km grid (zoom in, then click a cell to select it)",
            indent=False,
            layout=W.Layout(width="auto"),
        )
        self.w_cal = W.Checkbox(
            value=False,
            description="Show the WorldCereal crop calendar points (click a point for its season dates)",
            indent=False,
            layout=W.Layout(width="auto"),
        )
        self.w_calinfo = W.HTML()
        self.cal_layer = None
        self.grid_layer = self.sel_layer = self._cg = None
        self.w_run = W.Button(
            description="Run",
            icon="play",
            disabled=True,
            layout=W.Layout(width="140px", height="34px", margin="0 0 0 16px"),
        )
        self.w_run.add_class("cl-btn")
        self.w_run.style.button_color = TEAL
        self.w_run.style.text_color = "white"
        self.w_status = W.HTML()
        self.w_legend = W.HTML()
        self.w_layer = _Seg(
            W, options=[("Crop map", "map"), ("Probability", "prob"), ("In-season date", "date")], value="map"
        )
        self.w_layer.disabled = True
        self.w_country.observe(lambda c: self._pick_country(c["new"]), names="value")
        self.w_season.observe(lambda c: self._pick_season(c["new"]), names="value")
        self.w_crops.observe(lambda c: self._refresh(), names="value")
        self.w_year.observe(lambda c: self._refresh(), names="value")
        self.w_grid.observe(lambda c: self._grid(), names="value")
        self.w_cal.observe(lambda c: self._cal(), names="value")
        self.m.observe(lambda c: self._cal(), names=["bounds"])
        self.m.observe(lambda c: self._grid(), names=["bounds"])
        self.w_run.on_click(lambda b: self._run())
        self.w_layer.observe(lambda c: self._show_layer(), names="value")
        card = lambda kids: W.VBox(
            kids,
            layout=W.Layout(border="1px solid #e1ebe9", padding="14px 18px", margin="0 0 14px 0"),
            _dom_classes=["cl-card"],
        )
        row = lambda t, w: W.HBox([_label(t), w], layout=W.Layout(align_items="flex-start", margin="0 0 4px 0"))
        head = W.HTML(
            "<div style='font-size:24px;font-weight:600;color:#0b3d3a;margin:4px 0 2px'>CropLogic</div>"
            "<div style='color:#6b7b7a;font-size:14px;margin-bottom:14px'>Crop type maps for your area. Pick a model, choose a 10 x 10 km cell or draw your own patch, and run it on openEO "
            "(about 10 minutes, uses your free openEO credits).</div>"
        )
        self.ui = W.VBox(
            [
                W.HTML(STYLE),
                head,
                card(
                    [
                        row("Country", self.w_country),
                        row("Season", self.w_season.box),
                        row("Crops", self.w_crops.box),
                        self.w_info,
                    ]
                ),
                card(
                    [
                        W.HBox([self.w_grid, self.w_cal], layout=W.Layout(flex_flow="row wrap", gap="24px")),
                        self.w_calinfo,
                        self.m,
                        self.w_area,
                        W.HBox(
                            [_label("Year"), self.w_year, self.w_run],
                            layout=W.Layout(align_items="center", margin="8px 0 0 0"),
                        ),
                        self.w_win,
                    ]
                ),
                self.w_status,
                row("Show", self.w_layer.box),
                self.w_legend,
            ]
        )

    @classmethod
    def run(cls, **kw):
        """Create the app and display it.

        Args:
            kw: Input value for ``kw`` used by this operation.

        Returns:
            The computed app.
        """
        from IPython.display import display

        app = cls(**kw)
        display(app.ui)
        return app

    # ------------------------------------------------------------------ selection
    def _clicked(self, feature=None, **kw) -> None:
        """A country was clicked on the map.

        Args:
            feature: Input value for ``feature`` used by this operation.
            kw: Input value for ``kw`` used by this operation.
        """
        code = (feature or {}).get("properties", {}).get("code")
        if code in self.cat:
            self.w_country.value = code

    def _pick_country(self, code) -> None:
        """Country chosen: offer its seasons.

        Args:
            code: Input value for ``code`` used by this operation.
        """
        self.country = code
        on = not code  # blue fill only while no country is chosen, so the result map stays visible
        self.layer.style = {"color": "#1f6feb", "weight": 1, "fillColor": "#1f6feb", "fillOpacity": 0.35 if on else 0.0}
        self.layer.hover_style = {"fillOpacity": 0.65 if on else 0.0, "weight": 2}
        self._cg = None
        if not code:
            self.w_season.options = []
            self.frame = None
            self.w_crops.options = []
            self.w_info.value = ""
            self._refresh()
            self._grid()
            return
        frames = sorted(self.cat[code])
        self.w_season.options = [
            (f"{SEASON_LABEL.get(f, f)}: " + ", ".join(PRETTY.get(c, c) for c in sorted(self.cat[code][f])), f)
            for f in frames
        ]
        self.w_season.value = frames[0]
        f = self.geo.get(code)
        if f:
            from shapely.geometry import shape

            g = shape(f["geometry"])
            self._cg = g
            b = g.bounds
            la, lo = (b[1] + b[3]) / 2, (b[0] + b[2]) / 2
            zl = math.log2(900 * 360 / (256 * max(b[2] - b[0], 0.01)))
            zh = math.log2(480 * 360 / (256 * max((b[3] - b[1]) / max(math.cos(math.radians(la)), 0.2), 0.01)))
            self.m.center = (la, lo)
            self.m.zoom = int(max(2, min(10, math.floor(min(zl, zh)))))

    def _pick_season(self, frame) -> None:
        """Season chosen: offer its crops.

        Args:
            frame: Input pandas DataFrame.
        """
        self.frame = frame
        if not frame or not self.country:
            self.w_crops.options = []
            self._refresh()
            return
        ent = self.cat[self.country][frame]
        self.w_crops.options = [(PRETTY.get(c, c), c) for c in sorted(ent)]
        self.w_crops.value = tuple(sorted(ent))
        self._refresh()

    def selected(self):
        """Registry entries of the selected crops.

        Returns:
            The computed result of selected.
        """
        if not (self.country and self.frame):
            return []
        ent = self.cat[self.country][self.frame]
        return [ent[c] for c in self.w_crops.value if c in ent]

    def _notes(self, es):
        """Notes about the selected models (route, reliability, warnings).

        Args:
            es: Input value for ``es`` used by this operation.

        Returns:
            The value returned by ``'<br>'.join``.
        """
        out = []
        for e in es:
            n = []
            if e["route"] == "joint":
                n.append(f"joint model: trained on {self.country} together with a country of similar crops")
            elif e["route"] != "own":
                src = (e.get("model") or "").split(":")[-1]
                n.append(
                    f"model of {src}: not trained on {self.country} data"
                    + (f", similarity {e.get('similarity')}" if e.get("similarity") else "")
                )
            if e.get("reliable") is False:
                n.append("below the quality gate in its training country: use with care")
            if e.get("expected_f1") is not None:
                n.append(
                    f"expected F1 {e['expected_f1']:.2f}"
                    + (
                        " (held out, same country)"
                        if e["route"] == "own"
                        else " (in the country the model was trained in)"
                    )
                )
            out.append(f"<b>{PRETTY.get(e['crop'], e['crop'])}</b>: " + "; ".join(n))
        return "<br>".join(out)

    def _refresh(self) -> None:
        """Update the model notes and the date window."""
        es = self.selected()
        self.w_info.value = f"<div style='color:#555;font-size:13px'>{self._notes(es)}</div>" if es else ""
        if es:
            try:
                s, e_, sh = job_window(es, int(self.w_year.value))
                today = datetime.date.today()
                self.w_win.value = (
                    f"<i>Model window: {s} to {e_} (12 months matching the season of {', '.join(PRETTY.get(x['crop'], x['crop']) for x in es)}, harvest year {self.w_year.value})</i>"
                    + (
                        "<br><span style='color:#c60'>This season is not finished yet: the map uses the data available so far.</span>"
                        if e_ > today
                        else ""
                    )
                )
            except Exception as ex:
                self.w_win.value = f"<span style='color:#d73027'>cannot place the season window: {ex}</span>"
        else:
            self.w_win.value = ""
        self._check()

    # ------------------------------------------------------------------ WorldCereal crop calendar
    def _cal(self) -> None:
        """Show the WorldCereal calendar points in view (zoom 4 or closer)."""
        from ipyleaflet import GeoJSON

        if self.w_calinfo.value.startswith("<i>Zoom"):
            self.w_calinfo.value = ""
        if self.cal_layer is not None:
            self.m.remove_layer(self.cal_layer)
            self.cal_layer = None
        if not self.w_cal.value or self.m.zoom < 4:
            return
        try:
            d = load_calendar()
            (south, west), (north, east) = self.m.bounds
            v = d[(d.lat >= south) & (d.lat <= north) & (d.lon >= west) & (d.lon <= east)]
            if len(v) > 2500:
                self.w_calinfo.value = "<i>Zoom in further to see the calendar points.</i>"
                return
            feats = [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(r["lon"]), float(r["lat"])]},
                    "properties": {"txt": calendar_text(r)},
                }
                for r in v.to_dict("records")
            ]
            self.cal_layer = GeoJSON(
                data={"type": "FeatureCollection", "features": feats},
                point_style={"radius": 5, "color": "#6a3d9a", "fillColor": "#6a3d9a", "fillOpacity": 0.7, "weight": 1},
                name="WorldCereal crop calendar",
            )
            self.cal_layer.on_click(
                lambda feature=None, **kw: (
                    setattr(
                        self.w_calinfo,
                        "value",
                        f"<div style='color:#555;font-size:13px'>{feature['properties']['txt']}</div>",
                    )
                    if feature
                    else None
                )
            )
            self.m.add_layer(self.cal_layer)
        except Exception as ex:
            self.w_calinfo.value = f"<span style='color:#d73027'>crop calendar not available: {ex}</span>"

    def _cal_here(self) -> None:
        """Calendar point nearest to the selected patch."""
        if not (self.w_cal.value and self.bbox):
            return
        try:
            d = load_calendar()
            la = (self.bbox["north"] + self.bbox["south"]) / 2
            lo = (self.bbox["east"] + self.bbox["west"]) / 2
            r = d.iloc[((d.lat - la) ** 2 + (d.lon - lo) ** 2).values.argmin()]
            self.w_calinfo.value = (
                f"<div style='color:#555;font-size:13px'>Nearest to your patch. {calendar_text(r)}</div>"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------ 10 km grid
    def _grid(self) -> None:
        """Draw the 10 x 10 km grid cells in view (zoom 7 or closer) and let the user click one."""
        from ipyleaflet import GeoJSON

        if self.grid_layer is not None:
            self.m.remove_layer(self.grid_layer)
            self.grid_layer = None
        if not self.w_grid.value or not self.country or self.m.zoom < 7:
            return
        try:
            from shapely.geometry import box

            south, west, north, east = (
                self.m.bounds[0][0],
                self.m.bounds[0][1],
                self.m.bounds[1][0],
                self.m.bounds[1][1],
            )
            km = 111.195
            dlat = 10.0 / km
            feats = []
            for j in range(int(math.floor(south / dlat)), int(math.ceil(north / dlat)) + 1):
                s0, n0 = j * dlat, (j + 1) * dlat
                if abs(s0) > 80:
                    continue
                dlon = 10.0 / (km * math.cos(math.radians((s0 + n0) / 2)))
                for i in range(int(math.floor(west / dlon)), int(math.ceil(east / dlon)) + 1):
                    c = box(i * dlon, s0, (i + 1) * dlon, n0)
                    if self._cg is None or self._cg.intersects(c):
                        feats.append(
                            {"type": "Feature", "geometry": c.__geo_interface__, "properties": {"cell": f"{j}_{i}"}}
                        )
                if len(feats) > 3000:
                    break
            if len(feats) > 3000:
                return
            self.grid_layer = GeoJSON(
                data={"type": "FeatureCollection", "features": feats},
                style={"color": "#555", "weight": 1, "opacity": 0.6, "fillColor": "#ffffff", "fillOpacity": 0.02},
                hover_style={"fillColor": "#ff8c00", "fillOpacity": 0.35, "weight": 2},
                name="10 km grid",
            )
            self.grid_layer.on_click(self._cell)
            self.m.add_layer(self.grid_layer)
        except Exception as ex:
            self.w_area.value = f"<span style='color:#d73027'>grid not available: {ex}</span>"

    def _cell(self, feature=None, **kw) -> None:
        """A grid cell was clicked: use it as the patch.

        Args:
            feature: Input value for ``feature`` used by this operation.
            kw: Input value for ``kw`` used by this operation.
        """
        from ipyleaflet import GeoJSON

        if not feature:
            return
        self._clear_sel()
        g = feature["geometry"]
        xs = [p[0] for p in g["coordinates"][0]]
        ys = [p[1] for p in g["coordinates"][0]]
        self.bbox = {"west": min(xs), "south": min(ys), "east": max(xs), "north": max(ys)}
        self.geom = g
        try:
            self.draw.clear()
        except Exception:
            pass
        self.sel_layer = GeoJSON(
            data={"type": "Feature", "geometry": g, "properties": {}},
            style={"color": "#d73027", "weight": 3, "fillColor": "#d73027", "fillOpacity": 0.15},
            name="selected cell",
        )
        self.m.add_layer(self.sel_layer)
        self._check()
        self._cal_here()

    def _clear_sel(self) -> None:
        """Remove the highlighted grid cell."""
        if self.sel_layer is not None:
            self.m.remove_layer(self.sel_layer)
            self.sel_layer = None

    def _drawn(self, target, action, geo_json) -> None:
        """A patch was drawn or deleted on the map.

        Args:
            target: Target object or value for the operation.
            action: Input value for ``action`` used by this operation.
            geo_json: Input value for ``geo_json`` used by this operation.
        """
        if action == "deleted":
            self._clear_sel()
            self.bbox = None
            self.w_area.value = (
                "<i>2. Draw a rectangle or polygon on the map, or click a grid cell (areas above %d km&sup2; are split into tiles).</i>"
                % MAX_KM2
            )
            self._check()
            return
        if action not in ("create", "created"):
            return
        self._clear_sel()
        try:
            g = geo_json["geometry"]
            xs = [p[0] for p in g["coordinates"][0]]
            ys = [p[1] for p in g["coordinates"][0]]
            self.bbox = {"west": min(xs), "south": min(ys), "east": max(xs), "north": max(ys)}
            self.geom = g
            self.w_area.value = "Area: %.1f km&sup2;" % bbox_area_km2(**self.bbox)
            self._check()
            self._cal_here()
        except Exception as ex:
            self.w_area.value = f"<span style='color:#d73027'>cannot read the drawn area: {ex}</span>"

    def _check(self) -> None:
        """Check the patch size and enable the run button."""
        ok = bool(self.selected())
        if self.bbox:
            km2 = bbox_area_km2(self.bbox["west"], self.bbox["south"], self.bbox["east"], self.bbox["north"])
            n_t = len(tiles_for(self.bbox))
            if n_t > 1:
                colour = "#c60"
                msg = (
                    f"Area: {km2:,.0f} km&sup2;. <span style='color:{colour}'>It is larger than {MAX_KM2:.0f} km&sup2;, so it runs "
                    f"as <b>{n_t} tiles</b> (one openEO job each, {MAX_PARALLEL} at a time) and the tiles are shown together. "
                    f"More tiles use more credits and time.</span>"
                )
            else:
                msg = f"Area: {km2:.1f} km&sup2;"
            f = self.geo.get(self.country) if self.country else None
            if f:
                try:
                    from shapely.geometry import box, shape

                    if not shape(f["geometry"]).intersects(
                        box(self.bbox["west"], self.bbox["south"], self.bbox["east"], self.bbox["north"])
                    ):
                        msg += " <span style='color:#d73027'>(this area is outside the selected country)</span>"
                        ok = False
                except ImportError:
                    pass
            self.w_area.value = msg
        else:
            ok = False
        self.w_run.disabled = not ok
        if self.bbox and not ok and "red" not in self.w_area.value and "#d73027" not in self.w_area.value:
            why = (
                "choose a country"
                if not self.country
                else "choose a season"
                if not self.frame
                else "select at least one crop"
            )
            self.w_area.value += f" <span style='color:#c60'>(cannot run yet: {why})</span>"

    # ------------------------------------------------------------------ run
    def _run(self) -> None:
        """Submit the openEO job, wait for it and read the result.

        Raises:
            RuntimeError: If the operation cannot complete under the required conditions.
        """
        import openeo

        es = self.selected()
        year = int(self.w_year.value)
        self.w_run.disabled = True
        self.w_layer.disabled = True
        try:
            self.w_status.value = "Signing in to openEO (a browser window may open) ..."
            conn = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
            results = {}
            tiles = tiles_for(self.bbox)
            epsg = utm_epsg((self.bbox["west"] + self.bbox["east"]) / 2, (self.bbox["south"] + self.bbox["north"]) / 2)
            for gi, grp in enumerate(groups_for(es)):
                start, end, shifts = job_window(grp, year)
                names = ", ".join(PRETTY.get(e["crop"], e["crop"]) for e in grp)
                folder = os.path.join(
                    os.path.join(os.getcwd(), "results"), f"{self.country}_{self.frame}_{year}_{gi + 1}"
                )
                os.makedirs(folder, exist_ok=True)
                todo = list(enumerate(tiles))
                running, tifs = [], {}
                t0 = time.time()
                while todo or running:
                    while todo and len(running) < MAX_PARALLEL:
                        ti, tb = todo.pop(0)
                        self.w_status.value = (
                            f"Submitting {names}, tile {ti + 1} of {len(tiles)}, window {start} to {end} ..."
                        )
                        title = f"CropLogic {self.country} {self.frame} {year} tile {ti + 1}/{len(tiles)}"
                        running.append((ti, submit_job(conn, tb, start, end, grp, shifts, title, epsg=epsg)))
                    for ti, job in list(running):
                        st = retrying(job.status)
                        if st == "finished":
                            tdir = os.path.join(folder, f"tile_{ti + 1}")
                            os.makedirs(tdir, exist_ok=True)
                            retrying(lambda: job.get_results().download_files(tdir))
                            tifs[ti] = [
                                os.path.join(tdir, f) for f in os.listdir(tdir) if f.lower().endswith((".tif", ".tiff"))
                            ][0]
                            running.remove((ti, job))
                        elif st in ("error", "canceled"):
                            raise RuntimeError(
                                f"the openEO job of tile {ti + 1} ended with status {st} (job id {job.job_id}; open it in the openEO web editor for the log)"
                            )
                    self.w_status.value = (
                        f"{names}: <b>{len(tifs)} of {len(tiles)} tiles done</b>, {len(running)} running "
                        f"({int(time.time() - t0) // 60} min). You can leave this cell running."
                    )
                    if todo or running:
                        time.sleep(20)
                tif = merge_tifs([tifs[i] for i in sorted(tifs)], os.path.join(folder, "mosaic.tif"))
                r = read_result(tif, [e["crop"] for e in grp], start, int(grp[0]["cfg"]["window_len"]))
                results.update({"bounds": r["bounds"]})
                results.setdefault("crops", {}).update(r["crops"])
            self.job_result = results
            self._show_results()
            self.w_status.value = "<b>Done.</b> Files are in the <code>results</code> folder next to this notebook."
        except Exception as ex:
            self.w_status.value = f"<span style='color:#d73027'><b>Stopped:</b> {type(ex).__name__}: {ex}</span>"
        finally:
            self.w_run.disabled = False

    def _show_results(self) -> None:
        """Show the first result layer."""
        self.w_layer.disabled = False
        self.w_layer.value = "map"
        self._show_layer()
        b = self.job_result["bounds"]
        self.m.fit_bounds(b)

    def _show_layer(self) -> None:
        """Draw the chosen result layer on the map."""
        from ipyleaflet import ImageOverlay

        for o in self.overlays:
            self.m.remove_layer(o)
        self.overlays = []
        if not self.job_result:
            return
        kind = self.w_layer.value
        leg = []
        for crop, r in self.job_result["crops"].items():
            if r.get(kind):
                o = ImageOverlay(
                    url=r[kind], bounds=self.job_result["bounds"], name=f"{PRETTY.get(crop, crop)} ({kind})"
                )
                self.m.add_layer(o)
                self.overlays.append(o)
            ha, tot = r["area_ha"], r["cropland_ha"]
            leg.append(
                f"<span style='display:inline-block;width:12px;height:12px;background:{CROP_COLORS.get(crop, '#c00')};margin-right:5px'></span><b>{PRETTY.get(crop, crop)}</b>: {ha:,.0f} ha detected of {tot:,.0f} ha cropland in the patch"
                + (f"; first called {r['first']} to {r['last']}" if kind == "date" and r["first"] else "")
            )
        grad = {
            "map": "",
            "prob": "<br>probability 0% <span style='display:inline-block;width:150px;height:10px;background:linear-gradient(90deg,#f7fbff,#9ecae1,#4292c6,#084594)'></span> 100%",
            "date": "<br>earliest <span style='display:inline-block;width:150px;height:10px;background:linear-gradient(90deg,#ffeda0,#feb24c,#f03b20,#800026)'></span> latest detection",
        }[kind]
        self.w_legend.value = "<br>".join(leg) + grad
