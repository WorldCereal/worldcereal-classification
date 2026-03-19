import glob
from pathlib import Path

import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject


def merge_maps(outdir: Path) -> dict[str, Path]:
    """Merge all product maps in the output directory into .tif files.

    Returns a mapping of product name -> merged output path.
    """

    if not outdir.exists():
        raise FileNotFoundError(f"Output directory {outdir} does not exist.")

    # Find all .tif files in the output directory
    tifs = glob.glob(str(outdir / "*" / "*.tif"))
    if len(tifs) == 0:
        raise FileNotFoundError("No tif files found in the output directory to merge.")

    product_groups: dict[str, list[str]] = {}
    for tif in tifs:
        product = Path(tif).name.split("_")[0]
        product_groups.setdefault(product, []).append(tif)

    merged_outputs: dict[str, Path] = {}

    def _merge_tifs(product_name: str, product_tifs: list[str]) -> Path:
        reprojected_tifs = []
        with rasterio.Env(CPL_LOG="ERROR"):
            for tif in product_tifs:
                # reproject to EPSG:3857 if not already in that CRS
                with rasterio.open(tif) as src:
                    dst_crs = "EPSG:3857"
                    transform, width, height = calculate_default_transform(
                        src.crs, dst_crs, src.width, src.height, *src.bounds
                    )

                    kwargs = src.meta.copy()
                    kwargs.update(
                        {
                            "crs": dst_crs,
                            "transform": transform,
                            "width": width,
                            "height": height,
                        }
                    )

                    memfile = MemoryFile()
                    with memfile.open(**kwargs) as dst:
                        for i in range(1, src.count + 1):
                            reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=dst_crs,
                                resampling=Resampling.nearest,
                            )
                        dst.descriptions = src.descriptions
                    reprojected_tifs.append(memfile.open())

            # Merge all reprojected rasters
            mosaic, out_trans = merge(reprojected_tifs)

            # Use metadata from one of the input files and update
            out_meta = reprojected_tifs[0].meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": "lzw",
                }
            )

            # Write to output
            outfile = outdir / f"{product_name}_merged.tif"
            with rasterio.open(outfile, "w", **out_meta) as dest:
                dest.write(mosaic)
                # Preserve band descriptions (if any)
                for idx, desc in enumerate(reprojected_tifs[0].descriptions, start=1):
                    if desc:
                        dest.set_band_description(idx, desc)

        return outfile

    for product_name, product_tifs in product_groups.items():
        merged_outputs[product_name] = _merge_tifs(product_name, product_tifs)

    return merged_outputs
