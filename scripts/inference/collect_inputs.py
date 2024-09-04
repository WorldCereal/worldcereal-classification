from openeo_gfmap import BoundingBoxExtent, TemporalContext

from worldcereal.job import collect_inputs


def main():

    # Set the spatial extent
    # bbox_utm = (664000.0, 5611120.0, 665000.0, 5612120.0)
    bbox_utm = (664000, 5611134, 684000, 5631134)  # Large test
    epsg = 32631
    spatial_extent = BoundingBoxExtent(*bbox_utm, epsg)

    # Set temporal range
    temporal_extent = TemporalContext(
        start_date="2020-11-01",
        end_date="2021-10-31",
    )

    outfile = "local_presto_inputs_large.nc"
    collect_inputs(spatial_extent, temporal_extent, output_path=outfile)


if __name__ == "__main__":
    main()
