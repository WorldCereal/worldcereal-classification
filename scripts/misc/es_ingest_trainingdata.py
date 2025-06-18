import argparse
from pathlib import Path
from worldcereal.utils.estypes import WordCerealTrainingData


def main(geoparquet_file: Path):
    td = WordCerealTrainingData.from_geoparquet(geoparquet_file)
    td.update()

if __name__ == "__main__":

    # Argparsing
    parser = argparse.ArgumentParser(description="Extract data from a collection")
    parser.add_argument(
        "geoparquet_file", type=Path, help="The path to the geoparquet"
    )
    args = parser.parse_args()

    main(args.geoparquet_file)

