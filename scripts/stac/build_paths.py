"""From a folder of extracted patches, generates a list of paths to the patches
to be later parsed by the Spark cluster.
"""

import argparse
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from worldcereal.stac.constants import COLLECTION_REGEXES, ExtractionCollection


def iglob_files(path, notification_folders, pattern=None):
    """
    Generator that finds all subfolders of path containing the regex `pattern`
    """
    root_dir, folders, files = next(os.walk(path))

    for f in files:

        if (pattern is None) or len(re.findall(pattern, f)):
            file_path = os.path.join(root_dir, f)
            yield file_path

    for d in folders:
        # If the folder is in the notification folders list, print a message
        if os.path.join(path, d) in notification_folders:
            print(f"Searching in {d}")
        new_path = os.path.join(root_dir, d)
        yield from iglob_files(new_path, notification_folders, pattern)


def glob_files(path, notification_folders, pattern=None, threads=50):
    """
    Return all files within path and subdirs containing the regex `pattern`
    """
    with ThreadPoolExecutor(max_workers=threads) as ex:
        files = list(
            ex.map(lambda x: x, iglob_files(path, notification_folders, pattern))
        )

    return files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a list of paths of patches to parse."
    )

    parser.add_argument(
        "collection",
        type=ExtractionCollection,
        choices=list(ExtractionCollection),
        help="The collection to extract",
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="The path to the folder containing the extracted patches.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="The path to the pickle file where the paths will be saved.",
    )

    args = parser.parse_args()

    # Pattern to filter the files that are sentinel-1 patches
    pattern = COLLECTION_REGEXES[args.collection]

    root_subfolders = [
        str(path) for path in args.input_folder.iterdir() if path.is_dir()
    ]

    files = glob_files(
        path=args.input_folder,
        notification_folders=root_subfolders,
        pattern=pattern,
    )

    with open(args.output_path, "wb") as f:
        pickle.dump(files, f)
