from pathlib import Path
import argparse
from loguru import logger

from worldcereal.cib.database import (DataBase, _PRODUCTS)


def main(cib_experiment_id, cib_rootdir, jsonfiles, spark):

    experiment_dir = str(Path(cib_rootdir) / cib_experiment_id)

    # Setup the CIB database
    db = DataBase.from_folder(experiment_dir, jsonfiles=jsonfiles,
                              spark=spark)

    # Read in all available CIB sample files
    db.populate()

    # Add root datapath
    db.add_root()

    # # Add the AEZ information
    # NOTE: this information is now added during feature computation
    # db.add_aez('/data/worldcereal/data/AEZ/AEZ.geojson')

    # Look for the data paths
    db.add_products(products=_PRODUCTS, remove_errors=False)

    # Check for issues
    issues, l8issues = db.check_issues(products=_PRODUCTS)
    issues['source'] = db.database.loc[issues.index]['source']
    issues.to_json(
        str(Path(cib_rootdir) / cib_experiment_id / 'issues.json'))

    # Remove issues from database
    db.drop(issues.index)

    # add attribute signaling l8 problem
    db.add_l8(l8issues)

    # Write to output file
    db.to_json(str(
        Path(cib_rootdir) / cib_experiment_id / 'database.json'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cib', type=str, required=True,
                        help='CIB experiment ID')
    parser.add_argument('--cibroot', type=str, required=True,
                        help='CIB root directory')
    parser.add_argument('-s', '--spark',
                        action="store_true",  help='Run on spark')

    args = parser.parse_args()
    jsonfiles = None

    # Get the parameters
    cib_experiment_id = args.cib
    cib_rootdir = args.cibroot
    spark = args.spark

    # # for debugging
    # cib_experiment_id = 'CIB_V1'
    # cib_rootdir = '/data/worldcereal/cib'
    # spark = False
    # jsonfiles = ['/data/worldcereal/cib/CIB_V1/POLY/2019_EG_WAPOR-1/2019_EG_WAPOR-1_POLY_111_samples.json',
    #              #  '/data/worldcereal/cib/CIB_V1/POLY/2018_BE_LPIS-Flanders/2018_BE_LPIS-Flanders_POLY_110_samples.json'
    #              ]

    # Log some information for this run
    logger.info('-'*80)
    logger.info('USING FOLLOWING SETTINGS:')
    logger.info(f'cib_experiment_id: {cib_experiment_id}')
    logger.info(f'cib_rootdir: {cib_rootdir}')
    logger.info(f'Use spark: {spark}')
    logger.info('-'*80)

    main(cib_experiment_id, cib_rootdir, jsonfiles, spark)
