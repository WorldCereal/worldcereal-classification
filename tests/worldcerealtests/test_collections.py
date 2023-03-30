import pytest
from worldcereal.utils.io import check_collection, CollectionError
from worldcereal.utils import get_coll_maxgap


def test_WorldCerealOpticalTiledCollection_filternodata(
        WorldCerealOpticalTiled_collection):
    '''Mind that at the moment optical L8 is disabled
    and hence automatically removed from the test collection!
    '''

    origsize = len(WorldCerealOpticalTiled_collection.df)

    # With min_keep_fraction of 1, all acquisitions should be retained
    keep_all = WorldCerealOpticalTiled_collection.filter_nodata(
        min_keep_fraction=1, min_acquisitions=5)

    assert len(keep_all.df) == origsize

    # With default settings, 2 acquisitions should be dropped
    keep_default = WorldCerealOpticalTiled_collection.filter_nodata(
        min_acquisitions=5
    )
    assert len(keep_default.df) == origsize - 2

    # With a mask_th of 0.9, one additional acquisition should be dropped
    # Need to increase min_keep_fraction otherwise this does not happen
    keep_conservative = WorldCerealOpticalTiled_collection.filter_nodata(
        mask_th=0.9, min_acquisitions=5, min_keep_fraction=0.5)
    assert len(keep_conservative.df) == origsize - 3

    # With a mask_th of 0.9, but a min_keep_fraction of 0.90 as well,
    # we're back at dropping just 2 acquisitions
    less_conservative = WorldCerealOpticalTiled_collection.filter_nodata(
        mask_th=0.9, min_acquisitions=5, min_keep_fraction=0.9)
    assert len(less_conservative.df) == origsize - 2


def test_WorldCerealThermalTiledCollection_filternodata(
        WorldCerealThermalTiled_collection):

    origsize = len(WorldCerealThermalTiled_collection.df)

    # With min_keep_fraction of 1, all acquisitions should be retained
    keep_all = WorldCerealThermalTiled_collection.filter_nodata(
        min_keep_fraction=1,
        min_acquisitions=4)

    assert len(keep_all.df) == origsize


def test_collectionfailure(WorldCerealOpticalTiled_collection):
    '''Test if a collection of < 2 acquisitions raises the expected
    error
    '''

    # Check with exactly one item
    WorldCerealOpticalTiled_collection.df = WorldCerealOpticalTiled_collection.df.iloc[0:1, :]  # NOQA

    try:
        check_collection(WorldCerealOpticalTiled_collection, 'OPTICAL',
                         '2018-10-01', '2018-10-20',
                         fail_threshold=get_coll_maxgap('OPTICAL'))
    except CollectionError as e:
        print(f'Successfully raised exception: {e}')

    # Check with no items
    WorldCerealOpticalTiled_collection.df = WorldCerealOpticalTiled_collection.df.iloc[0:0, :]  # NOQA

    try:
        check_collection(WorldCerealOpticalTiled_collection, 'OPTICAL',
                         '2018-10-01', '2018-10-20',
                         fail_threshold=get_coll_maxgap('OPTICAL'))
    except CollectionError as e:
        print(f'Successfully raised exception: {e}')


@pytest.mark.xfail
def test_L8ThermalTiledCollection_filternodata(
        L8ThermalTiled_collection):
    '''Fails on Jenkins because collection location
    cannot be reached
    '''

    origsize = len(L8ThermalTiled_collection.df)

    # With min_keep_fraction of 1, all acquisitions should be retained
    keep_all = L8ThermalTiled_collection.filter_nodata(
        min_keep_fraction=1,
        min_acquisitions=5)

    assert len(keep_all.df) == origsize

    keep_default = L8ThermalTiled_collection.filter_nodata(
        min_acquisitions=5
    )
    assert len(keep_default.df) == 62

    keep_39 = L8ThermalTiled_collection.filter_nodata(min_keep_fraction=0.5,
                                                      min_acquisitions=5)
    assert len(keep_39.df) == 39
