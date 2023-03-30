import tempfile
import pytest
from worldcereal.worldcereal_products import run_tile


def test_run_tile(annualcropland_config):
    '''Just a functional test of the wrapper. Processing doesn't run!
    '''

    run_tile('31UFS',
             configfile=annualcropland_config,
             outputfolder=tempfile.TemporaryDirectory().name,
             process=False,
             postprocess=False,
             aez_id=None)


def test_force_aez(annualcropland_config):
    '''Just a functional test of the wrapper. Processing doesn't run!
    '''

    # aez_id 9000 does not exist, should raise an error
    with pytest.raises(ValueError):
        run_tile('31UFS',
                 configfile=annualcropland_config,
                 outputfolder=tempfile.TemporaryDirectory().name,
                 process=False,
                 postprocess=False,
                 aez_id=9000)

    # Force existing aez id
    # should log a warning
    run_tile('31UFS',
             configfile=annualcropland_config,
             outputfolder=tempfile.TemporaryDirectory().name,
             process=False,
             postprocess=False,
             aez_id=46172)
