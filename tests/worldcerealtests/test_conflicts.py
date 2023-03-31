from worldcereal.processors import CropTypeProcessor


def test_conflictresolver(worldcereal_outputs):

    tile = '30TXT'
    aez = 46172
    block_id = 6
    season = 'tc-maize-main'

    processor = CropTypeProcessor(
        worldcereal_outputs,
        models={},
        collections={},
        season=season,
        aez=aez,
        active_marker=True,
        end_date='2019-10-31'
    )

    processor.resolve_conflicts(tile, block_id)
