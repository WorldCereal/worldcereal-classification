from loguru import logger
import pandas as pd


def irr_max_ndvi_filter(df: pd.DataFrame) -> pd.DataFrame:
    '''Function to filter irr samples based on max NDVI
    '''
    if df is None:
        return None

    ref_ids = ['2019_GLO_irr-eleaf', '2021_GLO_irr-eleaf-asc',
               '2021_GLO_irr-eleaf-desc']

    if 'OPTICAL-ndvi-p90-10m' not in df.columns:
        raise ValueError('Required NDVI feature'
                         ' for pixel filtering missing!')
    ignorelabels = [10000, 0]
    drop_idx = df[(df['OPTICAL-ndvi-p90-10m'] < 0.4) &
                  (~df['IRR'].isin(ignorelabels)) &
                  (df['ref_id'].isin(ref_ids))].index
    irrpixels = df[~df['IRR'].isin(ignorelabels)]
    logger.warning(f'Dropping {len(drop_idx)}/{len(irrpixels)} IRR pixels!')

    df_filtered = df.drop(drop_idx)

    return df_filtered
