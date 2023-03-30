# coefficients derived from:
# Nedkov, R. (2017).
# ORTHOGONAL TRANSFORMATION OF SEGMENTED IMAGES FROM THE
# SATELLITE SENTINEL-2.
# Comptes rendus de l'Académie bulgare des sciences.
# 70. 687-692.


def tc_brightness(B01, B02, B03, B04, B05, B06, B07, B08,
                  B09, B10, B11, B12, B8A):

    result = (0.0356 * B01
              + 0.0822 * B02
              + 0.1360 * B03
              + 0.2611 * B04
              + 0.2964 * B05
              + 0.3338 * B06
              + 0.3877 * B07
              + 0.3895 * B08
              + 0.0949 * B09
              + 0.0009 * B10
              + 0.3882 * B11
              + 0.1366 * B12
              + 0.4750 * B8A)

    return result


def tc_greenness(B01, B02, B03, B04, B05, B06, B07, B08,
                 B09, B10, B11, B12, B8A):

    result = (- 0.0635 * B01
              - 0.1128 * B02
              - 0.1680 * B03
              - 0.3480 * B04
              - 0.3303 * B05
              + 0.0852 * B06
              + 0.3302 * B07
              + 0.3165 * B08
              + 0.0467 * B09
              - 0.0009 * B10
              - 0.4578 * B11
              - 0.4064 * B12
              + 0.3625 * B8A)

    return result


def tc_wetness(B01, B02, B03, B04, B05, B06, B07, B08,
               B09, B10, B11, B12, B8A):

    result = (0.0649 * B01
              + 0.1363 * B02
              + 0.2802 * B03
              + 0.3072 * B04
              + 0.5288 * B05
              + 0.1379 * B06
              - 0.0001 * B07
              - 0.0807 * B08
              - 0.0302 * B09
              + 0.0003 * B10
              - 0.4064 * B11
              - 0.5602 * B12
              - 0.1389 * B8A)

    return result
