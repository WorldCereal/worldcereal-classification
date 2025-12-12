import openeo
from skimage.morphology import footprints


def convolve(img, radius):
    """OpenEO method to apply convolution
    with a circular kernel of `radius` pixels.
    NOTE: make sure the resolution of the image
    matches the expected radius in pixels!
    """
    kernel = footprints.disk(radius)
    img = img.apply_kernel(kernel)
    return img


def scl_mask_erode_dilate(
    scl_cube: openeo.DataCube,
    erode_r: int = 3,
    dilate_r: int = 21,
    target_crs=None,
):
    """OpenEO method to construct a Sentinel-2 mask based on SCL.
    It involves an erosion step followed by a dilation step.

    Args:
        ...
        erode_r (int, optional): Erosion radius (pixels). Defaults to 3.
        dilate_r (int, optional): Dilation radius (pixels). Defaults to 13.

    Returns:
        DataCube: DataCube containing the resulting mask
    """

    first_mask = scl_cube == 0
    for mask_value in [1, 3, 8, 9, 10, 11]:
        first_mask = (first_mask == 1) | (scl_cube == mask_value)

    # Invert mask for erosion
    first_mask = first_mask.apply(lambda x: (x == 1).not_())

    # Apply erosion by dilation the inverted mask
    erode_cube = convolve(first_mask, erode_r)

    # Invert again
    erode_cube = erode_cube > 0.1
    erode_cube = erode_cube.apply(lambda x: (x == 1).not_())

    # Now dilate the mask
    dilate_cube = convolve(erode_cube, dilate_r)

    # Get binary mask. NOTE: >0.1 is a fix to avoid being triggered
    # by small non-zero oscillations after applying convolution
    dilate_cube = dilate_cube > 0.1

    return dilate_cube
