import numpy as np


def shift_image(
    img: np.ndarray[np.uint8],
    horizontal_shift: int,
    vertical_shift: int,
) -> np.ndarray[np.uint8]:
    """
    Function to shift an image by fixed numbers of pixels and fill in with 0.

    The image origin in numpy are at the top-left corner so to get a positive
    vertical shift, we need to negate the vertical shift value passed in.

    :param img: The input image
    :type img: np.ndarray[np.uint8]
    :param horizontal_shift: the number of pixels to shift horizontally
    :type horizontal_shift: int
    :param vertical_shift: the number of pixels to shift vertically
    :type vertical_shift: int
    :return: The shifted image
    :rtype: np.ndarray[np.uint8]
    """
    # Negate the vertical shift to compensate for the origin at the top-left of image
    vertical_shift = -vertical_shift
    shift_img = np.roll(img, vertical_shift, axis=0)
    shift_img = np.roll(shift_img, horizontal_shift, axis=1)
    if vertical_shift > 0:
        shift_img[:vertical_shift, :] = 0
    elif vertical_shift < 0:
        shift_img[vertical_shift:, :] = 0
    if horizontal_shift > 0:
        shift_img[:, :horizontal_shift] = 0
    elif horizontal_shift < 0:
        shift_img[:, horizontal_shift:] = 0
    return shift_img
