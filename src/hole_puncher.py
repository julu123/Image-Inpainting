
import numpy as np
import math


class HolePuncher:

    _mask = None

    _flat_mask = None
    _flat_inv_mask = None

    def __init__(self, shape, rel_diameter=0.5):

        assert len(shape) == 3

        # determine hole properties
        width, height, *res_shape = shape
        radius = rel_diameter * min(width, height) / 2.0
        center = (width - 1.0) / 2.0, (height - 1.0) / 2.0

        self._mask = np.full(shape, False)

        for x in range(shape[0]):
            for y in range(shape[1]):

                # set mask values
                distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
                fill = np.full(shape=res_shape, fill_value=distance <= radius)
                self._mask[x, y] = fill

        # cache flattened masks
        self._flat_mask = self._mask.flatten()
        self._flat_inv_mask = np.invert(self._flat_mask)

    def fill(self, img):
        """
        Fills the hole with the mean pixel value

        :param img: numpy array of same shape as the HolePuncher was initialized
        """

        assert img.shape == self._mask.shape

        mean = np.mean(img)
        img[self._mask] = mean

    def split(self, img):
        """
        Splits the image into flattened arrays of hole and non-hole pixels

        :param img: numpy array of same shape as the HolePuncher was initialized
        :returns x, y where x are the non-hole pixels and y the hole pixels
        """

        assert img.shape == self._mask.shape

        flat_img = img.flatten()

        x = flat_img[self._flat_inv_mask]
        y = flat_img[self._flat_mask]

        return x, y

    def merge(self, x, y):
        """
        Reverse operation to 'split' method.
        Merges two flattened arrays of hole and non-hole pixels

        :returns numpy array of same shape as the HolePuncher was initialized
        """

        assert len(x) + len(y) == self._flat_mask.shape[0]

        img = np.zeros(self._flat_mask.shape, dtype=x.dtype)
        img[self._flat_inv_mask] = x
        img[self._flat_mask] = y

        img = img.reshape(self._mask.shape)

        return img