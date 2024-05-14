import numpy as np


class LoopingIterator(object):
    """
    looping iterator of distinguishable colors
    """
    def __init__(self, items: (list, tuple) = None):
        self._crt = 0
        self._items = [None] if items is None else items

    def __iter__(self):
        return self

    def __next__(self):
        item = self._items[self._crt]
        self._crt = (self._crt + 1) % len(self._items)
        return item

    def reset(self):
        self._crt = 0

    def next(self):
        return self.__next__()


class ColorIterator(LoopingIterator):
    """
    looping iterator of distinguishable RGB colors (normalized to 0-1)
    """
    def __init__(self, colors: (tuple, list) = None):
        """
        :param colors: array-like (N, 3) 0-255 RGB color values; will be converted to 0-1 RGB internally.
        """
        if colors is None:
            colors = [
                (0, 0, 255),
                (255, 0, 0),
                (0, 255, 0),
                (0, 155, 255),
                (255, 155, 0),
                (150, 150, 0),
                (255, 0, 255),
                (230, 25, 75),
                (60, 220, 60),
                (0, 130, 200),
                (245, 130, 48),
                (145, 30, 180),
                (70, 240, 240),
                (240, 50, 230),
                (210, 245, 60),
                (250, 190, 190),
                (0, 128, 128),
                (230, 190, 255),
                (170, 110, 40),
                (255, 250, 200),
                (128, 0, 0),
                (170, 255, 195),
                (128, 128, 0),
                (255, 215, 180),
                (255, 225, 25),
                (0, 0, 128),
                (128, 128, 128),
                (0, 0, 0)
            ]

        colors = np.array(colors, float) / 255
        colors = [colors[i] for i in range(colors.shape[0])]
        super().__init__(items=colors)
