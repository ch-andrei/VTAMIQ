import numpy as np


class ColorIterator:
    """
    looping iterator of distinguishable colors
    """
    def __init__(self):
        self._crt = 0

        self._colors = []

        # define colors conveniently as 0-255 RGB tuples in this list
        # will be converted to 0-1 RGB
        color_to_add = [
            (0, 155, 255),
            (255, 155, 0),
            (0, 0, 255),
            (150, 150, 0),
            (255, 0, 0),
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

        for r, g, b in color_to_add:
            self.__add_color__(r, g, b)

    def __add_color__(self, r, g, b):
        self._colors.append(np.array((r, g, b), np.float32) / 255)

    def __iter__(self):
        return self

    def __next__(self):
        color = self._colors[self._crt]
        self._crt = (self._crt + 1) % len(self._colors)
        return color

    def reset(self):
        self._crt = 0

    def next(self):
        return self.__next__()
