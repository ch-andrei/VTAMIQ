import numpy as np


class TemporaryNumpySeed(object):
    # on enter, set a given random state; on exit, set previous random state
    def __init__(self, seed=None):
        self.seed_temp = seed  # if None, will not modify random states
        self.seed_prev = None

    def __enter__(self):
        if self.seed_temp is not None:
            self.seed_prev = np.random.get_state()
            np.random.seed(self.seed_temp)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed_temp is not None:
            np.random.set_state(self.seed_prev)
