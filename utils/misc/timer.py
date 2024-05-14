import time


# Timer counting seconds since start.
# can be reused multiple times and get average runtime by repeatedly calling start() + stop().
# can be paused by calling pause() between start() and stop().
class Timer:
    def __init__(self, name="", start=False, verbose=True):
        self.name = "Timer-" + name if name else f"Timer-{str(time.time())}"
        self.start_time = None
        self.paused = False
        self.stopped = False
        self.paused_time = None
        self.total_paused = 0
        self.delta_avg = 0
        self.delta_min = 1e9
        self.delta_max = -1e9
        self.use_count = 0
        self.delta = 0
        self.deltas = []
        self.verbose = verbose
        if start:
            self.start()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        delta = self.stop()
        if self.verbose:
            print(f"Timer {self.name} delta={delta} sec.")

    def reset(self):
        self.__init__(name=self.name, start=False)

    def restart(self):
        self.paused = False
        self.stopped = False
        self.paused_time = None
        self.total_paused = 0
        # self.use_count = 0

    def start(self):
        self.restart()
        self.start_time = time.time()

    def stop_start(self):
        self.stop()
        self.start()

    def pause(self):
        if not self.paused:
            self.paused = True
            self.paused_time = time.time()

    def unpause(self):
        if self.paused:
            self.total_paused += time.time() - self.paused_time
            self.paused_time = None
        self.paused = False

    def total(self):
        return sum(self.deltas)

    def stop(self):
        if not self.stopped:
            self.stopped = True
            self.delta = time.time() - self.start_time - self.total_paused
            self.deltas.append(self.delta)
            if self.delta_max < self.delta:
                self.delta_max = self.delta
            if self.delta_min > self.delta:
                self.delta_min = self.delta
            self.delta_avg += 1.0 / (self.use_count + 1) * (self.delta - self.delta_avg)
            self.use_count += 1
        return self.delta

    def time_runs(self, num_runs, function: callable, *args, **kwargs):
        with self:
            for i in range(num_runs):
                function(*args, **kwargs)