from datetime import datetime


# need this because Logger must return msg not just print it
def format_msg(*args, sep=" ", end="\n", timestamp=True):
    msg = sep.join([str(arg) for arg in args]) + end
    if timestamp:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg = f"[{current_time}] {msg}"
    return msg


class Logger(object):
    def __init__(self,
                 verbose=True,
                 timestamp=True
                 ):
        self.verbose = verbose
        self.timestamp = timestamp

    def __call__(self, *args, sep=" ", end="\n"):
        msg = format_msg(*args, sep=sep, end=end, timestamp=self.timestamp)
        if self.verbose:
            print(msg, sep="", end="")
        return msg


class FileLogger(Logger):
    def __init__(self, file_path, **kwargs):
        super().__init__(**kwargs)
        self.file_path = file_path

    def __call__(self, *args, sep=" ", end="\n"):
        msg = super(FileLogger, self).__call__(*args, sep=sep, end=end)
        if self.file_path is not None:
            with open(self.file_path, 'a') as file:
                file.write(msg)
        return msg
