from datetime import datetime


def format_msg(*args, sep=" ", end="\n", timestamp=True):
    msg = sep.join([str(arg) for arg in args]) + end
    if timestamp:
        current_time = datetime.now().strftime("%H:%M:%S")
        msg = "[{}] {}".format(current_time, msg)
    return msg


class Logger(object):
    def __init__(self,
                 verbose=True
                 ):
        self.verbose = verbose

    def __call__(self, *args, sep=" ", end="\n"):
        msg = format_msg(*args, sep=sep, end=end)
        if self.verbose:
            print(msg, sep="", end="")
        return msg


class FileLogger(Logger):
    def __init__(self, file_path, verbose=True):
        super().__init__(verbose)
        self.file_path = file_path

    def __call__(self, *args, sep=" ", end="\n"):
        msg = super(FileLogger, self).__call__(*args, sep=sep, end=end)
        if self.file_path is not None:
            with open(self.file_path, 'a') as file:
                file.write(msg)
