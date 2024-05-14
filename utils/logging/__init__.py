from .logger import Logger, FileLogger


__global_logger = Logger()


def __tag_args(tag, *args):
    if tag:
        args = (f'[{tag}]',) + args
    return args


def log(*args, sep=" ", end="\n", tag=""):
    args = __tag_args(tag, *args)
    __global_logger(*args, sep=sep, end=end)


def log_warn(*args, sep=" ", end="\n", tag=""):
    args = __tag_args(tag, *args)
    log(*args, sep=sep, end=end, tag="WARNING")


class LogOnTaskComplete(object):
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        log("Starting...", tag=self.task_name)

    def __exit__(self, *args):
        log("Complete.", tag=self.task_name)
