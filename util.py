from contextlib import contextmanager


args = {}


# TODO: this doesn't work
@contextmanager
def define_args():
    global args

    before = {k: v for k, v in globals().items()}
    yield
    after = globals()
    args = {k: after[k] for k in set(after) - set(before)}
    print(args)


def get_args():
    return args
