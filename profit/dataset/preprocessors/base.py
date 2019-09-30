
class BasePreProcessor(object):
    """Base class for preprocessing."""

    def __init__(self):
        pass

    def process(self, filepath):
        raise NotImplementedError