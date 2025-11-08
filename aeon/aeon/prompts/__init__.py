import sys

from aeon.config import PROJECT_ROOT


MODULE = sys.modules[__name__]


class Prompt:

    def __init__(self, name: str):
        self.name = name
        self.prompt = getattr(MODULE, name)
        

    def __str__(self):
        return f"{type(self).__name__}(name={self.name})"