"""Misc function/class decorators."""
from typing import Callable


def tab_completion(func: Callable):
    """Create class attributes from the results of some function. Helpful for creating minimal
    classes that mostly exist to provide tab completion.

    Arguments
    ---------
    func : callable
        A function that returns iterable[str]. Any periods in returned strings will be replaced by
        a double underscore '__' in the attribute name (the value will remain unchanged) because
        dots cannot be included in varnames. (We do not check for other invalid chars, we just do
        this because importable paths are a known use case for tab_completion.)

    Examples
    --------
    @tab_comletion(get_pet_names)
    class Pets:
        ...
    """
    values = func()
    def decorator(cls):
        for val in values:
            setattr(cls, val.upper().replace(".", "__"), val)
        return cls
    return decorator
