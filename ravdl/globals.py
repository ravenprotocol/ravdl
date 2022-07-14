from .utils_singleton import Singleton

@Singleton
class Globals(object):
    def __init__(self):
        self._zero = None
        self._half = None
        self._one = None
        self._two = None


    @property
    def zero(self):
        return self._zero

    @zero.setter
    def zero(self, zero):
        self._zero = zero

    @property
    def half(self):
        return self._half

    @half.setter
    def half(self, half):
        self._half = half

    @property
    def one(self):
        return self._one

    @one.setter
    def one(self, one):
        self._one = one
    
    @property
    def two(self):
        return self._two

    @two.setter
    def two(self, two):
        self._two = two


globals = Globals.Instance()
