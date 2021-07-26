import numpy as np


class timefunc:
    def __init__(self, name):
        self.name = name

    def __call__(self, n=1):
        raise NotImplementedError


class halfnorm(timefunc):
    def __init__(self, scale=np.sqrt(np.pi / 2)):
        timefunc.__init__(self, "halfnorm")

        self.scale = scale

    def __call__(self, n=1):
        return np.abs(np.random.normal(scale=self.scale, size=n))


class pareto(timefunc):
    def __init__(self, power=5.0):
        timefunc.__init__(self, "pareto")

        self.power = power
        self.offset = (power - 1.0) / power

    def __call__(self, n=1):
        return self.offset * (1.0 + np.random.pareto(a=self.power, size=n))


class exponential(timefunc):
    def __init__(self, rate=1.0):
        timefunc.__init__(self, "exponential")

        self.rate = rate
        self.scale = 1.0 / rate

    def __call__(self, n=1):
        return np.random.exponential(scale=self.scale, size=n)
