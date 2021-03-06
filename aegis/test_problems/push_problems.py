"""Real-world robot pushing problems: push4 and push8.

push4 is the "4-D ACTION" problem from [1]_, a four-dimensional robot pushing
task in which a robot has to push and object to an unknown target and receives
feedback, after completing its task, in the form of the distance from the
pushed object to the target. The robot is parametrised by its initial
location, the angle of its rectangular hand (used for pushing) and the number
of time-steps it pushes for.

push8 is an extension of the push4 problem in which there are two robots
pushing to targets to unknown target locations. The robots can block each
other and therefore the problem will likely be much harder than the push4
problem.

Note that once initialised (details in each problem definition) the problem
domain for each problem is mapped to [0, 1]^D because the time-step parameter
(in [0, 300]) is an order of magnitude larger than the initial robot position
(in [-5, 5]) and hand orientation [0, 2*numpy.pi].

References
----------
.. [1] Zi Wang and Stefanie Jegelka. 2017.
    Max-value entropy search for efficient Bayesian optimization.
    In Proceedings of the 34th International Conference on Machine Learning.
    PMLR, 3627–3635.
"""
import numpy as np
from .push_world import push_4D, push_8D


class push4:
    """Robot pushing simulation: One robot pushing one object.

    The object's initial location is at [0, 0] and the robot travels in the
    direction of the object's initial position. See paper for full problem
    details.

    Parameters
    ----------
    tx_1 : float
        x-axis location of the target, should reside in [-5, 5].
    ty_1 : float
        y-axis location of the target, should reside in [-5, 5].

    Examples
    --------
    >> f_class = push4
    >> # set initial target location (unknown to the robot and only
    >> # used for distance calculation after it has finished pushing)
    >> tx_1 = 3.5; ty_1 = 4
    >> # instantiate the test problem
    >> f = f_class(tx_1, ty_1)
    >> # evaluate some solution x in [0, 1]^4
    >> x = numpy.array([0.5, 0.7, 0.2, 0.3])
    >> f(x)
    array([9.0733461])
    """

    def __init__(self, t1_x=0, t1_y=0):
        self.dim = 4
        self.lb = np.array([-5, -5, 1, 0])
        self.ub = np.array([5, 5, 300, 2 * np.pi])

        # object target location
        self.t1_x = t1_x
        self.t1_y = t1_y

        # initial object location (0,0) as in Wang et al. (see module comment)
        self.o1_x = 0
        self.o1_y = 0

        # optimum location unknown as defined by inputs
        self.yopt = np.array([0.0])
        self.xopt = None

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        val = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            val[i, :] = push_4D(
                x[i, :], self.t1_x, self.t1_y, self.o1_x, self.o1_y, draw=False
            )

        return val.ravel()


class push8:
    """Robot pushing simulation: Two robots pushing an object each.

    The objects' initial locations are at [-3, 0] and [3, 0] respectively,
    with the robot 1 pushing the first target and robot 2 pushing the second
    target. See paper for full problem details.

    Parameters
    ----------
    tx_1 : float
        x-axis location for the target of robot 1, should reside in [-5, 5].
    ty_1 : float
        y-axis location for the target of robot 1, should reside in [-5, 5].
    tx_2 : float
        x-axis location for the target of robot 2, should reside in [-5, 5].
    ty_2 : float
        y-axis location for the target of robot 2, should reside in [-5, 5].

    Examples
    --------
    >> f_class = push8
    >> # initial positions (tx_1, ty_1) and (tx_2, ty_2) for both robots
    >> tx_1 = 3.5; ty_1 = 4
    >> tx_2 = -2; ty_2 = 1.5
    >> # instantiate the test problem
    >> f = f_class(tx_1, ty_1, tx_2, ty_2)
    >> # evaluate some solution x in [0, 1]^8
    >> x = numpy.array([0.5, 0.7, 0.2, 0.3, 0.3, 0.1, 0.5, 0.6])
    >> f(x)
    array([24.15719287])
    """

    def __init__(self, t1_x=-5, t1_y=-5, t2_x=5, t2_y=5):
        self.dim = 8
        self.lb = np.array([-5, -5, 1, 0, -5, -5, 1, 0])
        self.ub = np.array([5, 5, 300, 2 * np.pi, 5, 5, 300, 2 * np.pi])

        # object target locations
        self.t1_x = t1_x
        self.t1_y = t1_y
        self.t2_x = t2_x
        self.t2_y = t2_y

        # initial object locations (-3, 0) and (3, 0)
        self.o1_x = -3
        self.o1_y = 0
        self.o2_x = 3
        self.o2_y = 0

        # optimum location unknown as defined by inputs
        self.yopt = np.array([0.0])
        self.xopt = None

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        val = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            val[i, :] = push_8D(
                x[i, :],
                self.t1_x,
                self.t1_y,
                self.t2_x,
                self.t2_y,
                self.o1_x,
                self.o1_y,
                self.o2_x,
                self.o2_y,
                draw=False,
            )

        return val.ravel()
