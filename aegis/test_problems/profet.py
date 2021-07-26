import os
import numpy as np


class ProfetFunction:
    def __init__(self, name, idx, has_cost=False):
        # only load this if we explicitly use the function
        from emukit.examples.profet import meta_benchmarks

        self.idx = idx
        self.name = name
        self.has_cost = has_cost
        self.f_class = getattr(meta_benchmarks, f"meta_{self.name}")

        # problems index from 0 but we number of problems from 1.
        self.idx -= 1

        # path to the directory that stores the objective (and cost) file
        path_base = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            f"profet_data/samples/{self.name}/",
        )

        # set up the arguments to the function
        args = {
            "fname_objective": f"{path_base}/sample_objective_{self.idx}.pkl",
            "noise": False,
        }
        if self.has_cost:
            args["fname_cost"] = f"{path_base}/sample_cost_{self.idx}.pkl"

        # instantiate the problem and its parameter space
        self.f, pspace = self.f_class(**args)

        # extract the problem dimensionality and problem bounds
        self.dim = pspace.dimensionality

        bounds = np.array(pspace.get_bounds())
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]

        # no optimal locations
        self.xopt = None
        self.yopt = np.array([0.0])

        self.cf = None

    def __call__(self, x, return_cost=False):
        # ensure it is 2d
        x = np.reshape(x, (-1, self.dim))

        # self.f returns a tuple of (function evaluation, cost)
        fx, cost = self.f(x)
        fx = fx.ravel()
        cost = cost.ravel()

        # typically we won't be getting the cost.
        if not return_cost:
            return fx

        return fx, cost


class svm(ProfetFunction):
    def __init__(self, problem_instance=1):
        ProfetFunction.__init__(self, "svm", problem_instance, True)


class fcnet(ProfetFunction):
    def __init__(self, problem_instance=1):
        ProfetFunction.__init__(self, "fcnet", problem_instance, True)


class xgboost(ProfetFunction):
    def __init__(self, problem_instance=1):
        ProfetFunction.__init__(self, "xgboost", problem_instance, True)


class forrester(ProfetFunction):
    def __init__(self, problem_instance=1):
        ProfetFunction.__init__(self, "forrester", problem_instance, False)
