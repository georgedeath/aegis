import torch
import botorch
from .base import SelectionParetoFront, SelectionRandom
from .ThompsonSampling import BatchTS
from ..util import acq_func_getter


def epsilon_chooser(epsilon, dim):
    # ensure dim >= 4
    dim = 4 if dim < 4 else dim

    dim = torch.tensor(float(dim))

    if epsilon == "d":
        epsilon = 1.0 / dim

    elif epsilon == "sqrtd":
        epsilon = 1.0 / torch.sqrt(dim)

    elif epsilon == "dtake2":
        epsilon = 1.0 / (dim - 2.0)

    elif epsilon == "logdplus3":
        epsilon = 1.0 / torch.log(dim + 3.0)

    else:
        epsilon = torch.tensor(epsilon)

    # clip epsilon to always being <= 0.5
    return torch.min(epsilon, torch.tensor(0.5))


class aegisBase(BatchTS):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        epsilon,
        n_features,
        n_opt_samples,
        n_opt_bfgs,
    ):

        BatchTS.__init__(
            self,
            model,
            lb,
            ub,
            under_evaluation,
            n_features,
            n_opt_samples,
            n_opt_bfgs,
        )

        self.epsilon = epsilon_chooser(epsilon, self.dim)

    def _rand_selection(self):
        raise NotImplementedError

    def _get_next(self):
        # random selection epsilon proportion of the time
        if torch.rand(1) < self.epsilon:
            return self._rand_selection()

        # greedily optimise a sample path
        return BatchTS._get_next(self)


class aegisExploitBase(aegisBase):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        epsilon,
        eta,
        n_workers,
        n_features,
        n_opt_samples,
        n_opt_bfgs,
    ):

        aegisBase.__init__(
            self,
            model,
            lb,
            ub,
            under_evaluation,
            epsilon,
            n_features,
            n_opt_samples,
            n_opt_bfgs,
        )

        # number of workers
        self.n_workers = n_workers

        # current optimisation step
        self.t = 0

        # proportion of times to do TS (within the 1-e of times)
        self.eta = eta

        # exploitative acquisition function and its bounds
        self.acq_name = "mean"
        self.problem_bounds = torch.stack((self.lb, self.ub)).type(self.dtype)

    def _exploit_model(self):
        # acquisition function
        acq_func = acq_func_getter(self.acq_name, self.model)

        xnew, _ = botorch.optim.optimize_acqf(
            acq_function=acq_func,
            q=1,
            bounds=self.problem_bounds,
            num_restarts=self.n_opt_bfgs,
            raw_samples=self.n_opt_samples,
        )

        return xnew

    def _rand_selection(self):
        raise NotImplementedError

    def _get_next(self):
        # if epsilon < 0.5 then always exploit on the first iteration
        # then only exploit on iterations after some measurements have come in,
        # and then only (1 - 2 * epsilon) proportion of the time
        if ((self.t == 0) and (self.epsilon < 0.5)) or (
            (self.t > self.n_workers)
            and (torch.rand(1) < (1.0 - 2.0 * self.epsilon))
        ):
            print("Exploiting")
            xnew = self._exploit_model()

        # else roll again because we either TS or explore with equal
        # probability at this point.
        else:
            if torch.rand(1) < self.eta:
                print("Sample path selection")
                xnew = BatchTS._get_next(self)
            else:
                print("Random selection")
                xnew = self._rand_selection()

        self.t += 1
        return xnew


class aegisExploitRandom(SelectionRandom, aegisExploitBase):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        epsilon,
        eta,
        n_workers,
        n_features,
        n_opt_samples,
        n_opt_bfgs,
    ):
        aegisExploitBase.__init__(
            self,
            model,
            lb,
            ub,
            under_evaluation,
            epsilon,
            eta,
            n_workers,
            n_features,
            n_opt_samples,
            n_opt_bfgs,
        )

        SelectionRandom.__init__(self)


class aegisExploitParetoFront(SelectionParetoFront, aegisExploitBase):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        epsilon,
        eta,
        n_workers,
        n_features,
        n_opt_samples,
        n_opt_bfgs,
    ):
        aegisExploitBase.__init__(
            self,
            model,
            lb,
            ub,
            under_evaluation,
            epsilon,
            eta,
            n_workers,
            n_features,
            n_opt_samples,
            n_opt_bfgs,
        )

        SelectionParetoFront.__init__(self)
