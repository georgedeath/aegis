import os
import torch
import botorch
import numpy as np


class UniformProblem:
    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim

        self.real_lb = problem.lb
        self.real_ub = problem.ub

        self.lb = np.zeros(self.dim)
        self.ub = np.ones(self.dim)

        if problem.xopt is not None:
            self.xopt = (problem.xopt - problem.lb) / (problem.ub - problem.lb)
        else:
            self.xopt = problem.xopt

        self.yopt = problem.yopt

        self.real_cf = problem.cf
        self.set_cf()

    def __call__(self, x):
        x = np.atleast_2d(x)

        # map x back to original space
        x = x * (self.real_ub - self.real_lb) + self.real_lb

        return self.problem(x)

    def set_cf(self):
        if self.real_cf is None:
            self.cf = None
            return

        def cf_wrapper(x):
            x = np.atleast_2d(x)

            # map x back to original space
            x = x * (self.real_ub - self.real_lb) + self.real_lb

            return self.real_cf(x)

        self.cf = cf_wrapper


class TorchProblem:
    def __init__(self, problem):
        self.problem = problem
        self.dim = problem.dim

        self.lb = torch.from_numpy(problem.lb)
        self.ub = torch.from_numpy(problem.ub)

        if problem.xopt is not None:
            self.xopt = torch.from_numpy(problem.xopt)
        else:
            self.xopt = problem.xopt

        self.yopt = torch.from_numpy(problem.yopt)

        if self.problem.cf is not None:

            def cf(x):
                if not isinstance(x, np.ndarray):
                    x = x.numpy()
                return self.problem.cf(x)

            self.cf = cf
        else:
            self.cf = None

    def __call__(self, x):
        fx = self.problem(x.numpy())
        torchfx = torch.from_numpy(fx)
        # cast to same datatype as x
        return torchfx.type(x.dtype)


class LowerConfidenceBound(botorch.acquisition.UpperConfidenceBound):
    def __init__(self, model, beta, objective=None, maximize=True):
        super().__init__(
            model=model, beta=beta, objective=objective, maximize=maximize
        )

    def forward(self, X):
        return -super().forward(X)


class PosteriorMean(botorch.acquisition.PosteriorMean):
    def __init__(self, model, objective=None, maximize=True):
        super().__init__(model=model, objective=objective)
        self.maximize = maximize

    def forward(self, X):
        output = super().forward(X)

        if not self.maximize:
            output *= -1.0

        return output


def acq_func_getter(name, model):
    acq_params = {"maximize": False}

    if name == "EI":
        acq_params["best_f"] = model.train_targets.min()
        acq_func = botorch.acquisition.ExpectedImprovement

    elif name == "UCB":
        acq_func = LowerConfidenceBound

        t = model.train_targets.numel()
        delta = 0.01
        D = model.train_inputs[0].shape[1]

        acq_params["beta"] = 2 * np.log(D * t ** 2 * np.pi ** 2 / (6 * delta))

    elif name == "PI":
        acq_func = botorch.acquisition.ProbabilityOfImprovement
        acq_params["best_f"] = model.train_targets.min()

    elif name == "mean":
        acq_func = PosteriorMean

    return acq_func(model, **acq_params)


def generate_save_filename(
    time_name,
    problem_name,
    budget,
    n_workers,
    acq_name,
    run_no,
    problem_params={},
    acq_params={},
    repeat_no=None,
    results_dir="results",
):
    # append dim if different from default
    if "d" in problem_params:
        problem_name = f'{problem_name:s}{problem_params["d"]:d}'

    if "aegis" in acq_name:
        epsilon = acq_params["epsilon"]
        try:
            eta = acq_params["eta"]
        except KeyError:
            eta = 0.5

        try:
            epsilon = f"{float(epsilon):0.2f}"
        except ValueError:
            pass
        acq_name = f"{acq_name:s}-{epsilon:s}"

        if eta != 0.5:
            acq_name += f"-{float(eta):g}"

    if "BatchBO" in acq_name:
        acq_name = f'{acq_name:s}-{acq_params["acq_name"]:s}'

    fname_components = [
        "async",
        f"_{time_name:s}",
        f"_{budget:d}",
        f"_workers={n_workers:d}",
        f"_{acq_name:s}",
        f"_{problem_name:s}",
        f"_run={run_no:03d}",
        f"-{repeat_no:d}" if repeat_no is not None else "",
        ".pt",
    ]

    fname = "".join(fname_components)

    return os.path.join(results_dir, fname)


def generate_data_filename(
    problem_name, run_no, problem_params={}, repeat_no=None, data_dir="data",
):
    # append dim if different from default
    if "d" in problem_params:
        problem_name = f'{problem_name:s}{problem_params["d"]:d}'

    fname_components = [
        f"{problem_name:s}",
        f"_{run_no:03d}",
        f"-{repeat_no:d}" if repeat_no is not None else "",
        ".pt",
    ]

    fname = "".join(fname_components)

    return os.path.join(data_dir, fname)


# class for storing which vectors are under evaluation
# need method to remove a completed vector from it
# one to return the list under evaluation
class UnderEval:
    """
    TODO: Write docstring!

    Notes:
        Duplicate vectors are allowed to be added. If there two
        identical vectors x1 and x2 stored, then calling remove(x1)
        will only remove one of them.
    """

    def __init__(self, max_under_eval, dim, dtype):
        self.max = max_under_eval
        self.dim = dim

        # mask of those under eval
        self.free_mask = torch.ones(self.max, dtype=torch.bool)

        # storage
        self.data = torch.zeros((self.max, dim), dtype=dtype)

    def _get_next_free(self):
        # get the next free index, based on the mask
        free = torch.nonzero(self.free_mask, as_tuple=True)[0]

        if len(free) == 0:
            msg = "No free slots to store data."
            msg += f" Already at a maximum capacity of {self.max}."
            raise ValueError(msg)

        return free[0].item()

    def _check_vector(self, x):
        if x.ndim == 2 and x.shape[0] > 1:
            err = "Only one vector can dealt with at a time."
            raise ValueError(err)

        if x.shape[-1] != self.dim:
            err = f"The vector's dimensionality ({x.shape[-1]})"
            err += f" does not match the expected size ({self.dim})"
            raise ValueError(err)

        return x.flatten()

    def add(self, x):
        x = self._check_vector(x)

        # add x to the data array
        idx = self._get_next_free()
        self.data[idx] = x
        self.free_mask[idx] = False

    def remove(self, x):
        self._check_vector(x)

        # removes x from the data array, only check the array slots
        # marked as not being free
        location = torch.where(
            torch.all(self.data[~self.free_mask] == x, axis=1)
        )

        if len(location) == 0:
            msg = f"Vector not stored in this object: \n\t{x}"
            raise ValueError(msg)

        # get the first location that matches -- note that this index
        # will be in terms of the False values in self.free_mask, and so
        # may not correspond to the entire mask.
        masked_idx = location[0][0].item()

        # map the masked index back to the mask's shape
        idx = torch.arange(self.max)[~self.free_mask][masked_idx]

        # sanity check
        assert self.free_mask[idx].item() is False

        self.free_mask[idx] = True

    def get(self):
        # returns the locations currently under evaluation
        return self.data[~self.free_mask]
