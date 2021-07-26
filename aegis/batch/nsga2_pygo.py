import torch
import pygmo as pg
import numpy as np


def NSGA2_pygmo(model, fevals, lb, ub, cf=None):
    """Finds the estimated Pareto front of a gpytorch model using NSGA2 [1]_.

    Parameters
    ----------
    model: gpytorch.models.ExactGP
        gpytorch regression model on which to find the Pareto front
        of its mean prediction and standard deviation.
    fevals : int
        Maximum number of times to evaluate a location using the model.
    lb : (D, ) torch.tensor
        Lower bound box constraint on D
    ub : (D, ) torch.tensor
        Upper bound box constraint on D
    cf : callable, optional
        Constraint function that returns True if it is called with a
        valid decision vector, else False.

    Returns
    -------
    X_front : (F, D) numpy.ndarray
        The F D-dimensional locations on the estimated Pareto front.
    musigma_front : (F, 2) numpy.ndarray
        The corresponding mean response and standard deviation of the locations
        on the front such that a point X_front[i, :] has a mean prediction
        musigma_front[i, 0] and standard deviation musigma_front[i, 1].

    Notes
    -----
    NSGA2 [1]_ discards locations on the pareto front if the size of the front
    is greater than that of the population size. We counteract this by storing
    every location and its corresponding mean and standard deviation and
    calculate the Pareto front from this - thereby making the most of every
    GP model evaluation.

    References
    ----------
    .. [1] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, and T. Meyarivan.
       A fast and elitist multiobjective genetic algorithm: NSGA-II.
       IEEE Transactions on Evolutionary Computation 6, 2 (2001), 182–197.
    """
    # internal class for the pygmo optimiser
    class GPYTORCH_WRAPPER(object):
        def __init__(self, model, lb, ub, cf, evals):
            # model = gpytorch model
            # lb = torch.tensor of lower bounds on X
            # ub = torch.tensor of upper bounds on X
            # cf = callable constraint function
            # evals = total evaluations to be carried out
            self.model = model
            self.lb = lb.numpy()
            self.ub = ub.numpy()
            self.nd = lb.numel()
            self.got_cf = cf is not None
            self.cf = cf
            self.i = 0  # evaluation pointer
            self.dtype = model.train_targets.dtype

        def get_bounds(self):
            return (self.lb, self.ub)

        def get_nobj(self):
            return 2

        def fitness(self, X):
            X = np.atleast_2d(X)
            X = torch.as_tensor(X, dtype=self.dtype)

            f = model_fitness(
                X,
                self.model,
                self.cf,
                self.got_cf,
                self.i,
                self.i + X.shape[0],
            )

            self.i += X.shape[0]
            return f.ravel()

        def has_batch_fitness(self):
            return True

        def batch_fitness(self, X):
            X = X.reshape(-1, self.nd)
            return self.fitness(X)

    # fitness function for the optimiser
    def model_fitness(X, model, cf, got_cf, start_slice, end_slice):
        n = X.shape[0]

        f = np.zeros((n, 2))
        valid_mask = np.ones(n, dtype="bool")

        # if we select a location that violates the constraint,
        # ensure it cannot dominate anything by having its fitness values
        # maximally bad (i.e. set to infinity)
        if got_cf:
            for i in range(n):
                if not cf(X[i]):
                    f[i] = [np.inf, np.inf]
                    valid_mask[i] = False

        if np.any(valid_mask):
            output = model(X[valid_mask])
            output = model.likelihood(
                output,
                noise=torch.full_like(
                    output.mean, model.likelihood.noise.mean()
                ),
            )

            # note the negative stdev here as NSGA2 is minimising
            # so we want to minimise the negative stdev
            f[valid_mask, 0] = output.mean.numpy()
            f[valid_mask, 1] = -np.sqrt(output.variance.numpy())

        # store every location ever evaluated
        model_fitness.X[start_slice:end_slice, :] = X
        model_fitness.Y[start_slice:end_slice, :] = f

        return f

    # get the problem dimensionality
    D = lb.numel()

    # NSGA-II settings
    POPSIZE = D * 100
    # -1 here because the pop is evaluated first before iterating N_GENS times
    N_GENS = int(np.ceil(fevals / POPSIZE)) - 1
    TOTAL_EVALUATIONS = POPSIZE * (N_GENS + 1)

    _nsga2 = pg.nsga2(
        gen=1,  # number of generations to evaluate per evolve() call
        cr=0.8,  # cross-over probability.
        eta_c=20.0,  # distribution index (cr)
        m=1 / D,  # mutation rate
        eta_m=20.0,  # distribution index (m)
    )

    # batch fitness evaluator -- this is the strange way we
    # tell pygmo that we have a batch_fitness method
    bfe = pg.bfe()

    # tell nsgaII about it
    _nsga2.set_bfe(bfe)
    nsga2 = pg.algorithm(_nsga2)

    # preallocate the storage of every location and fitness to be evaluated
    model_fitness.X = np.zeros((TOTAL_EVALUATIONS, D))
    model_fitness.Y = np.zeros((TOTAL_EVALUATIONS, 2))

    # problem instance
    gpytorch_problem = GPYTORCH_WRAPPER(model, lb, ub, cf, TOTAL_EVALUATIONS)
    problem = pg.problem(gpytorch_problem)

    # skip all gradient calculations as we don't need them
    with torch.no_grad():
        # initialise the population -- in batch (using bfe)
        population = pg.population(problem, size=POPSIZE, b=bfe)

        # evolve the population
        for i in range(N_GENS):
            population = nsga2.evolve(population)

    # indices non-dominated points across the entire NSGA-II run
    front_inds = pg.non_dominated_front_2d(model_fitness.Y)

    X_front = model_fitness.X[front_inds, :]
    musigma_front = model_fitness.Y[front_inds, :]

    # convert the standard deviations back to positive values; nsga2 minimises
    # the negative standard deviation (i.e. maximises the standard deviation)
    musigma_front[:, 1] *= -1

    # convert it to torch
    X_front = torch.as_tensor(X_front, dtype=model.train_targets.dtype)
    musigma_front = torch.as_tensor(
        musigma_front, dtype=model.train_targets.dtype
    )

    return X_front, musigma_front
