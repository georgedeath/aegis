import torch
from .nsga2_pygo import NSGA2_pygmo


class BaseBatchBO:
    """Batch Bayesian optimisation class.

    This provides the functionality to take in a trained gpytorch Gaussian
    process model and, when the method get_next(q) is called, return q
    locations to expensively evaluate.

    Methods implementing this interface should work the same if q=10 locations
    are requested at once or q=1 location is requested 10 times.

    Parameters
    ----------
    model: gpytorch.models.ExactGP
        Gaussian process model.

    lb, ub: torch.tensor
        Vector of lower and upper bound values of the box constraint on the
        problem domain.

    under_evaluation: torch.tensor (n x d) or None
        Locations that are currently under evaluation. Defaults to None,
        meaning that there are no locations under evaluation. Otherwise it is
        expected to be a tensor containing n d-dimensional locations.
    """

    def __init__(self, model, lb, ub, under_evaluation=None):
        self.model = model
        self.lb = lb
        self.ub = ub
        self.ue = under_evaluation

        # problem dimensionality
        self.dim = lb.numel()

        # training data datatype
        self.dtype = model.train_targets.dtype

        # was the model just updated -- for lazy instantiation
        self.updated = True

    def update(self, model, under_evaluation):
        """
        Updates the acquisition function with the latest model and locations
        under evaluation.
        """
        self.model = model
        self.ue = under_evaluation
        self.updated = True

    def _get_next(self):
        """
        This method should get one location to evaluate, and adjust the GP
        model accordingly (if necessary).
        """
        raise NotImplementedError

    def get_next(self, q=1):
        """Get q locations to expensively evaluate.

        Parameters
        ----------
        q: int, optional
            Number of locations to evaluate (Default 1).

        Returns
        -------
        x_locations: torch.tensor, shape (q, d)
            The q location(s) at which to evaluate.
        """
        res = torch.zeros(q, self.dim, dtype=self.dtype)

        for i in range(q):
            res[i] = self._get_next()

        return res


class SelectionRandom:
    def __init__(self):
        pass

    def _rand_selection(self):
        # return a random location in [lb, ub]
        return (
            torch.rand(size=(self.dim,), dtype=self.dtype)
            * (self.ub - self.lb)
            + self.lb
        )


class SelectionParetoFront:
    def __init__(self):
        self._pf = None

    @property
    def pf(self):
        # if we've got a new (updated) model, get a new pareto front
        if self.updated or self._pf is None:
            self.updated = False
            self._pf, _ = NSGA2_pygmo(
                self.model, self.n_opt_samples, self.lb, self.ub, cf=None
            )

        return self._pf

    def _rand_selection(self):
        # return a random location on the Pareto front
        return self.pf[torch.randint(self.pf.shape[0], size=(1,))]
