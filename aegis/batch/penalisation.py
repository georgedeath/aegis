import botorch
import torch
import scipy

from ..util import acq_func_getter
from .acquisitions import AcqBaseBatchBO


class PenalisationBaseBatchBO(AcqBaseBatchBO):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        acq_name,
        n_opt_samples,
        n_opt_bfgs,
    ):
        AcqBaseBatchBO.__init__(
            self,
            model,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

        self._acq = None

        if (under_evaluation is not None) and (under_evaluation.numel() != 0):
            self._add_locations(under_evaluation)

    def _get_penalisation_acq_func(self):
        raise NotImplementedError

    @property
    def acq(self):
        # lazy instantiation of the acquisition function
        # repeat when the model has been updated
        if self.updated or self._acq is None:
            self.updated = False
            self._acq = self._get_penalisation_acq_func()

        return self._acq

    def _add_locations(self, X):
        # if X is a vector, change to a 1 x d tensor
        if X.ndim == 1:
            X = X.reshape(1, self.dim)

        self.acq.update_batch_locations(X)

    def get_next(self, q=1):
        res = torch.zeros(q, self.dim, dtype=self.dtype)

        for i in range(q):
            res[i] = self._get_next()
            self._add_locations(res[i])

        return res


class LocalPenalisationBatchBO(PenalisationBaseBatchBO):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        acq_name,
        n_opt_samples,
        n_opt_bfgs,
    ):
        PenalisationBaseBatchBO.__init__(
            self,
            model,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

    def _get_penalisation_acq_func(self):
        return LocalPenalisation(
            self.model, None, self.lb, self.ub, self.acq_name, None
        )


class HardLocalPenalisationBatchBO(PenalisationBaseBatchBO):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        acq_name,
        n_opt_samples,
        n_opt_bfgs,
    ):
        PenalisationBaseBatchBO.__init__(
            self,
            model,
            lb,
            ub,
            under_evaluation,
            acq_name,
            n_opt_samples,
            n_opt_bfgs,
        )

    def _get_penalisation_acq_func(self):
        return HardLocalPenalisation(
            self.model, None, self.lb, self.ub, self.acq_name, None
        )


class LocalPenalisation(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(self, model, x_batch, lb, ub, acq_name, transform=None):

        botorch.acquisition.AnalyticAcquisitionFunction.__init__(self, model)

        self.transform = transform
        self.lb = lb
        self.ub = ub

        # default data type
        self.dtype = model.train_targets.dtype

        # calculate the global Lipschitz constant -- None = global
        self.L = estimate_L(None, self.model, lb, ub)

        # no batch locations to start with
        self.X = torch.zeros((0, lb.numel()), dtype=self.dtype)

        # non-penalised version of the acquisition function
        self.acq_func = acq_func_getter(acq_name, self.model)

        # best location
        self.best = model.train_targets.min()

        # instance of the Normal distribution to evaluate its cdf in the hammer
        self._Normal = torch.distributions.Normal(0.0, 1.0)

        # store the batch locations and precompute the hammer functions
        self.update_batch_locations(x_batch)

    def update_batch_locations(self, X):
        if X is None or X.shape[0] == 0:
            return

        # extend the current tensors
        self.X = torch.cat((self.X, X))

        self._hammer_function_precompute(self.X)

    def _hammer_function_precompute(self, X):
        # if we've started with no locations to penalise
        if X is None or X.shape[0] == 0:
            return

        with torch.no_grad():
            output = self.model.likelihood(
                self.model(X),
                noise=torch.full(
                    (X.shape[0],), self.model.likelihood.noise.mean()
                ),
            )
            mu = output.mean
            var = output.variance
            var[var < 1e-16] = 1e-16
            std = torch.sqrt(var)

        r_mu = torch.abs(mu - self.best) / self.L
        r_std = std / self.L

        self.r_mu = r_mu.flatten()
        self.r_std = r_std.flatten()

    def _hammer_function(self, x):
        # should be this shape in batch setting (from botorch)
        if x.ndim == 3:
            x = x.view(-1, x.shape[2])

        # if we've started with no locations to penalise
        if self.X is None or self.X.shape[0] == 0:
            return torch.ones_like(x)

        dx = torch.norm(x[:, None, :] - self.X[None, :, :], dim=2)

        # hammer function at each location for each penalised location
        h_vals = self._Normal.cdf((dx - self.r_mu) / self.r_std)

        return h_vals

    def forward(self, x):
        fval = self.acq_func(x)
        if self.transform == "softplus":
            fval = torch.log1p(torch.exp(fval))

        if self.X is not None:
            h_vals = self._hammer_function(x)
            fval *= torch.prod(h_vals, axis=-1)

        return fval


class HardLocalPenalisation(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(self, model, x_batch, lb, ub, acq_name, transform=None):

        botorch.acquisition.AnalyticAcquisitionFunction.__init__(self, model)

        self.lb = lb
        self.ub = ub
        self.transform = transform

        # default data type
        self.dtype = model.train_targets.dtype

        # non-penalised version of the acquisition function
        self.acq_func = acq_func_getter(acq_name, self.model)

        # best location
        self.best = model.train_targets.min()

        # no batch locations to start with
        self.X = torch.zeros((0, lb.numel()), dtype=self.dtype)
        self.L = torch.zeros((0,), dtype=self.dtype)

        # store the batch locations and precompute the hammer functions
        self.update_batch_locations(x_batch)

    def update_batch_locations(self, X):
        if X is None or X.shape[0] == 0:
            return

        n = X.shape[0]

        # get the new local Lipschitz contant for each X
        L = torch.zeros((n), dtype=self.dtype)
        for i, x in enumerate(X):
            L[i] = estimate_L(x, self.model, self.lb, self.ub)

        # extend the current tensors
        self.L = torch.cat((self.L, L))
        self.X = torch.cat((self.X, X))

        # precompute the hammer function parameters
        self._hammer_function_precompute(self.X, self.L)

    def _hammer_function_precompute(self, X, L):
        # if we've started with no locations to penalise
        if X is None or X.shape[0] == 0:
            return

        with torch.no_grad():
            output = self.model.likelihood(
                self.model(X),
                noise=torch.full(
                    (X.shape[0],), self.model.likelihood.noise.mean()
                ),
            )
            mu = output.mean
            var = output.variance
            std = torch.sqrt(var)

        self.r_mu = torch.abs(mu.flatten() - self.best) / L
        self.r_std = std.flatten() / L

    def _hammer_function(self, x, p=-5):
        # should this shape in batch setting (from botorch)
        if x.ndim == 3:
            x = x.view(-1, x.shape[2])

        # if we've started with no locations to penalise
        if self.X is None or self.X.shape[0] == 0:
            return torch.ones_like(x, dtype=self.dtype)

        dx = torch.norm(x[:, None, :] - self.X[None, :, :], dim=2)
        h_vals = (1.0 / (self.r_mu + self.r_std)) * dx

        # hammer function at each location for each penalised location
        h_vals = torch.pow(torch.pow(h_vals, p) + 1, 1 / p)

        return h_vals

    def forward(self, x):
        fval = self.acq_func(x)
        if self.transform == "softplus":
            fval = torch.log1p(torch.exp(fval))

        if self.X is not None:
            h_vals = self._hammer_function(x)
            fval *= torch.prod(h_vals, axis=-1)

        return fval


# gradient norm of a gpytorch model
def gp_grad_norm(x, model):
    # we expect x to have require_grad=True.
    d = model.train_inputs[0].shape[-1]

    if x.ndim == 1:
        x.reshape(-1, d)

    observed_pred = model.likelihood(
        model(x),
        noise=torch.full((x.shape[0],), model.likelihood.noise.mean()),
    )
    dydx = torch.autograd.grad(observed_pred.mean.sum(), x)[0]

    # calculate its norm
    normdfdx = torch.norm(dydx, dim=1)

    return normdfdx


# Estimate the Lipschitz constant of the GP by
# finding the largest gradient magnitude.
def estimate_L(xj, model, lb, ub):
    ndim = lb.numel()
    samples = torch.rand(
        (500, ndim), requires_grad=True, dtype=model.train_targets.dtype
    )

    # if we have a location we're centred on, i.e. local Lipschitz constant in
    # the Alvi et al. 2019 paper, set up the bounds in which to search for the
    # largest gradient norm to be centred on the location +- lengthscale / 2,
    # else we just use the original given bounds
    if xj is not None:
        ls = model.covar_module.base_kernel.lengthscale / 2

        # bounds for scipy, clipped to the problem domain
        lb = torch.max(xj - ls, lb)
        ub = torch.min(xj + ls, ub)

    # rescale the data
    samples = samples * (ub - lb) + lb

    # bounds for scipy
    bounds = [*zip(lb.detach().numpy().ravel(), ub.detach().numpy().ravel())]

    samples_dfdx = gp_grad_norm(samples, model)

    # select the starting point as the location with the largest gradient
    x0 = samples[torch.argmax(samples_dfdx)]

    # wrapper function for scipy
    def scipy_objective(x, model, x0, ndim):
        # x = numpy location we want to evaluate
        # model = gpytorch model
        # x0 = torch tensor the same shape as x, with require_gradient=True

        # give x0 the value of x
        x0.data = torch.as_tensor(x, dtype=x0.dtype)
        x0 = x0.view(1, ndim)

        # calculate the gradient
        dydx = gp_grad_norm(x0, model).item()

        # negate it as scipy is minimising and we want the largest
        # gradient norm (hence largest negative gradient norm)
        return -dydx

    xopt, minusL, _ = scipy.optimize.fmin_l_bfgs_b(
        scipy_objective,
        x0.detach().numpy(),
        bounds=bounds,
        args=(model, x0, ndim),
        maxiter=2000,
        approx_grad=True,
    )

    L = -minusL

    # to avoid problems in cases in which the model is flat.
    L = L if L > 1e-7 else 10

    return L
