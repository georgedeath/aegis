import torch
import botorch
import scipy

from .base import BaseBatchBO
from .hallucination import HalluBaseBatchBO
from ..samplepath import SamplePathSampler


class BatchTS(BaseBatchBO):
    """

    Parameters
    ----------
    n_features: int, Default 2000
        Number of Fourier features to approximate the posterior with.

    n_opt_samples: int, Default 5000
        Number of sobol samples generated and evaluated to find the
        best 'n_opt_bfgs' samples from to locally optimise.

    n_opt_bfgs: int, Default 10
        Number of samples to locally optimise with L-BFGS-B.
    """

    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        n_features=2000,
        n_opt_samples=5000,
        n_opt_bfgs=10,
    ):
        BaseBatchBO.__init__(self, model, lb, ub, under_evaluation)

        self.n_features = n_features
        self.n_opt_samples = n_opt_samples
        self.n_opt_bfgs = n_opt_bfgs

        # storage
        self._sps = None

    @property
    def sps(self):
        # lazy instantiation
        if self.updated or self._sps is None:
            self.updated = False

            self._sps = SamplePathSampler(
                self.model, n_features=self.n_features
            )

        return self._sps

    def _get_next(self):
        # draw a realisation of the posterior
        sample_path = self.sps.draw_sample_path()

        # randomly draw self.n_samples sobol evaluations
        se = torch.quasirandom.SobolEngine(self.dim, scramble=True)
        X = se.draw(self.n_opt_samples, dtype=self.dtype)

        # evaluate the samples and negate their values
        # because we're minimising
        fX = sample_path(X) * -1.0

        # select the best locations
        best_inds = BoltzmannSampling(fX, self.n_opt_bfgs, eta=1.0)
        best_X = X[best_inds]

        # setup the local optimisation variables
        bounds = scipy.optimize.Bounds(self.lb, self.ub, keep_feasible=True)
        var = torch.zeros(self.dim, requires_grad=True, dtype=self.dtype)
        maximise = False
        with_grad = True
        args = (sample_path, var, with_grad, maximise)

        # results storage
        res_f = torch.zeros(self.n_opt_bfgs, dtype=self.dtype)
        res_X = torch.zeros(self.n_opt_bfgs, self.dim, dtype=self.dtype)

        # perform the local optimisation
        for i in range(self.n_opt_bfgs):
            rdict = scipy.optimize.minimize(
                fun=scipy_objective,
                x0=best_X[i],
                args=args,
                bounds=bounds,
                jac=True,
                method="L-BFGS-B",
            )

            res_f[i] = rdict["fun"]
            res_X[i] = torch.as_tensor(rdict["x"], dtype=self.dtype)

        # select the best (lowest) value as the location
        # to expensively evaluate next
        am = torch.argmin(res_f)

        return res_X[am]


class BatchTSHallu(BatchTS, HalluBaseBatchBO):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        n_features=2000,
        n_opt_samples=5000,
        n_opt_bfgs=10,
    ):
        BatchTS.__init__(
            self,
            model,
            lb,
            ub,
            under_evaluation,
            n_features=n_features,
            n_opt_samples=n_opt_samples,
            n_opt_bfgs=n_opt_bfgs,
        )
        HalluBaseBatchBO.__init__(self, model, lb, ub, under_evaluation)


def BoltzmannSampling(fX, num_samples, eta=1.0):
    # Boltzmann sampling -- similar to botorch version

    sfX = botorch.utils.transforms.standardize(fX)
    weights = torch.exp(eta * sfX)

    inds = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False
    )

    return inds


def scipy_objective(x, f, var, with_grad=True, maximise=True):
    """Scipy wrapper for a torch function with gradient
    x: np.ndarray
    f: callable torch function that we can compute the gradient for
    var: torch tensor with requires_grad set to True.
    """
    # reset the gradient to zero before we calculate it,
    # if it has been previously set
    if var.grad is not None:
        var.grad.zero_()

    # set the tensor with gradient to have the value of x,
    # with the same data type as the tensor
    var.data = torch.as_tensor(x, dtype=var.dtype)

    # forward pass; i.e. evaluate the variable in the sample path
    output = f(var)
    y = output.detach().numpy().astype(x.dtype).item()

    if maximise:
        y *= -1

    if not with_grad:
        return y

    # compute the gradient
    output.backward()

    # extract it and convert to numpy array
    g = var.grad.detach().view(-1).numpy().astype(x.dtype)

    # return the function value (float) and gradient (vector)
    return y, g
