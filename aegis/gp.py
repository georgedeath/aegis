import copy
import torch
import botorch
import gpytorch
from botorch.models.gpytorch import GPyTorchModel


class GP_FixedNoise(gpytorch.models.ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(
        self, train_x, train_y, ls_prior, os_prior, noise_size=1e-6, ARD=False
    ):

        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            torch.full((train_y.shape[0],), noise_size)
        )
        super(GP_FixedNoise, self).__init__(train_x, train_y, likelihood)

        base_kernel = gpytorch.kernels.MaternKernel(
            lengthscale_prior=ls_prior,
            nu=5 / 2,
            ard_num_dims=train_x.shape[-1] if ARD else None,
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_prior=os_prior,
        )
        self.mean_module = gpytorch.means.ZeroMean()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_model_restarts(mll, n_restarts, verbose=False):
    def model_loss(mll):
        mll.train()
        output = mll.model(*mll.model.train_inputs)
        loss = -mll(output, mll.model.train_targets)
        return loss.sum().item()

    noise_prior = "noise_prior" in mll.model.likelihood.noise_covar._priors

    # start by assuming the current parameters are the best
    best_params = copy.deepcopy(mll.state_dict())
    best_loss = model_loss(mll)

    for i in range(n_restarts):
        # sample new hyperparameters from the kernel priors
        mll.model.covar_module.base_kernel.sample_from_prior(
            "lengthscale_prior"
        )
        mll.model.covar_module.sample_from_prior("outputscale_prior")

        #  if we have one, sample from noise prior
        if noise_prior:
            mll.model.likelihood.noise_covar.sample_from_prior("noise_prior")

        # try and fit the model using bfgs, starting at the sampled params
        botorch.fit_gpytorch_model(mll, method="L-BFGS-B")

        # calculate the loss
        curr_loss = model_loss(mll)

        # if we've ended up with better hyperparameters, save them to use
        if curr_loss < best_loss:
            best_params = copy.deepcopy(mll.state_dict())
            best_loss = curr_loss

    # load the best found parameters into the model
    mll.load_state_dict(best_params)

    if verbose:
        ls = mll.model.covar_module.base_kernel.lengthscale.detach().numpy()
        ops = mll.model.covar_module.outputscale.item()
        print("Best found hyperparameters:")
        print(f"\tLengthscale: {ls.ravel()}")
        print(f"\tOutputscale: {ops}")


def create_and_fit_GP(
    train_x,
    train_y,
    ls_bounds,
    out_bounds,
    n_restarts,
    verbose=False,
    noise_size=1e-4,
):

    # Kernel lengthscale and variance (output scale) priors
    ls_prior = gpytorch.priors.UniformPrior(*ls_bounds)
    os_prior = gpytorch.priors.UniformPrior(*out_bounds)
    ls_prior._validate_args = False
    os_prior._validate_args = False

    # instantiate model
    model = GP_FixedNoise(
        train_x, train_y, ls_prior, os_prior, noise_size, ARD=False
    )

    # train it
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    train_model_restarts(mll, n_restarts=n_restarts, verbose=verbose)

    return model, model.likelihood
