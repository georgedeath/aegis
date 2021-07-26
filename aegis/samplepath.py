import torch
import gpytorch
from numpy import pi


class SamplePathSampler:
    def __init__(self, model, n_features=1000):
        self.model = model
        self.l = n_features

        # default data type: assume same as model's training target's datatype
        self.dtype = self.model.train_targets.dtype

        # constant value precompute
        self.sqrt_2_over_l = torch.sqrt(
            torch.tensor(2.0 / self.l, dtype=self.dtype)
        )

        # initialise the features
        self._reset_features()

    def _reset_features(self):
        # pointers to the training data
        self.X = self.model.train_inputs[0]
        self.Y = self.model.train_targets.reshape(-1, 1)

        # training data shape
        self.n, self.d = self.X.shape

        # kernel hyperparameters
        self.model_noise = self.model.likelihood.noise[0].detach()
        self.kern_ls = self.model.covar_module.base_kernel.lengthscale.detach()

        # weight and bias for the random Fourier features
        theta = torch.randn(self.l, self.d, dtype=self.dtype)

        if isinstance(
            self.model.covar_module.base_kernel, gpytorch.kernels.RBFKernel
        ):
            self.theta = theta

        elif isinstance(
            self.model.covar_module.base_kernel, gpytorch.kernels.MaternKernel
        ):
            dist = torch.distributions.Gamma(
                concentration=self.model.covar_module.base_kernel.nu,
                rate=self.model.covar_module.base_kernel.nu,
            )
            self.theta = theta * torch.sqrt(dist.sample((self.l, self.d)))

        else:
            s = f"Unknown kernel type: {self.model.covar_module.base_kernel.__class__}"
            raise ValueError(s)

        self.tau = torch.normal(
            mean=0, std=2 * pi, size=(self.l, 1), dtype=self.dtype
        )

        # calculate feature matrix Phi (l features, n training)
        self.Phi = self.sqrt_2_over_l * torch.cos(
            self.theta @ (self.X.T / self.kern_ls.T) + self.tau
        )

        # precompute the kernel inverse for v (as this doesn't change)
        with torch.no_grad():
            KXX = self.model.covar_module(self.X)
            KXX = KXX.evaluate()  # KXX is a lazy tensor, need to initialise
        self.Kinv = torch.pinverse(KXX + self.model_noise * torch.eye(self.n))

    # this should return a callable function that is effectively
    # an instance of a sample path and can be queried anywhere
    def draw_sample_path(self):
        # draw a random weighting defining the sample path
        w = torch.normal(0, 1, size=(self.l, 1), dtype=self.dtype)

        def f(x):
            if x.ndim == 1:
                if self.d == 1:
                    reshape_size = (-1, 1)

                else:
                    reshape_size = (1, self.d)

                x = x.view(reshape_size)

            # feature responses at x
            phis = self.sqrt_2_over_l * torch.cos(
                self.theta @ (x.T / self.kern_ls.T) + self.tau
            )

            # kernel responses at x
            # with torch.no_grad():
            Kx = self.model.covar_module(x, self.X)
            Kx = Kx.evaluate()

            # weight space prior -- equiv to, but faster than
            # np.sum(phis * w, axis=0)
            lhs = w.T @ phis

            # function-space update
            update = (self.Phi.T @ w) + self.model_noise * torch.normal(
                0, 1, size=(self.n, 1)
            )
            v = self.Kinv @ (self.Y.reshape(-1, 1) - update)
            rhs = Kx @ v

            return lhs.squeeze() + rhs.squeeze()

        return f
