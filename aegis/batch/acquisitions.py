import torch
import botorch

from .base import BaseBatchBO


class AcqBaseBatchBO(BaseBatchBO):
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
        BaseBatchBO.__init__(self, model, lb, ub, under_evaluation)

        self.n_opt_samples = n_opt_samples
        self.n_opt_bfgs = n_opt_bfgs
        self.problem_bounds = torch.stack((lb, ub)).type(
            model.train_targets.dtype
        )

        self.acq_name = acq_name

    @property
    def acq(self):
        raise NotImplementedError

    def _get_next(self):
        # perform Boltzmann sampling with n_opt_samples samples and
        # optimise the best n_opt_bfgs of these with l-bfgs-b

        n_samples = self.n_opt_samples / 2

        # try a few times just in case we get unlucky and all our samples
        # are in flat regions of space (unlikely but can happen with EI)
        MAX_ATTEMPTS = 5

        for attempts in range(MAX_ATTEMPTS):
            n_samples *= 2

            try:
                train_xnew, acq_f = botorch.optim.optimize_acqf(
                    acq_function=self.acq,
                    q=1,
                    bounds=self.problem_bounds,
                    num_restarts=self.n_opt_bfgs,
                    raw_samples=self.n_opt_samples,
                )
                return train_xnew

            # botorch throws a RuntimeError with the reason:
            # invalid multinomial distribution (sum of probabilities <= 0)
            except RuntimeError:
                continue

        # if we've reached this point we've failed to get a valid location,
        # so raise an exception
        msg = "Failed to optimise the acquisition function after"
        msg += f" {MAX_ATTEMPTS} attempts. The acquisition function had a "
        msg += " sum of probabilities <= 0 every time. This is very unlikely!"
        raise RuntimeError
