import torch
from .base import SelectionParetoFront, SelectionRandom
from .ThompsonSampling import BatchTS
from .aegis import aegisBase, aegisExploitBase


# --------------------------------------------------------------------------- #
# No exploitation
# --------------------------------------------------------------------------- #
class ablationNoExploitRS(SelectionRandom, aegisBase):
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
        SelectionRandom.__init__(self)

        assert self.epsilon == 0.5


class ablationNoExploitPF(SelectionParetoFront, aegisBase):
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
        SelectionParetoFront.__init__(self)

        assert self.epsilon == 0.5


# --------------------------------------------------------------------------- #
# No sample path selection
# --------------------------------------------------------------------------- #
class ablationNoSamplepathBase(aegisExploitBase):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        epsilon,
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
            eta=0.5,  # not used
            n_workers=n_workers,
            n_features=n_features,
            n_opt_samples=n_opt_samples,
            n_opt_bfgs=n_opt_bfgs,
        )

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

        else:
            print("Random selection")
            xnew = self._rand_selection()

        self.t += 1
        return xnew


class ablationNoSamplepathRS(SelectionRandom, ablationNoSamplepathBase):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        epsilon,
        n_workers,
        n_features,
        n_opt_samples,
        n_opt_bfgs,
    ):
        ablationNoSamplepathBase.__init__(
            self,
            model,
            lb,
            ub,
            under_evaluation,
            epsilon,
            n_workers,
            n_features,
            n_opt_samples,
            n_opt_bfgs,
        )
        SelectionRandom.__init__(self)


class ablationNoSamplepathPF(SelectionParetoFront, ablationNoSamplepathBase):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        epsilon,
        n_workers,
        n_features,
        n_opt_samples,
        n_opt_bfgs,
    ):
        ablationNoSamplepathBase.__init__(
            self,
            model,
            lb,
            ub,
            under_evaluation,
            epsilon,
            n_workers,
            n_features,
            n_opt_samples,
            n_opt_bfgs,
        )
        SelectionParetoFront.__init__(self)


# --------------------------------------------------------------------------- #
# Random sampling
# --------------------------------------------------------------------------- #
class ablationNoRandom(aegisExploitBase):
    def __init__(
        self,
        model,
        lb,
        ub,
        under_evaluation,
        epsilon,
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
            eta=0.5,  # not used
            n_workers=n_workers,
            n_features=n_features,
            n_opt_samples=n_opt_samples,
            n_opt_bfgs=n_opt_bfgs,
        )

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

        else:
            print("Sample path selection")
            xnew = BatchTS._get_next(self)

        self.t += 1
        return xnew
