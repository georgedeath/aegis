import torch
from .base import BaseBatchBO
from .acquisitions import AcqBaseBatchBO
from ..util import acq_func_getter


class HalluBaseBatchBO(BaseBatchBO):
    def __init__(self, model, lb, ub, under_evaluation):
        BaseBatchBO.__init__(self, model, lb, ub, under_evaluation)

        if (under_evaluation is not None) and (under_evaluation.numel() != 0):
            self._add_locations(under_evaluation)

    def _add_locations(self, X):
        # if X is a vector, change to a 1 x d tensor
        if X.ndim == 1:
            X = X.reshape(1, self.dim)

        # predict the outputs of X
        with torch.no_grad():
            Y = self.model.likelihood(
                self.model(X),
                noise=torch.full(
                    (X.shape[0],), self.model.likelihood.noise.mean()
                ),
            ).mean

        # self.model = self.model.get_fantasy_model(
        #     inputs=X,
        #     targets=Y,
        #     noise=torch.full_like(Y, self.model.likelihood.noise[0]),
        # )

        # set the model's training data
        # note: we have to do this rather than use get_fantasy_model() (above)
        # because calling get_fantasy_model multiple times does not appear to
        # work in the same way set_train_data works.
        self.model.set_train_data(
            inputs=torch.cat((self.model.train_inputs[0], X)),
            targets=torch.cat((self.model.train_targets, Y.view(-1))),
            strict=False,
        )

        # set the corresponding noise
        self.model.likelihood.noise = torch.full_like(
            self.model.train_targets, self.model.likelihood.noise[0]
        )

    def get_next(self, q=1):
        res = torch.zeros(q, self.dim, dtype=self.dtype)

        for i in range(q):
            res[i] = self._get_next()
            self._add_locations(res[i])

        return res


class HalluBatchBO(HalluBaseBatchBO, AcqBaseBatchBO):
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
        HalluBaseBatchBO.__init__(self, model, lb, ub, under_evaluation)

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

    @property
    def acq(self):
        return acq_func_getter(self.acq_name, self.model)
