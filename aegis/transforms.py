import torch
from scipy.stats import median_abs_deviation


class Transform_Base(object):
    """
    Base class for transformations based on some data.
    """

    def __init__(self, Ytr):
        self.Ytr = Ytr

    # Transform the mean
    def scale_mean(self, mu):
        return mu

    # Reverse the transformation to the mean
    def unscale_mean(self, mu):
        return mu

    # Reverse the transformation to the variance
    def unscale_var(self, var):
        return var


class Transform_Standardize(Transform_Base):
    """
    Standardize the data
    """

    def __init__(self, Ytr):
        super().__init__(Ytr)
        self.Ytr_mean = Ytr.mean()
        self.Ytr_std = Ytr.std()
        self.Ytr_var = Ytr.var()

    def scale_mean(self, mu):
        return (mu - self.Ytr_mean) / self.Ytr_std

    def unscale_mean(self, mu):
        return mu * self.Ytr_std + self.Ytr_mean

    def unscale_var(self, var):
        return var * self.Ytr_var


class Transform_StandardizeRobustly(Transform_Base):
    """
    Robustly standardize the data by estimating its scale
    """

    def __init__(self, Ytr):
        super().__init__(Ytr)
        self.Ytr_median = Ytr.median()
        Ytr_numpy = Ytr.numpy().ravel()
        self.Ytr_scale = torch.tensor(median_abs_deviation(Ytr_numpy))
        self.Ytr_scaleSQR = self.Ytr_scale ** 2

    def scale_mean(self, mu):
        return (mu - self.Ytr_median) / self.Ytr_scale

    def unscale_mean(self, mu):
        return mu * self.Ytr_scale + self.Ytr_median

    def unscale_var(self, var):
        return var * self.Ytr_scaleSQR
