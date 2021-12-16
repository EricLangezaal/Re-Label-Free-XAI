import numpy as np

import math
import torch


def matrix_log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian for all combination of bacth pairs of
    `x` and `mu`. I.e. return tensor of shape `(batch_size, batch_size, dim)`
    instead of (batch_size, dim) in the usual log density.
    Parameters
    ----------
    x: torch.Tensor
        Value at which to compute the density. Shape: (batch_size, dim).
    mu: torch.Tensor
        Mean. Shape: (batch_size, dim).
    logvar: torch.Tensor
        Log variance. Shape: (batch_size, dim).
    batch_size: int
        number of training images in the batch
    """
    batch_size, dim = x.shape
    x = x.view(batch_size, 1, dim)
    mu = mu.view(1, batch_size, dim)
    logvar = logvar.view(1, batch_size, dim)
    return log_density_gaussian(x, mu, logvar)


def log_density_gaussian(x, mu, logvar):
    """Calculates log density of a Gaussian.
    Parameters
    ----------
    x: torch.Tensor or np.ndarray or float
        Value at which to compute the density.
    mu: torch.Tensor or np.ndarray or float
        Mean.
    logvar: torch.Tensor or np.ndarray or float
        Log variance.
    """
    normalization = - 0.5 * (math.log(2 * math.pi) + logvar)
    inv_var = torch.exp(-logvar)
    log_density = normalization - 0.5 * ((x - mu)**2 * inv_var)
    return log_density


def log_importance_weight_matrix(batch_size, dataset_size):
    """
    Calculates a log importance weight matrix
    Parameters
    ----------
    batch_size: int
        number of training images in the batch
    dataset_size: int
    number of training images in the dataset
    """
    N = dataset_size
    M = batch_size - 1
    strat_weight = (N - M) / (N * M)
    W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
    W.view(-1)[::M + 1] = 1 / N
    W.view(-1)[1::M + 1] = strat_weight
    W[M - 1, 0] = strat_weight
    return W.log()


def off_diagonal_sum(mat: np.ndarray) -> np.ndarray:
    return np.sum(mat) - np.trace(mat)


def cos_saliency(saliency: np.ndarray) -> np.ndarray:
    latent_dim = saliency.shape[1]
    cos_avg = np.ones((latent_dim, latent_dim))
    cos_std = np.ones((latent_dim, latent_dim))
    for dim1 in range(1, latent_dim):
        for dim2 in range(dim1):
            saliency_dim1, saliency_dim2 = saliency[:, dim1], saliency[:, dim2]
            normalization = np.sqrt(np.sum(saliency_dim1**2, axis=(-2, -1)) *
                                    np.sum(saliency_dim2**2, axis=(-2, -1)))
            cos_dim1_dim2 = np.sum(saliency_dim1*saliency_dim2, axis=(-2, -1))/normalization
            cos_avg[dim1, dim2] = np.mean(cos_dim1_dim2)
            cos_std[dim1, dim2] = np.std(cos_dim1_dim2)
            cos_avg[dim2, dim1] = cos_avg[dim1, dim2]
            cos_std[dim2, dim1] = cos_std[dim1, dim2]
    return cos_avg, cos_std