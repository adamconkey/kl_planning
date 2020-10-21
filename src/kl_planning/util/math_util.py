import sys
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from kl_planning.models import GaussianMixture
from kl_planning.util import ui_util


def compute_sigma_points(mu, sigma, alpha=1, beta=2, kappa=1):
    B = mu.size(0)
    n = mu.size(-1)
    n_sigma = 2 * n + 1
    lambda_ = alpha**2 * (n + kappa) - n

    M = torch.cholesky((n + lambda_) * sigma) # One (fast) way to do matrix sqrt
        
    sigma_points = mu.unsqueeze(1).repeat(1, n_sigma, 1)  # (B, 2n+1, n)
    for i in range(n):
        sigma_points[:,i+1,:] += M[:,i,:]
        sigma_points[:,n+i+1,:] -= M[:,i,:]
    return sigma_points
    

def unscented_transform(mu, sigma, g, alpha=1, beta=2, kappa=1):
    """
    beta=2 optimal for Gaussians

    Not sure about the alpha/kappa defaults
    """
    B = mu.size(0)
    n = mu.size(-1)
    n_sigma = 2 * n + 1
    lambda_ = alpha**2 * (n + kappa) - n

    w_m = torch.full((B, n_sigma, 1), 1. / (2 * (n + lambda_)))
    w_m[:,0] = lambda_ / (n + lambda_)
    w_c = torch.full((B, n_sigma, 1, 1), 1. / (2 * (n + lambda_)))
    w_c[:,0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    sigma_points = compute_sigma_points(mu, sigma, alpha, beta, kappa)
    Y = g(sigma_points.view(B * n_sigma, -1)).view(B, n_sigma, -1) # (B, 2n+1, n)

    mu_prime = torch.sum(w_m * Y, dim=1)
    sigma_prime = torch.zeros(B, n, n)
    for i in range(n_sigma):
        y = Y[:,i,:] - mu_prime
        sigma_prime += w_c[:,i] * y.unsqueeze(2) * y.unsqueeze(1) # outer product

    # TODO can add in this process noise which is in standard UKF, however I think UKF
    # assumes you're using a deterministic nonlinear function. I have stochastic dynamics
    # so I think it should be equivalent to not model it here and instead just compute
    # sigma points being passed through stochastic nonlinear function.
    # sigma_prime += torch.diag_embed(torch.rand(B, n), dim1=-2, dim2=-1) * 0.005
        
    return mu_prime, sigma_prime, Y


def kl_gmm_gmm(p, q):
    """
    Computes approximate KL divergence between two GMMs. Either arg can be either 
    MultivariateNormal or GaussianMixture, and if any are MVN they get treated as
    a single-component GMM.

    Uses unscented transform as more efficient alternative to sampling. Method described in:
      [1] https://www.isca-speech.org/archive/archive_papers/interspeech_2005/i05_1985.pdf
    """
    # TODO Need to figure out batch size, assuming at least one is MVN which has batch,
    # if you need general GMM-GMM then maybe need to figure out batch version of that
    if isinstance(p, MultivariateNormal):
        B = p.loc.size(0)
    elif isinstance(q, MultivariateNormal):
        B = q.loc.size(0)
    else:
        ui_util.print_error("\nNone of the dists are MVN, cannot infer batch")
        return None
    
    
    if isinstance(p, MultivariateNormal):
        mus = p.loc.unsqueeze(0)
        sigmas = p.covariance_matrix.unsqueeze(0)
        weights = torch.ones(1)
    elif isinstance(p, GaussianMixture):
        mus = p.mu_init.squeeze().unsqueeze(1).repeat(1, B, 1) # (k, b, n)
        sigmas = torch.diag_embed(p.var_init.squeeze()).unsqueeze(1).repeat(1, B, 1, 1) # (k, b, n, n)
        weights = p.pi.squeeze()
    else:
        ui_util.print_error("\nUnknown distribution type for first arg to KL gmm-gmm\n")
        return None

    # print("MUS", mus.shape)
    # print("SIGMAS", sigmas.shape)
    # print("WEIGHTS", weights.shape)

    n = mus.size(-1)
    n_components = len(weights)

    kl_approx = torch.zeros(n_components, B)
    for k in range(n_components):
        p_log_p = -0.5 * (torch.logdet(sigmas[k]) + n)
        # Approximate p*log(q) using sigma point method in [1]
        sigma_points = compute_sigma_points(mus[k], sigmas[k]) # (B, 2n+1, n)
        n_sigma = sigma_points.size(1)
        if isinstance(q, GaussianMixture):
            log_lh = q.score_samples(sigma_points.view(B * n_sigma, -1)).view(B, n_sigma, -1)
            log_lh = log_lh.sum(dim=1)
        elif isinstance(q, MultivariateNormal):
            # TODO need to fix this to be able to work batch computation for sigma points also
            log_lh = 0
            for i in range(n_sigma):
                log_lh += q.log_prob(sigma_points[:,i])
        else:
            ui_util.print_error("\nUnknown distribution type for second arg to KL gmm-gmm\n")
            return None
        p_log_q = log_lh / float(n_sigma)    
        kl_approx[k] = weights[k] * (p_log_p - p_log_q.squeeze())

    kl_approx = kl_approx.sum(dim=0) # Sum over components
        
    # print("KL", kl_approx)
    # sys.exit()

    return kl_approx
