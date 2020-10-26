import sys
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from kl_planning.distributions import GaussianMixture, DiracDelta
from kl_planning.util import ui_util


def compute_sigma_points(mu, sigma, beta=2):
    B = mu.size(0)
    n = mu.size(-1)
    n_sigma = 2 * n + 1

    L = torch.cholesky(sigma) # One (fast) way to do matrix sqrt        
    P_sigma = mu.unsqueeze(1).repeat(1, n_sigma, 1)  # (B, 2n+1, n), P_sigma[0] stays mean
    for i in range(n):
        P_sigma[:,i+1,:] += beta * L[:,i,:]
        P_sigma[:,n+i+1,:] -= beta * L[:,i,:]
    return P_sigma
    

def unscented_transform(mu, sigma, g, beta=2, device=torch.device('cuda')):
    """
    The nonlinear function g is assumed to include any stochasticity being 
    modeled, so no need to add the Q_t term in standard UKF.
    """
    B = mu.size(0)
    n_state = mu.size(-1)
    n_sigma = 2 * n_state + 1

    P_sigma = compute_sigma_points(mu, sigma, beta)
    S = g(P_sigma.view(B * n_sigma, -1)).view(B, n_sigma, -1) # (B, 2n+1, n)

    mu_prime = torch.sum(S / (2. * n_state + 1), dim=1)
    sigma_prime = torch.zeros(B, n_state, n_state, device=device)
    for i in range(n_sigma):
        x = S[:,i,:] - mu_prime
        sigma_prime += x.unsqueeze(2) * x.unsqueeze(1) / (2. * beta**2) # outer product
        
    return mu_prime, sigma_prime, S


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
        raise TypeError("None of the dists are MVN, cannot infer batch")

    # Handle MVN/GMM for first arg, can be either
    if isinstance(p, MultivariateNormal):
        mus = p.loc.unsqueeze(0)
        sigmas = p.covariance_matrix.unsqueeze(0)
        weights = torch.ones(1)
    elif isinstance(p, GaussianMixture):
        mus = p.mu_init.squeeze().unsqueeze(1).repeat(1, B, 1) # (k, b, n)
        sigmas = torch.diag_embed(p.var_init.squeeze()).unsqueeze(1).repeat(1, B, 1, 1) # (k, b, n, n)
        weights = p.pi.squeeze()
    else:
        raise TypeError("Unknown distribution type for first arg to KL gmm-gmm")

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

    return kl_approx


def kl_dirac_mvn(dirac, mvn, device=torch.device('cuda')):
    """
    Computes KL divergence between DiracDelta and MultivariateNormal.

    Solving you get -log(q(x)) which for MVN reduces to a constant plus 
    precision-weighted Euclidean distance. 

    Note this computes the M-projection where the MVN is the one you have 
    some control over. The I-projection is infinite because it puts the 
    dirac in the denominator of the log in KL which is zero everywhere 
    except one point, causing division by zero and it's infinite.
    """
    if not isinstance(dirac, DiracDelta):
        raise TypeError("First arg must be type DiracDelta")
    if not isinstance(mvn, MultivariateNormal):
        raise TypeError("Second arg must be type MultivariateNormal")
    if dirac.state.shape != mvn.loc.shape:
        raise ValueError(f"Dirac state shape {dirac.state.shape} "
                         f"must match MVN mean shape {mvn.loc.shape}")

    if dirac.force_identity_precision:
        # TODO hard-coded device
        precision = torch.eye(mvn.loc.size(-1), device=device)
        precision = precision.unsqueeze(0).repeat(mvn.loc.size(0), 1, 1)
    else:
        precision = mvn.precision_matrix

    diff = dirac.state - mvn.loc
    t1 = torch.bmm(diff.unsqueeze(1), precision)
    kl = 0.5 * torch.bmm(t1, diff.unsqueeze(-1))
    return kl.squeeze()
    
