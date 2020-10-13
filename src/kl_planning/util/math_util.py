import torch


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
    w_c = torch.full((B, n_sigma, 1, 1), 1. / (2 * (n + lambda_)))
    w_m[:,0] = lambda_ / (n + lambda_)
    w_c[:,0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    M = torch.cholesky((n + lambda_) * sigma) # One (fast) way to do matrix sqrt
        
    X = mu.unsqueeze(1).repeat(1, n_sigma, 1)  # (B, 2n+1, n)
    for i in range(n):
        X[:,i+1,:] += M[:,i,:]
        X[:,n+i+1,:] -= M[:,i,:]
        
    Y = g(X.view(B * n_sigma, -1)).view(B, n_sigma, -1) # (B, 2n+1, state_size)

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
