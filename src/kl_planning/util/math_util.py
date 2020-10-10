import torch


def unscented_transform(mu, sigma, g, alpha=1, beta=2, kappa=1):
    """
    beta=2 optimal for Gaussians

    Not sure about the alpha/kappa defaults
    """
    B = mu.size(0)
    n = mu.size(-1)
    lambda_ = alpha**2 * (n + kappa) - n

    Q = torch.diag_embed(torch.rand(B, n), dim1=-2, dim2=-1) * 0.005

    w_m = torch.full((B, 2*n + 1, 1), 1. / (2 * (n + lambda_)))
    w_c = torch.full((B, 2*n + 1, 1, 1), 1. / (2 * (n + lambda_)))
    w_m[:,0] = lambda_ / (n + lambda_)
    w_c[:,0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    M = torch.cholesky((n + lambda_) * sigma) # One (fast) way to do matrix sqrt
        
    X = mu.unsqueeze(1).repeat(1, 2*n + 1, 1)  # (B, 2n+1, n)
    for i in range(n):
        X[:,i+1,:] += M[:,i,:]
        X[:,n+i+1,:] -= M[:,i,:]
        
    Y = g(X) # (B, 2n+1, state_size)

    mu_prime = torch.sum(w_m * Y, dim=1)
    sigma_prime = torch.zeros(B, n, n)
    for i in range(2*n+1):
        y = Y[:,i,:] - mu_prime
        sigma_prime += w_c[:,i] * y.unsqueeze(2) * y.unsqueeze(1) # outer product
    sigma_prime += Q
        
    return mu_prime, sigma_prime, Y
