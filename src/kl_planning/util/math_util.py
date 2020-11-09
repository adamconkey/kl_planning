"""
Utility functions for math operations (e.g. custom KL divergence functions, 
sigma point algorithm, orientation representation utilities).
"""
import sys
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from pyquaternion import Quaternion
from scipy import interpolate
from geometry_msgs.msg import Pose

from kl_planning.distributions import GaussianMixture, DiracDelta
from kl_planning.util import ui_util


def compute_sigma_points(mu, sigma, beta=2):
    """
    Computes sigma points for unscented transform. This algorithm is essentially
    that presented in [1] which has fewer parameters to tune than the original.
    The only difference here from [1] is we include also the mean so there are
    2n+1 points instead of 2n.

    Args:
        mu (Tensor): Mean of current distribution (n_batch, n_state)
        sigma (Tensor): Covariance of current distribution (n_batch, n_state, n_state)
        beta (float): Parameter governing how spread out sigma points are from mean, 
                      higher values spread points out more.
    Returns:
        sigma_points (Tensor): Computed sigma points (n_batch, 2*n_state+1, n_state)

    [1] Manchester, Zachary, and Scott Kuindersma. "Derivative-free trajectory 
        optimization with unscented dynamic programming." 2016 IEEE 55th 
        Conference on Decision and Control (CDC). IEEE, 2016.
    """
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
    Computes the unscented transform of the current Gaussian distribution 
    passed through a nonlinear function. The nonlinear function g is assumed to 
    include any stochasticity being modeled, so no need to add the Q_t term in 
    standard UKF.

    Args:
        mu (Tensor): Mean of current distribution (n_batch, n_state)
        sigma (Tensor): Covariance of current distribution (n_batch, n_state, n_state)
        beta (float): Parameter governing how spread out sigma points are from mean, 
                      higher values spread points out more.
        device (device): Torch device to perform computations on
    Returns:
        mu_prime (Tensor): Mean of the transformed distribution (n_batch, n_state)
        sigma_prime (Tensor): Covariance of the transformed distribution (n_batch, n_state, n_state)
        S (Tensor): Sigma points of the transformed distribution (n_batch, 2*n_state+1, n_state)
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


def kl_gmm_gmm(p, q, device=torch.device('cuda')):
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
        mus = p.loc.unsqueeze(0).to(device)
        sigmas = p.covariance_matrix.unsqueeze(0).to(device)
        weights = torch.ones(1, device=device)
    elif isinstance(p, GaussianMixture):
        mus = p.mu_init.squeeze().unsqueeze(1).repeat(1, B, 1).to(device) # (k, b, n)
        sigmas = torch.diag_embed(p.var_init.squeeze()).unsqueeze(1).repeat(1, B, 1, 1) # (k, b, n, n)
        sigmas = sigmas.to(device)
        weights = p.pi.squeeze().to(device)
    else:
        raise TypeError("Unknown distribution type for first arg to KL gmm-gmm")

    n = mus.size(-1)
    n_components = len(weights)

    kl_approx = torch.zeros(n_components, B, device=device)
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



def pos_from_homogeneous(T):
    """
    Extracts position elements from a homogeneous transformation matrix.
    """
    return T[:3, 3]


def rot_from_homogeneous(T):
    """
    Extraction rotation matrix from homogeneous transformation matrix.
    """
    return T[:3, :3]


def quat_from_homogeneous(T):
    """
    Converts rotation matrix from homogenous TF matrix to quaternion.
    """
    q = Quaternion(matrix=T) # w, x, y, z
    q = np.array([q.x, q.y, q.z, q.w]) # Need to switch to x, y, z, w
    return q


def pose_to_homogeneous(p, q):
    """
    Converts a postition-quaternion pose to a homogeneous transformation matrix.
    
    Args:
        p (array): 3-D position (x, y, z)
        q (qrray): 4-D quaternion (x, y, z, w)
    """
    q = Quaternion(q[3], q[0], q[1], q[2]) # w, x, y, z
    T = q.transformation_matrix
    T[:3, 3] = p
    return T


def rot_from_quat(q):
    """
    Computes (in batch) rotation matrix from quaternion. 

    Args:
        q (Tensor): Quaternion (x, y, z, w) with shape (n_batch, 4). Assumes unit quaternion.
    Returns:
        R (Tensor): Computed rotation matrix (n_batch, 3, 3)

    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    """
    q_i = q[:,0]
    q_j = q[:,1]
    q_k = q[:,2]
    q_r = q[:,3]

    R = torch.stack([
        torch.stack([1. - 2.*(q_j**2 + q_k**2), 2.*(q_i*q_j - q_k*q_r), 2.*(q_i*q_k + q_j*q_r)]),
        torch.stack([2.*(q_i*q_j + q_k*q_r), 1. - 2.*(q_i**2 + q_k**2), 2.*(q_j*q_k - q_i*q_r)]),
        torch.stack([2.*(q_i*q_k - q_j*q_r), 2.*(q_j*q_k + q_i*q_r), 1. - 2.*(q_i**2 + q_j**2)])
    ])

    # TODO I don't understand immediately why the batch ends up in the last dim instead of first
    return R.transpose(0, 2).transpose(1, 2)


def interpolate_poses(start_pose, end_pose, n_points):
    """
    Interpolates between start and end position-quaternion poses.
    
    Args:
        start_pose (Pose): ROS Pose message for start pose
        end_pose (Pose): ROS Pose message for end pose
        n_points (int): Number of points to interpolate between start and end
    Returns:
        interp_points (list): List of interpolated points, each a ROS Pose message
    """
    # Interpolate position
    s = start_pose.position
    e = end_pose.position
    xs_f = interpolate.interp1d([0, 1], [s.x, e.x])
    ys_f = interpolate.interp1d([0, 1], [s.y, e.y])
    zs_f = interpolate.interp1d([0, 1], [s.z, e.z])
    ts = np.linspace(0, 1, n_points)
    xs = xs_f(ts)
    ys = ys_f(ts)
    zs = zs_f(ts)

    # Interpolate orientation
    wp1 = start_pose.orientation
    wp2 = end_pose.orientation
    # We don't want to interpolate if they're already equal, will cause divide by zero
    wp1_arr = np.array([wp1.w, wp1.x, wp1.y, wp1.z])
    wp2_arr = np.array([wp2.w, wp2.x, wp2.y, wp2.z])
    if np.allclose(wp1_arr, wp2_arr):
        qs = [wp1_arr for _ in range(len(ts))]
    else:
        q1 = Quaternion(x=wp1.x, y=wp1.y, z=wp1.z, w=wp1.w)
        q2 = Quaternion(x=wp2.x, y=wp2.y, z=wp2.z, w=wp2.w)
        qs = Quaternion.intermediates(q1, q2, len(ts), include_endpoints=True)
        qs = [q.elements for q in qs]  # getting list form generator

    interp_points = []
    for i in range(len(ts)):
        p = Pose()
        p.position.x = xs[i]
        p.position.y = ys[i]
        p.position.z = zs[i]
        p.orientation.x = qs[i][1]
        p.orientation.y = qs[i][2]
        p.orientation.z = qs[i][3]
        p.orientation.w = qs[i][0]
        interp_points.append(p)
    return interp_points
    

if __name__ == '__main__':
    q = Quaternion.random()
    actual = q.rotation_matrix

    qe = q.elements
    q_input = torch.tensor([qe[1], qe[2], qe[3], qe[0]])
    R = rot_from_quat(torch.stack([q_input, q_input]))

    print("R", R.shape)
    
    print("COMPUTED\n", R.numpy())
    print("ACTUAL\n", actual)
