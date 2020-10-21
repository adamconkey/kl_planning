

class DiracDelta:
    """
    Dirac-delta distribution about a given state. Models having infinite density 
    at one point x such that p(x) = 1 and for all points y != x, p(y) = 0.
    """

    def __init__(self, state, force_identity_precision=False):
        """
        Args:
            state (Tensor): Size (B, N)
            force_identity_precision (bool): Set true to use identity precision in KL against
               MVN instead of using the MVN's actual precision matrix.
        """
        self.state = state
        self.force_identity_precision = force_identity_precision
