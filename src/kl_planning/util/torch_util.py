"""
Convenience functions for working with pytorch.
"""
from torch.distributions.kl import _KL_REGISTRY


def view_kl_options():
    """
    Prints all distribution combinations for which KL divergence function is
    defined for in pytorch. Iterates over registry to find the pairs.
    """
    names = [(k[0].__name__, k[1].__name__) for k in _KL_REGISTRY.keys()]
    max_name_len = max([len(t[0]) for t in names])
    for arg1, arg2 in sorted(names):
        print(f"  {arg1:>{max_name_len}} || {arg2}")

        
if __name__ == '__main__':
    view_kl_options()
