from torch.distributions.kl import _KL_REGISTRY

def view_kl_options():
    names = [(k[0].__name__, k[1].__name__) for k in _KL_REGISTRY.keys()]
    max_name_len = max([len(t[0]) for t in names])
    for arg1, arg2 in sorted(names):
        print(f"  {arg1:>{max_name_len}} || {arg2}")


view_kl_options()
