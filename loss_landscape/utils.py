import torch

def filter_norm_direction(optima, inplace=False):
    """
    Parameters
    ----------

    Returns
    -------

    """
    direction = {}
    for block in optima.keys():
        weights = optima[block].to(torch.float16)
        rand_dir = torch.randn_like(weights)
        direction[block] = rand_dir * \
            torch.norm(weights) / torch.norm(rand_dir)
    return direction