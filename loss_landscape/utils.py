import torch
from typing import Dict


def filter_norm_direction(optima: Dict, inplace=False) -> Dict:
    """
    Creates random direction in parameters space with filter normalization.

    Parameters
    ----------
    optima : dict
        a model's state dict, which requires random direction 

    Returns
    -------
    direction : dict 
        filter normalized random direction with the same dimensions as optima
    """
    direction = {}
    for block in optima.keys():
        weights = optima[block].to(torch.float16)
        rand_dir = torch.randn_like(weights)
        direction[block] = rand_dir * \
            torch.norm(weights) / torch.norm(rand_dir)
    return direction
