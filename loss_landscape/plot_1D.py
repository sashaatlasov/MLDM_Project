import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import figure
import torch
from typing import Tuple, Dict
from tqdm.auto import tqdm
from loss_landscape.utils import filter_norm_direction


def sum_state_dicts(optima1, optima2, alpha):
    """
    Iterpolates between two optimas with a cofficient.

    Parameters
    ----------
    optima1: dict
        first model state dict
    optima2 : dict
        second model state dict 
    alpha : float 
        iterpolation coefficient

    Returns
    -------
    sum : dict
        sum of two state dicts
    """
    sum = {}
    for block in optima1.keys():
        sum[block] = alpha * optima1[block] + (1 - alpha) * optima2[block]
    return sum


def calculate_metrics(model, criterion, x, y, optima1: Dict[str, torch.tensor], optima2: Dict[str, torch.tensor] = None,
                      coef: Tuple[float, float] = (-0.5, 1.5), num_steps: int = 50) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------

    Returns
    -------

    """
    grid = np.linspace(coef[0], coef[1], num_steps)
    if optima2 == None:
        direction = filter_norm_direction(optima1)
        optima2 = sum_state_dicts(optima1, direction, 0.5)

    losses = []
    for alpha in tqdm(grid):
        weights = sum_state_dicts(optima1, optima2, alpha)
        model.load_state_dict(weights)
        loss = criterion(model(x), y)
        losses.append(loss.item())
    return json.dumps({'grid': list(grid), 'loss': list(losses)})


def plot_1D(metrics: json, save: bool = False) -> figure:
    """
    Plots 1-dimensional linear interpolation of loss function between two solutions.

    Parameters
    ----------
    metrics : json
        json calculates metrics neccessary for plot

    Returns
    -------
    figure : figure
        matplotlib figure with 1D loss visualization

    """
    metrics = json.loads(metrics)
    grid = np.array(metrics['grid'])
    loss = np.array(metrics['loss'])

    plt.figure(dpi=300)
    plt.rcParams['text.usetex'] = True
    plt.plot(grid, loss)
    plt.title('')
    plt.xlabel('Interpolation coefficient')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
