import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import figure
import torch
from typing import Tuple, Dict
from tqdm.auto import tqdm
from utils import filter_norm_direction


def sum_state_dicts(optima1, optima2, alpha):
    sum = {}
    for block in optima1.keys():
        sum[block] = alpha * optima1[block] + (1 - alpha) * optima2[block]
    return sum


def calculate_metrics(model, criterion, x, y, optima1: Dict[str, torch.tensor], optima2: Dict[str, torch.tensor] = None,
                      coef: Tuple[float, float] = (-0.5, 1.5)) -> Tuple[np.array, np.array]:
    """
    Parameters
    ----------

    Returns
    -------

    """
    grid = np.linspace(coef[0], coef[1], 50)
    if optima2 == None:
        direction = filter_norm_direction(optima)
        optima2 = sum_state_dicts(optima1, direction, 0.5)

    losses = []
    for alpha in tqdm(grid):
        weights = sum_state_dicts(optima1, optima2, alpha)
        model.load_state_dict(weights)
        loss = criterion(model(x), y)
        losses.append(loss.item())
    return grid, losses


def plot_1D(metrics: pd.DataFrame) -> figure:
    """
    Plots 1-dimensional linear interpolation of loss function between two solutions.

    Parameters
    ----------

    Returns
    -------

    """
    grid = metrics['grid']
    loss = metrics['loss']

    plt.figure()
    plt.plot(grid, loss)
    plt.title('')
    plt.xlabel('Interpolation coefficient')
    plt.ylabel('Loss')
    plt.show()
