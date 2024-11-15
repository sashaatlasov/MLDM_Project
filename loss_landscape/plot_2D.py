import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import figure, cm
import torch
from typing import Tuple, Dict
from tqdm.auto import tqdm
from utils import filter_norm_direction

def sum_state_dicts(optima1, optima2, optima3, alpha, beta):
    sum = {}
    for block in optima1.keys():
        sum[block] = optima1[block] + alpha * optima2[block] + beta * optima3[block]
    return sum

def calculate_metrics(model, criterion, optima, x, y, coef=(-0.5, 1.5)):

    grid = np.linspace(coef[0], coef[1], 50)
    direction1 = filter_norm_direction(optima)
    direction2 = filter_norm_direction(optima)

    losses = []
    for alpha in grid:
        for beta in grid:
            weights = sum_state_dicts(optima, direction1, direction2, alpha, beta)
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
    x, y = np.meshgrid(grid, grid)
    z = np.array(metrics['loss']).reshape(x.shape[0], x.shape[0])

    plt.figure()
    CS = plt.contour(x, y, z, cmap='summer', levels=np.arange(np.min(z), np.max(z), 0.5))
    plt.clabel(CS, inline=1, fontsize=8)
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.show()
