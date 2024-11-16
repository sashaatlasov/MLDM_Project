import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import figure, cm
import torch
from typing import Tuple, Dict
from tqdm.auto import tqdm
from loss_landscape.utils import filter_norm_direction


def sum_state_dicts(optima: dict, dir1: dict, dir2: dict, alpha: float, beta: float) -> dict:
    """
    Shifts model's optima in two defined direction with corresponding coefficients.

    Parameters
    ----------
    optima : dict
        model state dict, optimal weights
    dir1 : dict
        first random direction in parameters space
    dir2 : dict
        second random direction in parameters space
    alpha : float
        dir1 coefficient
    beta : float
        dir2 coefficient

    Returns
    -------
    sum : dict
        shifted optima
    """
    sum = {}
    for block in optima.keys():
        sum[block] = optima[block] + alpha * \
            dir1[block] + beta * dir2[block]
    return sum


def calculate_metrics(model, criterion, optima, x, y, coef=(-1., 1.), num_steps: int = 50):

    grid = np.linspace(coef[0], coef[1], num_steps)
    direction1 = filter_norm_direction(optima)
    direction2 = filter_norm_direction(optima)

    losses = []
    for alpha in tqdm(grid):
        for beta in grid:
            weights = sum_state_dicts(
                optima, direction1, direction2, alpha, beta)
            model.load_state_dict(weights)
            loss = criterion(model(x), y)
            losses.append(loss.item())
    return json.dumps({'grid': list(grid), 'loss': list(losses)})


def plot_2D(metrics: json, vlevel: float = 0.5) -> figure:
    """
    Plots 2-dimensional linear interpolation of loss function between two solutions.

    Parameters
    ----------

    Returns
    -------

    """
    metrics = json.loads(metrics)
    grid = np.array(metrics['grid'])
    x, y = np.meshgrid(grid, grid)
    z = np.array(metrics['loss']).reshape(x.shape[0], x.shape[0])

    plt.figure(dpi=300)
    plt.rcParams['text.usetex'] = True
    CS = plt.contour(x, y, z, cmap='summer',
                     levels=np.arange(np.min(z), np.max(z), vlevel))
    plt.clabel(CS, inline=1, fontsize=8)
    plt.title('Contour plot around an optima')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.show()


def plot_3D(metrics: json) -> figure:
    """

    Parameters
    ----------

    Returns
    -------
    """
    metrics = json.loads(metrics)
    grid = np.array(metrics['grid'])
    x, y = np.meshgrid(grid, grid)
    z = np.array(metrics['loss']).reshape(x.shape[0], x.shape[0])

    fig = plt.figure(dpi=300)
    plt.rcParams['text.usetex'] = True
    ax = fig.add_subplot(projection='3d')

    ax.plot_surface(x, y, z, edgecolor='k', linewidth=0.3, cmap=cm.coolwarm)

    ax.set_title("3D Loss landscape")
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    plt.show()
