import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def train_epoch(model, dataloader, optimizer, criterion, device):
    
    losses = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        return np.mean(losses)
        

def train(model, dataloader, optimizer, criterion, 
          num_epochs=50, device='cpu', verbose=True, plot=True):
    
    losses = []
    for i in tqdm(range(num_epochs)):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        losses.append(loss)
        if verbose:
            print(f"Epoch {i + 1}/{num_epochs} | Loss {loss}")
    if plot:
        plt.figure(dpi=200, figsize=(10, 8))
        sns.set()
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
