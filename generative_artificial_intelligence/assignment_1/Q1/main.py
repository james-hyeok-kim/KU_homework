
import torch

from model import Flow1d
from data import train_loader, test_loader
import matplotlib.pyplot as plt
from torch.distributions import Uniform
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # GPU를 사용하지 않겠다고 명시

def loss_function(target_distribution, z, dz_by_dx):
    '''
        Use negative log-likelihood as your loss function
        - Note that log-likelihood is given as log p_X(x) = log p_Z(z) + log dz/dx 
    '''
    ### FILL IN ######################################################
    # pz Uniform
    log_pz = target_distribution.log_prob(z)
    # Stability 1e-8
    log_px = log_pz + torch.log(dz_by_dx + 1e-8)
    log_likelihood = log_px
    ##################################################################
    return -log_likelihood.mean()

def train(model, train_loader, optimizer, target_distribution):
    model.train()
    for x in train_loader:
        z, dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def eval_loss(model, data_loader, target_distribution):
    model.eval()
    total_loss = 0
    for x in data_loader:
        z, dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, dz_by_dx)
        total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).item()

def train_and_eval(epochs, lr, train_loader, test_loader, target_distribution, n_components=5):
    flow = Flow1d(n_components=n_components)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        train(flow, train_loader, optimizer, target_distribution)
        train_losses.append(eval_loss(flow, train_loader, target_distribution))
        test_losses.append(eval_loss(flow, test_loader, target_distribution))
    return flow, train_losses, test_losses

