import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def from_numpy(np_array):
    return torch.from_numpy(np_array).float().to(device)

def get_numpy(tensor):
    return tensor.to("cpu").detach().numpy()
