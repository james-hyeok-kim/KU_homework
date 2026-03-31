import torch
import torch.nn as nn
from torch.distributions.normal import Normal 

class Flow1d(nn.Module):
    def __init__(self, n_components):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

    def forward(self, x):
        '''
            Define z and dz/dx
            This should be same as the one you wrote in Q1

            - z: CDF of mixture of Gaussian distribution  
                > use weights and mu, sigma defined above. 
                > z = \sum_{i=1}^{n_components} w_{i} x CDF(x)

            - dz/dx: 
                > use weights and mu, sigma defined above. 
                > dz/dx = \sum_{i=1}^{n_components} w_{i} x PDF(x)

            Becareful with you dimensions!

        '''
        x = x.view(-1,1)
        weights = self.weight_logits.softmax(dim=0).view(1,-1)
        distribution = Normal(self.mus, self.log_sigmas.exp())

        ### FILL IN ######################################################
        # z = \sum w_i * CDF_i(x)
        z = (weights * distribution.cdf(x)).sum(dim=1, keepdim=True)
        # dz/dx = \sum w_i * PDF_i(x)
        dz_by_dx = (weights * distribution.log_prob(x).exp()).sum(dim=1, keepdim=True)
        # Stability Epsilon
        log_dz_by_dx = torch.log(dz_by_dx + 1e-8)
        ##################################################################
        return z, log_dz_by_dx

class LogitTransform(nn.Module):
    '''
        Define z and dz/dx
        You need to used the following definitions
    '''
    def __init__(self, alpha):
        super(LogitTransform, self).__init__()
        self.alpha = alpha 

    def forward(self, x):
        x_new = self.alpha/2 + (1-self.alpha)*x 
        z = torch.log(x_new) - torch.log(1-x_new)
        ### FILL IN ######################################################
        # dz/dx = (dz/dx_new) * (dx_new/dx)
        # dz/dx_new = 1/x_new + 1/(1-x_new) = 1 / (x_new * (1 - x_new))
        # dx_new/dx = (1 - alpha)
        # log(dz/dx) = log(1 - alpha) - log(x_new) - log(1 - x_new)
        log_dz_by_dx = torch.log(torch.tensor(1 - self.alpha)) - torch.log(x_new) - torch.log(1 - x_new)
        ##################################################################
        
        return z, log_dz_by_dx
        

class FlowComposable1d(nn.Module):
    def __init__(self, flow_models_list):
        super(FlowComposable1d, self).__init__()
        self.flow_models_list = nn.ModuleList(flow_models_list)

    def forward(self, x):

        ### FILL IN ######################################################
        z = x.view(-1, 1) 
        
        sum_log_dz_by_dx = torch.zeros_like(z)
        
        for model in self.flow_models_list:
            z, log_dz_by_dx = model(z)
            sum_log_dz_by_dx += log_dz_by_dx
        ##################################################################
        return z, sum_log_dz_by_dx