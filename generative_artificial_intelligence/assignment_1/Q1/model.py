import torch
import torch.nn as nn
from torch.distributions.normal import Normal 
from torch.distributions import Categorical
from torch.nn.functional import one_hot

class Flow1d(nn.Module):
    def __init__(self, n_components):
        super(Flow1d, self).__init__()
        '''
            Define parameters we want to learn
            Note that log_sigma and weight_logits are defined instead (for better stability during training)
        '''
        self.n_components = n_components

        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

        

    def forward(self, x):
        '''
            Define z and dz/dx
            You need to used the following definitions

            - z: CDF of mixture of Gaussian distribution  
                > use weights and mu, sigma defined above. 
                > z = \sum_{i=1}^{n_components} w_{i} x CDF(x)

            - dz/dx: 
                > use weights and mu, sigma defined above. 
                > dz/dx = \sum_{i=1}^{n_components} w_{i} x PDF(x)

            Becareful with you dimensions!

        '''
        x = x.view(-1,1)
        weights = self.weight_logits.softmax(dim=0).view(1,-1)  #softmax function is used to make sure w_i > 0 and sum w_i = 1.
        distribution = Normal(self.mus, self.log_sigmas.exp())

        ### FILL IN ######################################################
        cdfs = distribution.cdf(x)
        pdfs = distribution.log_prob(x).exp()
        # Mixture of Gaussians (z, dz/dx)
        z = (weights * cdfs).sum(dim=1)
        dz_by_dx = (weights * pdfs).sum(dim=1)
        ##################################################################
        return z, dz_by_dx
    
    
    def generate(self, z):
        '''
            Define function "sample" to generate x from z 
            - Becareful that you are drawing x from a "mixture of Gaussian"
                > You must draw Gaussian samples from each Gaussian
                > You must "ranomly" select one (membership) from a categorical distribution with trained weights  --> use Categorical api
            
        '''
        z = z.view(-1,1)
        weights = self.weight_logits.softmax(dim=0).view(1,-1)  #softmax function is used to make sure w_i > 0 and sum w_i = 1.
        distribution = Normal(self.mus, self.log_sigmas.exp())

        distribution_categorical = Categorical(weights)

        with torch.no_grad():
            ### FILL IN ######################################################
            # Gausian index
            batch_size = z.size(0)
            indices = distribution_categorical.sample((batch_size,))
            # Data
            samples = distribution.sample((batch_size,))
            # Gather
            x_hat = samples.gather(1, indices.view(-1, 1)).squeeze()
            ##################################################################

        return x_hat

