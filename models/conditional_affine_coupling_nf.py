"""
Torch/Pyro Implementation of (Unconditional) Normalizing Flow. In particular, as seen in RealNVP (z = x * exp(s) + t), where half of the 
dimensions in x are linearly scaled/transfromed as a function of the other half.
"""
import pyro
import torch
from torch import nn
from pyro.nn.dense_nn import ConditionalDenseNN
import pyro.distributions as dist
from pyro.distributions.transforms import permute, BatchNorm
from pyro.distributions.transforms.affine_coupling import ConditionalAffineCoupling
import itertools

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, input_dim=2, split_dim=1, context_dim=1, hidden_dim=128, num_layers=1, flow_length=10, 
                use_cuda=False):
        super(ConditionalNormalizingFlow, self).__init__()
        self.base_dist = dist.Normal(torch.zeros(input_dim), torch.ones(input_dim)) # base distribution is Isotropic Gaussian
        self.param_dims = [input_dim-split_dim, input_dim-split_dim]
        # Define series of bijective transformations
        self.transforms = [ConditionalAffineCoupling(split_dim, ConditionalDenseNN(split_dim, context_dim, [hidden_dim]*num_layers, self.param_dims)) for _ in range(flow_length)]
        self.perms = [permute(2, torch.tensor([1,0])) for _ in range(flow_length)]
        self.bns = [BatchNorm(input_dim=1) for _ in range(flow_length)]
        # Concatenate AffineCoupling layers with Permute and BatchNorm Layers
        self.generative_flows = list(itertools.chain(*zip(self.transforms, self.bns, self.perms)))[:-2] # generative direction (z-->x)
        self.normalizing_flows = self.generative_flows[::-1] # normalizing direction (x-->z)
        
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()
            nn.ModuleList(self.transforms).cuda()
    
    def model(self, X=None, H=None):
        N = len(X) if X is not None else None
        pyro.module("nf", nn.ModuleList(self.transforms))
        with pyro.plate("data", N):
                self.cond_flow_dist = self._condition(H)
                obs = pyro.sample("obs", self.cond_flow_dist, obs=X)
            
    def guide(self, X=None, H=None):
        pass
    
    def forward(self, z, H):
        zs = [z]
        _ = self._condition(H)
        for flow in self.generative_flows:
            z_i = flow(zs[-1])
            zs.append(z_i)
        return zs, z_i
    
    def backward(self, x, H):
        zs = [x]
        _ = self._condition(H)
        for flow in self.normalizing_flows:
            z_i = flow._inverse(zs[-1])
            zs.append(z_i)
        return zs, z_i
    
    def sample(self, num_samples, H):
        z_0_samples = self.base_dist.sample([num_samples])
        zs, x = self.forward(z_0_samples, H)
        return x
    
    def log_prob(self, x, H):
        cond_flow_dist = self._condition(H)
        return cond_flow_dist.log_prob(x)
    
    def _condition(self, H):
        self.cond_transforms = [t.condition(H) for t in self.transforms]
        self.generative_flows = list(itertools.chain(*zip(self.cond_transforms, self.perms)))[:-1]
        self.normalizing_flows = self.generative_flows[::-1]
        return dist.TransformedDistribution(self.base_dist, self.generative_flows)