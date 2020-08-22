import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from abc import abstractmethod

class Actor(nn.Module):

    def __init__(self, model):
        super(Actor, self).__init__()
        self.model = model

    @abstractmethod
    def policy_distribution(self, obs):
        pass

    def sample_action(self, obs):
        with torch.no_grad():
            dist = self.policy_distribution(obs)
            act = dist.sample()
            return act.data, dist.log_prob(act), dist.entropy()
    
    def __call__(self, obs, act):
        dist = self.policy_distribution(obs)
        return dist.log_prob(act), dist.entropy()


class CategoricalActor(Actor):

    def __init__(self, model):
        super(CategoricalActor, self).__init__(self)
        self.model = model
    
    def policy_distribution(self, obs):
        return Categorical(logits=self.model(obs))


class GaussianActor(Actor):

    def __init__(self, model, act_dim):
        super(GaussianActor, self).__init__(model)
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim, dtype=torch.float32))

    def policy_distribution(self, obs):
        mi = self.model(obs)
        return Normal(mi, torch.exp(self.log_std))
    

def make_ann(dimensions, hidden_activation, prev_layers=None):
    layers = []
    
    if prev_layers is not None:
        layers += prev_layers
    
    D_in = dimensions[0]
    n = len(dimensions)
    for i in range(1, n):
        D_out = dimensions[i]
        layers += [nn.Linear(D_in, D_out, (hidden_activation() if i < n - 1 else nn.Identity()))]
        D_in = D_out
    #print('Initilaizing ANN model with layers:')
    #print(*layers)
    return nn.Sequential(*layers)
