import torch
from ABC import abstractmethod

class Actor(nn.Module):

    @abstractmethod
    def policy_distribution(self, obs):
        pass

    @abstractmethod
    def log_from_dist(self, distribution, action):
        pass

    def forward(self, obs, act=None):
        distrbution = self.policy_distribution(obs)
        log_probability = self.log_from_dist(distribution, act) if act is not None else None
        return distrbution, log_probability

class Critic(nn.Module):

    @abstractmethod
    def _forward_impl(self, obs):
        pass

    def forward(self, obs):
        return torch.squeeze(self._forward_impl(obs), -1)
