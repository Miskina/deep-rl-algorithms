import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import pdb

from gym_wrappers import TorchWrapper
import core

EPS = 1e-8

def make_ann(dimensions, hidden_activation):
    layers = []
    D_in = dimensions[0]
    n = len(dimensions)
    for i in range(1, n):
        D_out = dimensions[i]
        layers += [nn.Linear(D_in, D_out, (hidden_activation() if i < n - 1 else nn.Identity()))]
        D_in = D_out
    return nn.Sequential(*layers)

class Actor(nn.Module):

    def __init__(self, dims, hidden_activation=nn.ReLU):
        super(Actor, self).__init__()
        self.model = make_ann(dims, hidden_activation)
    
    def policy_distribution(self, obs):
        return Categorical(logits=self.model(obs))

    def sample_action(self, obs):
        dist = self.policy_distribution(obs)
        act = dist.sample()
        return act.item(), dist.log_prob(act), dist.entropy()

    def __call__(self, obs):
        dist = self.policy_distribution(obs)
        return torch.log(dist.probs), dist.entropy()

class Critic(nn.Module):

    def __init__(self, dims, hidden_activation=nn.ReLU):
        super(Critic, self).__init__()
        if dims[len(dims) - 1] > 1:
            dims += [1]
        self.model = make_ann(dims, hidden_activation)
    
    def forward(self, obs):
        return self.model(obs)

    def __call__(self, obs):
        return self.forward(obs)
    
    def parameters(self):
        return self.model.parameters()


def train(env,
          actor,
          critic,
          value_opt,
          actor_opt,
          max_kl=0.1,
          epsilon=0.2,
          epochs=50,
          batch_size=1000,
          gamma=0.99,
          lam=0.96,
          val_loss_coef=0.5,
          entropy_coeff=0.000,
          train_critic_iters=10,
          train_actor_iters=10):

    
    def rewards_and_advantages(rewards, obs, critic):
        n = len(rewards)
        adv_to_go = torch.zeros(n)
        rewards_to_go = torch.zeros(n)
        #vs = torch.zeros(n)
        #for i in range(n):
        #    vs[i] = critic(obs[i])
        # with torch.no_grad():
            # vs = critic(torch.stack(obs)).squeeze()
        vs = critic(torch.stack(obs)).squeeze()
        for i in reversed(range(n)):
            not_last = i < (n - 1)
            rewards_to_go[i] = rewards[i]
            adv_to_go[i] = rewards[i] - vs[i]

            if not_last:
                adv_to_go[i] += gamma * vs[i + 1] + gamma * lam * adv_to_go[i + 1]
                rewards_to_go[i] += gamma * rewards_to_go[i + 1]
            
        return adv_to_go, rewards_to_go
    
    def compute_loss(obs, adv, actions, old_logp):
        log_probs, entropy = actor(obs, actions)
        #log_probs = log_probs.gather(1, actions.unsqueeze(-1))
        ratio = torch.exp(log_probs - old_logp)

        clipped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        return torch.mean(-torch.min(ratio * adv, clipped * adv)) - entropy_coeff * entropy.mean()

    def loss_and_kl(obs, adv, actions, old_log_probs):
        log_probs, entropy = actor(obs, actions)
        #log_probs = log_probs.gather(1, actions.unsqueeze(-1))
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        clipped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        loss = torch.mean(-torch.min(ratio * adv, clipped * adv)) + entropy_coeff * entropy.mean()

        kl = (torch.exp(log_probs) * log_ratio).mean()

        return loss, kl

    def train_epoch():

        obs_batch = []
        action_batch = []
        log_probs = []
        entropies = []
        rewards_batch = []
        advantages_batch = []
        returns_batch = []
        
        obs, done = env.reset(), False
        
        episode_rewards = []
        #import pdb
        #pdb.set_trace()
        while True:
            
            obs_batch.append(obs)

            act, log_prob, entrpy = actor.sample_action(obs)
            #entropy += 

            obs, reward, done, _ = env.step(act.item())

            action_batch.append(act)
            log_probs.append(log_prob)
            entropies.append(entrpy)
            episode_rewards.append(reward)

            if done:
                returns_batch.append(sum(episode_rewards))
                #lengths_batch.append(len(episode_rewards))
                #obs_batch_tensor = torch.cat([ob.view(1, -1) for ob in obs_batch])
                advantages, rewards = rewards_and_advantages(episode_rewards, obs_batch, critic)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                rewards_batch.append(rewards)
                advantages_batch.append(advantages)

                obs, done = env.reset(), False
                episode_rewards = []
                if len(obs_batch) > batch_size:
                    break
        #pdb.set_trace()
        entropy = torch.stack(entropies)
        obs_tensor = torch.stack(obs_batch).data
        adv_tensor = torch.cat(advantages_batch).data
        reward_tensor = torch.cat(rewards_batch).data
        action_tensor = torch.as_tensor(action_batch).data
       
        log_probabilities = torch.stack(log_probs)
        old_log_probabilities = log_probabilities.detach() + EPS # dodano EPS zbog dijljenja, vrlo mali broj

        #prob_ratio = (log_probabilities - old_log_probabilities).exp()
        #pdb.set_trace()
        #loss_clip, kl_old_new = 0.0, 0.0
        loss_clip, kl_old_new = torch.tensor(0.0), torch.tensor(0.0)
        print(f'Training policy (actor) for {train_actor_iters} iterations')
        for i in range(1, train_actor_iters + 1):
            actor_opt.zero_grad()
            #loss_clip = compute_loss(obs=obs_tensor, adv=adv_tensor, actions=action_tensor, old_logp=old_log_probabilities)
            loss_clip, kl = loss_and_kl(obs=obs_tensor, adv=adv_tensor, actions=action_tensor, old_log_probs=old_log_probabilities)
            if kl > 1.5 * max_kl:
                 print(f'Early stopping at iteration {i}, reached max KL')
                 break
            kl_old_new = kl
            loss_clip.backward()
            actor_opt.step()


        # provjeri za nan i inf?
        #vector_to_parameters(params, actor.parameters())
        print(f'Training value function (critic) for {train_critic_iters} iterations')
        for _ in range(train_critic_iters):
            value_opt.zero_grad()
            value_loss = torch.mean((critic(obs_tensor) - reward_tensor) ** 2) * val_loss_coef
            value_loss.backward()
            value_opt.step()

        return loss_clip, value_loss, returns_batch, kl_old_new
    
    for i in range(1, epochs + 1):
        policy_loss, value_loss, returns, kl = train_epoch()
        print('Epoch: %3d \t Policy loss: %.3f \t Value loss: %.3f \t Avg Return: %.3f \t KL: %.3f'%
                (i, policy_loss, value_loss, torch.mean(torch.as_tensor(returns)), kl))


def test(env, model, rec_dir, episodes):

    env = gym.wrappers.Monitor(env, rec_dir, force=True)
    model.eval()

    for _ in range(episodes):
        obs, done = env.reset(), False

        while not done:
            
            with torch.no_grad():
                act = model.sample_action(obs)
            obs, _, done, _ = env.step(act)

if __name__ == '__main__':
    import argparse
    import gym
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    #parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing Proximal Policy Gradient.\n')
    env = TorchWrapper(gym.make(args.env_name))
    #train(env_name=args.env_name, render=args.render, lr=args.lr)
    #policy_model = Actor(dims=[env.observation_space.shape[0], 32, env.action_space.n])
    #policy_model = core.GaussianActor(core.make_ann([env.observation_space.shape[0], 64, env.action_space.shape[0]], nn.ReLU), env.action_space.shape[0])
    policy_model = core.CategoricalActor(core.make_ann([env.observation_space.shape[0], 16, 32, env.action_space.n], hidden_activation=nn.ReLU))
    value_model = Critic(dims=[env.observation_space.shape[0], 64, 1])
    
    opt_val = optim.Adam(value_model.parameters(), lr=3e-4)
    opt_policy = optim.SGD(policy_model.parameters(), lr=3e-4)

    train(env,
          actor=policy_model,
          critic=value_model,
          value_opt=opt_val,
          actor_opt=opt_policy)

    test(env, policy_model, 'ppo_proba', 5)







