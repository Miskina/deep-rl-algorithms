import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

import numpy as np


class SimplePGModel:

    def __init__(self, dimensions_policy, dimensions_value, hidden_activation_policy=nn.ReLU, hidden_activation_value=nn.ReLU):
        #super(PGAgent, self).__init__()

        def make_ann(dimensions, hidden_activation):
            layers = []
            D_in = dimensions[0]
            n = len(dimensions)
            for i in range(1, n):
                D_out = dimensions[i]
                layers += [nn.Linear(D_in, D_out, (hidden_activation() if i < n - 1 else nn.Identity()))]
                D_in = D_out
            #print('Initilaizing ANN model with layers:')
            #print(*layers)
            return nn.Sequential(*layers)

        if dimensions_value[len(dimensions_value) - 1] != 1:
            dimensions_value += [1]

        self.policy_model = make_ann(dimensions_policy, hidden_activation_policy)
        self.value_model = make_ann(dimensions_value, hidden_activation_value)
    
    def policy_paramters(self):
        return self.policy_model.parameters()
    
    def value_parameters(self):
        return self.value_model.parameters()

    def policy_distribution(self, observation):
        return Categorical(logits=self.policy_model(torch.as_tensor(observation, dtype=torch.float32)))

    def sample_action(self, observation):
        return self.policy_distribution(observation).sample().item()

    def predict_value(self, observation):
        return self.value_model(observation)
        
def train(env, model, policy_opt, value_opt, epochs=50, batch_size=1000, gamma=0.99, lam=0.97, render=False):


    def baselined_reward_to_go(rewards, observations, baseline_function):
        baselined_to_go = np.zeros_like(rewards)
        rewards_to_go = np.zeros_like(rewards)
        vs = np.zeros_like(rewards)
        n = len(rewards)
        #import pdb
        #pdb.set_trace()
        for i in range(n):
            vs[i] = baseline_function(observations[i])

        for i in reversed(range(n)):
            not_last = i < (n - 1)
            rewards_to_go[i] = rewards[i]
            baselined_to_go[i] = rewards[i] - vs[i]
            if not_last:
                baselined_to_go[i] += gamma * vs[i + 1] + gamma * lam * baselined_to_go[i + 1]
                rewards_to_go[i] += gamma * rewards_to_go[i + 1]
            #baselined_to_go[i] = rewards[i] + gamma * (vs[i + 1] if not_last else 0) - vs[i] + gamma * lam * (baselined_to_go[i + 1] if not_last else 0)
        return baselined_to_go, rewards_to_go

    def compute_loss(obs, adv, actions, old_logp):
        log_probs, entropy = actor(obs)
        log_probs = log_probs.gather(1, actions.unsqueeze(-1))
        ratio = torch.exp(log_probs - old_logp)

        clipped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        return torch.mean(-torch.min(ratio * adv, clipped * adv)) - entropy_coeff * entropy.mean()
        
    def compute_loss(obs, act, weights_policy, target_value):
        policy_loss = -(model.policy_distribution(obs).log_prob(act) * weights_policy).mean()
        value_loss = torch.mean((model.predict_value(obs) - target_value.reshape(-1, 1)) ** 2)
        return policy_loss, value_loss

    bl_function = lambda obs : model.predict_value(torch.as_tensor(obs, dtype=torch.float32)).item()
    def train_epoch():

        obs_batch = []
        action_batch = []
        rewards_batch = []
        baselined_batch = []
        returns_batch = []
        lengths_batch = []

        obs, done = env.reset(), False
        
        episode_rewards = []

        rendered_first_episode = False

        while True:
            
            if render and not rendered_first_episode:
                env.render()
            
            obs_batch.append(obs)
            #import pdb
            #pdb.set_trace()
            act = model.sample_action(obs)

            obs, reward, done, _ = env.step(act)

            action_batch.append(act)

            episode_rewards.append(reward)

            if done:
                returns_batch.append(sum(episode_rewards))
                lengths_batch.append(len(episode_rewards))
                
                advantages, rewards = baselined_reward_to_go(episode_rewards, obs_batch, bl_function)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                rewards_batch += list(rewards)
                baselined_batch += list(advantages)

                rendered_first_episode = True

                obs, done = env.reset(), False
                episode_rewards = []

                if len(obs_batch) > batch_size:
                    break
        
        policy_opt.zero_grad()
        value_opt.zero_grad()

        
        #pdb.set_trace()
        policy_loss, value_loss = compute_loss(obs=torch.as_tensor(obs_batch, dtype=torch.float32),
                                               act=torch.as_tensor(action_batch, dtype=torch.float32),
                                               weights_policy=torch.as_tensor(baselined_batch, dtype=torch.float32),
                                               target_value=torch.as_tensor(rewards_batch, dtype=torch.float32))

        policy_loss.backward()
        value_loss.backward()

        policy_opt.step()
        value_opt.step()

        return policy_loss, value_loss, returns_batch, lengths_batch
    
    for i in range(1, epochs + 1):
        policy_loss, value_loss, returns, lengths = train_epoch()
        print('Epoch: %3d \t Policy loss: %.3f \t Value loss: %.3f \t Avg Return: %.3f \t Avg Episode length: %.3f'%
                (i, policy_loss, value_loss, np.mean(returns), np.mean(lengths)))


if __name__ == '__main__':
    import argparse
    import gym
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    #parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing reward-to-go formulation of policy gradient.\n')
    env = gym.make(args.env_name)
    #train(env_name=args.env_name, render=args.render, lr=args.lr)
    model = SimplePGModel(dimensions_policy=[env.observation_space.shape[0], 32, env.action_space.n], 
                          dimensions_value=[env.observation_space.shape[0], 32, 1])
    
    opt_val = optim.SGD(model.value_parameters(), lr=1e-4)
    opt_policy = optim.Adam(model.policy_paramters(), lr=1e-2)

    train(env=env,
          model=model,
          policy_opt=opt_policy,
          value_opt=opt_val,
          epochs=50,
          batch_size=1000,
          gamma=1,
          render=args.render) #args.render


