import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
from torch.distributions.categorical import Categorical

import pdb

from gym_wrappers import TorchWrapper

EPS = 1e-8

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
          backtract_K=10,
          delta=0.05,
          epochs=50,
          batch_size=1000,
          gamma=0.99,
          lam=0.97,
          cg_steps=10,
          cg_damping=1e-1,
          entropy_coeff=0.00,
          train_critic_iters=3):

    
    def rewards_and_advantages(rewards, obs, critic):
        n = len(rewards)
        adv_to_go = torch.zeros(n)
        rewards_to_go = torch.zeros(n)
        vs = torch.zeros(n)
        for i in range(n):
            vs[i] = critic(obs[i])

        for i in reversed(range(n)):
            not_last = i < (n - 1)
            rewards_to_go[i] = rewards[i]
            adv_to_go[i] = rewards[i] - vs[i]

            if not_last:
                adv_to_go[i] += gamma * vs[i + 1] + gamma * lam * adv_to_go[i + 1]
                rewards_to_go[i] += gamma * rewards_to_go[i + 1]
            
        return adv_to_go, rewards_to_go




    def compute_loss(prob_ratio, advantages, entropy):
        return torch.mean(prob_ratio * advantages) + entropy_coeff * entropy

    def conjugate_gradient(Fx, b, nsteps=10, residaul_tolerane=1e-10):
        x = torch.zeros(b.size())
        r = b.clone().data
        p = b.clone().data
        rdotr = torch.dot(r, r)

        for i in range(nsteps):
            F_x = Fx(p)
            alpha = rdotr / torch.dot(p, F_x)
            x += alpha * p
            r -= alpha * F_x
            rdotr_new = torch.dot(r, r)
            beta = rdotr_new / rdotr
            p = r + beta * p
            rdtor = rdotr_new
            if rdtor < residaul_tolerane:
                break
        return x

    def fisher_vector_product(x, kl_function):

        actor.zero_grad()
        mean_kl = kl_function(actor).mean()
        gradients = torch.autograd.grad(mean_kl, actor.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in gradients])

        grad_kl_x = torch.dot(flat_grad_kl, x)
        gradients = torch.autograd.grad(grad_kl_x, actor.parameters())
        flat_second_grad = torch.cat([grad.contiguous().view(-1) for grad in gradients]).data

        return flat_second_grad + x.data * cg_damping

    def kl_divergence(obs, old_model, model):

        log_probs, _ = model(obs)
        log_probs = log_probs.detach() + EPS
        old_log_probs, _ = old_model(obs)
        #pdb.set_trace()
        return torch.sum(torch.exp(old_log_probs) * (old_log_probs - log_probs), dim=1)
    

    def loss_and_kl(model, new_params, obs, adv, actions, old_log_probs):
        with torch.no_grad():
            #old_params = parameters_to_vector(model.parameters())
            #pdb.set_trace()
            vector_to_parameters(new_params, model.parameters())

            log_probs_new, entropy = model(obs)
            log_probs_new = log_probs_new.gather(1, actions.unsqueeze(-1))
            #vector_to_parameters(old_params, model.parameters())
            return surr_loss_kl(old_log_probs, log_probs_new, adv, entropy)


    def surr_loss_kl(old_log_probs, probs_log, adv, entropy):
        log_ratio = probs_log - old_log_probs
        #ratio = probs / old_probs
        return -torch.mean(torch.exp(log_ratio) * adv) - entropy_coeff * entropy.mean(), (torch.exp(old_log_probs) * -log_ratio).mean()

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

            obs, reward, done, _ = env.step(act)

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
        
        
        #import pdb
        #pdb.set_trace()

        #value_opt.zero_grad()


        # TODO Prebaci u torch.Optimizer razred za lakše razumijevanje i lakše korištenje

        # spoji listu vjerojatnosti u tenzor i odaberi vjerojatnosti odabranih akcija u batchu
        #new_log_probabilites = torch.cat(log_probs).gather(1, torch.cat(action_batch))
        entropy = torch.stack(entropies)
        obs_tensor = torch.stack(obs_batch).data
        adv_tensor = torch.cat(advantages_batch).data
        reward_tensor = torch.cat(rewards_batch).data
        action_tensor = torch.as_tensor(action_batch)

        log_probabilities = torch.stack(log_probs)
        old_log_probabilities = log_probabilities.detach() + EPS # dodano EPS zbog dijljenja, vrlo mali broj


        #pdb.set_trace()
        policy_loss, mean_kl = surr_loss_kl(old_log_probs=old_log_probabilities, probs_log=log_probabilities, entropy=entropy, adv=adv_tensor)

        actor.zero_grad()
        #policy_loss = compute_policy_loss(prob_ratio, adv_tensor, entropy)
        policy_loss.backward(retain_graph=True) # surrogate_loss, retain_graph jer nam kasnije treba Hess

        policy_gradient = parameters_to_vector([params.grad for params in actor.parameters()])

        kl_function = lambda model: kl_divergence(obs=obs_tensor, old_model=actor, model=model)
        #pdb.set_trace()
        fv = lambda x : fisher_vector_product(x, kl_function)
        # smjer koraka
        #pdb.set_trace()
        x = conjugate_gradient(fv, -policy_gradient, nsteps=cg_steps)

        shs = 0.5 * x.dot(fv(x))

        # constraint: (1 / 2) * (O - Ok).T * H (O - Ok) <= delta  ==> (1 / 2 * delta) * ... <= 1
        # razlike su zapravo u ovom slucaju 'x'
        lagrange_multiplier = torch.sqrt(shs / delta)
        print(f'Lagrange Multiplier: {lagrange_multiplier}, gradient norm: {policy_gradient.norm()}')

        fullstep = x / lagrange_multiplier
        expected_improve = -policy_gradient.dot(fullstep)
        #pdb.set_trace()
    
        current_params = parameters_to_vector([params for params in actor.parameters()])
        #params = line_search(actor, current_params, loss_function, fullstep, expected_improve)
        #params = line_search(actor, current_params, loss_function, fullstep)
        step_size = 1.0
        kl_old_new = 0.0
        output_loss = policy_loss.item()
        old_loss = policy_loss.detach()
        for _ in range(backtract_K):
            #pdb.set_trace()
            params_new = current_params.data + fullstep * step_size
            
            loss_new, kl = loss_and_kl(model=actor,
                                       new_params=params_new,
                                       obs=obs_tensor,
                                       adv=adv_tensor,
                                       actions=action_tensor,
                                       old_log_probs=old_log_probabilities.data)
            improvement = loss_new - old_loss
            #print(f'Improve: {improvement}, Expected improve: {expected_improve}, KL: {kl}')
            #expected = expected_improve * step_size
            #ratio = improvement / expected_improve
            if kl <= delta * 1.5 and improvement.item() > 0:
            #if ratio.item() > 0.1 and improvement > 0: 
                print('Good improvement!')
                kl_old_new = kl
                output_loss = loss_new.item()
                break
            else:
                #pdb.set_trace()
                print(f'Failed, violated constraint: {kl > delta * 1.5}, improvement negative: {improvement.item() <= 0} -- lowering step')
                step_size *= 0.5
        else:
            print(f'No good parameters within region -- rolling back')
            vector_to_parameters(current_params, actor.parameters())


        # provjeri za nan i inf?
        #vector_to_parameters(params, actor.parameters())
        print(f'Training value function (critic) for {train_critic_iters} iterations')
        for _ in range(train_critic_iters):
            value_opt.zero_grad()
            value_loss = torch.mean((critic(obs_tensor) - reward_tensor) ** 2) * 0.5
            value_loss.backward()
            value_opt.step()

        return output_loss, value_loss, returns_batch, kl_old_new
    
    for i in range(1, epochs + 1):
        policy_loss, value_loss, returns, kl = train_epoch()
        print('Epoch: %3d \t Policy loss: %.3f \t Value loss: %.3f \t Avg Return: %.3f \t KL: %.3f'%
                (i, policy_loss, value_loss, torch.mean(torch.as_tensor(returns)), kl))


if __name__ == '__main__':
    import argparse
    import gym
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    #parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing Trust Region Policy Gradient.\n')
    env = TorchWrapper(gym.make(args.env_name))
    #train(env_name=args.env_name, render=args.render, lr=args.lr)
    policy_model = Actor(dims=[env.observation_space.shape[0], 32, env.action_space.n])
    value_model = Critic(dims=[env.observation_space.shape[0], 32, 1])
    
    opt_val = optim.SGD(value_model.parameters(), lr=1e-4)
    #opt_policy = optim.Adam(model.policy_paramters(), lr=1e-2)

    train(env,
          policy_model,
          value_model,
          value_opt=opt_val)









