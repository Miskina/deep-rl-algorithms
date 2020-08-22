import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

import gym
import random
import pdb

import gym_wrappers as wrappers

class EpsilonGreedyStrategy:

    def __init__(self, epsilon=0.9, decay_rate=0.99, min_epsilon=0.1):
        self.epsilon = epsilon
        self.decay = decay_rate
        self.min_eps = min_epsilon
    
    def pick_action(self, logits):
        result = None
        #pdb.set_trace()
        if random.random() > self.epsilon:
            result = torch.argmax(logits, dim=1).item()
        else:
            result = random.randint(0, logits.shape[-1] - 1)
        self.epsilon = max(self.epsilon * self.decay, self.min_eps)
        return result

    def __call__(self, logits):
        return self.pick_action(logits)

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

class DQN(nn.Module):

    def __init__(self, strategy, model):
        super(DQN, self).__init__()

        # self.conv_model = nn.Sequential([conv(inputs_shape[0], conv1_out, kernel_size=3, padding=1, stride=1, bias=True),
        #                                  pooling(2, stride=2),
        #                                  conv_activation(),
        #                                  conv(conv1_out, conv2_out, kernel_size=3, padding=1, stride=1, bias=True),
        #                                  pooling(2, stride=2),
        #                                  conv_activation()])

        # def conv_out(shape):
        #     out = self.conv_model(torch.zeros(1, *shape))
        #     return int(np.prod(out.size()))

        # self.fc_model = make_ann([conv_out(input_shape)] + fc_layres, fc_activation)
        # fc_dims[0] *= state_size
        self.model = model
        self.strategy = strategy

    def forward(self, obs):
        # conv_out = self.conv_model(obs)
        return self.model(obs)

    def sample_action(self, obs):
        with torch.no_grad():
            out = self.model(obs)
            return self.strategy.pick_action(out)


class ReplayBuffer:

    def __init__(self, size, single_state_size, batch_size, state_shapes, state_dtype=torch.float32,
                 action_dtype=torch.float32):

        self.max_size = size
        self.single_state_size = single_state_size
        self.batch_size = batch_size
        self.shape = state_shapes

        self.curr = 0
        self.size = 0

        self.actions = torch.empty(self.max_size, dtype=action_dtype)
        self.rewards = torch.empty(self.max_size, dtype=torch.float32)
        self.observations = torch.empty(self.max_size, *state_shapes, dtype=state_dtype)
        #self.next_obs = torch.empty(self.size, *state_shapes, dtype=torch.float32)
        self.done_flags = torch.empty(self.max_size, dtype=torch.bool)
        self.adv = torch.empty(self.max_size, dtype=torch.float32)

        self.index_buf = torch.empty(self.batch_size, dtype=torch.long)
        self.state_buf = torch.empty(self.batch_size, self.single_state_size, *state_shapes, dtype=state_dtype)
        self.next_state_buf = torch.empty_like(self.state_buf)

    def add(self, obs, action, reward, done):
        
        if obs.shape != self.observations.shape[1:]:
            raise ValueError('Given observation with invalid shape')
        #pdb.set_trace()
        self.actions[self.curr] = action
        self.observations[self.curr, ...] = obs
        self.rewards[self.curr] = reward
        self.done_flags[self.curr] = done
        #self.next_state_buf[self.curr] = next_state
        
        self.size = max(self.size, self.curr + 1)
        self.curr = (self.curr + 1) % self.max_size
    
    def has_batch(self):
        return self.size >= self.batch_size

    def get_batch(self):
        
        def fill_indices(index_buffer, done):
            #pdb.set_trace()
            for i in range(self.batch_size):

                index = random.randint(self.single_state_size, self.size - 1)
                while done[index - self.single_state_size : index].any() or \
                      (self.curr < index and self.curr + self.single_state_size >= index):
                    
                    index = random.randint(self.single_state_size, self.size - 1)
                
                index_buffer[i] = index

        def obs_at(index):
            if index < self.single_state_size:
                raise ValueError('Invalid index')
            return self.observations[index - self.single_state_size : index]

        if self.size < self.single_state_size or self.size < self.batch_size: 
            # Nema dovoljno u bufferu
            raise ValueError('Not enough in buffer')

        fill_indices(self.index_buf, self.done_flags)
        #pdb.set_trace()
        for i, index in enumerate(self.index_buf):
            self.state_buf[i] = obs_at(index)
            self.next_state_buf[i] = obs_at(index + 1)
        #pdb.set_trace()
        return self.state_buf, torch.index_select(self.actions, dim=0, index=self.index_buf), \
               torch.index_select(self.rewards, dim=0, index=self.index_buf), self.next_state_buf, \
               torch.index_select(self.done_flags, dim=0, index=self.index_buf)
        


def train(env, model, target_model, replay_memory, model_opt, target_update_step=100, gamma=0.99, epochs=3000):

    def compute_loss():
        #pdb.set_trace()
        states, actions, rewards, next_states, done = replay_memory.get_batch()
        q = model(states).gather(dim=-1, index=actions.unsqueeze(-1))
        q_next = torch.max(target_model(next_states), dim=-1, keepdim=True)[0]
        target = rewards.unsqueeze(-1) + torch.logical_not(done).unsqueeze(-1) * gamma * q_next

        return F.mse_loss(q, target)

    def episode(total_steps):

        obs, done = env.reset(), False
        state = torch.stack([obs.view(-1)] * 4)
        episode_length = 0
        total_reward = 0
        total_loss = 0
        while not done:

            if total_steps % target_update_step == 0:
                print(f'Updating target model, iteration: {total_steps}')
                target_model.load_state_dict(model.state_dict())

            action = model.sample_action(state.unsqueeze(0))
            #pdb.set_trace()
            obs, reward, done, _ = env.step(action)
            episode_length += 1

            next_state = torch.cat([state[1:], obs.unsqueeze(0)])

            replay_memory.add(obs, action, reward, done)

            total_reward += reward
            if replay_memory.has_batch():
                loss = compute_loss()

                model_opt.zero_grad()
                loss.backward()
                model_opt.step()
                total_loss += loss.item()
            state = next_state
            #total_steps += 1
        
        return total_reward, total_loss, episode_length


    for i in range(1, epochs):
        reward, loss, length = episode(i)
        print(f'Episode {i}: reward={reward}, loss={loss}, length={length}')
    
    return model, target_model

            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    #parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    #parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()

    env = gym.make('CartPole-v0')
    env = wrappers.TorchWrapper(env)    
    replay_memory = ReplayBuffer(512, 4, 32, env.observation_space.shape, action_dtype=torch.long)
    
    def make_model():
        return nn.Sequential(nn.Conv1d(4, 8, kernel_size=3, padding=1),
                             nn.Flatten(1, -1),
                             nn.Linear(8 * 4, 16),
                             nn.ReLU(),
                             nn.Linear(16, env.action_space.n))

    dqn = DQN(EpsilonGreedyStrategy(), make_model())

    target_dqn = DQN(EpsilonGreedyStrategy(), make_model())
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()
    
    optimizer = opt.SGD(dqn.parameters(), lr=args.lr)

    train(env, dqn, target_dqn, replay_memory, optimizer, 50)
    



