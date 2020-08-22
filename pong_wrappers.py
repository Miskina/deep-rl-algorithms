import numpy as np
import gym
import torch

class TorchImageDimsWrapper(gym.ObservationWrapper):

  def __init__(self, env=None):
    super(TorchImageWrapper, self).__init__(env)
    s = self.observation_space.shape
    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(s[-1], s[0], s[1]))

  def observation(self, obs):
    obs = np.moveaxis(obs, 2, 0)
  
class BufferedWrapper(gym.ObservationWrapper):

    def __init__(self, env, steps, dtype=np.float32):
        super(BUfferedWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(steps, axis=0),
                                                old_space.high.repeat(steps, axis=0),
                                                dtype=dtype)
    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())
    
    def observation(self, obs):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = obs
        return self.buffer

class TorchWrapper(gym.Wrapper):
    
    def __init__(self, env=None, obs_dtype=torch.float32, reward_dtype=torch.float32):
        super(TorchWrapper, self).__init__(env)
        self.obs_dtype = obs_dtype
        self.reward_dtype = reward_dtype
    
    def reset(self):
        obs = self.env.reset()
        return torch.as_tensor(obs, dtype=self.obs_dtype)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return torch.as_tensor(obs, dtype=self.obs_dtype), torch.as_tensor(reward, dtype=self.reward_dtype), done, info


class PreprocessWrapper(gym.ObservationWrapper):

  def __init__(self, env):
    super(PreprocessWrapper, self).__init__(env)
    obs_space_old = env.observation_space
    old_shape = obs_space_old.shape
    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(int((old_shape[0] - 50) / 2), int((old_shape[1]) / 2)))
    
  
  def observation(self, obs):
    return preprocess_image(obs)

class SkipAndPoolWrapper(gym.Wrapper):

    def __init__(self, env, skip=4, pooling_function='max'):
        super(SkipAndPoolWrapper, self).__init__(env)
        if pooling_function == 'max':
            self.pool = np.max
        elif pooling_function == 'mean':
            self.pool = np.mean
        else:
            raise ValueError('Only accepting "max" and "mean" pooling')

        self.buff = np.zeros((skip, env.observation_space.shape))
    
    def step(self, act):
        total_reward = 0.0
        done = False
        processed_before_done = 0
        for i in range(self.buff.shape[0]):
            obs, rw, done, info = self.env.step(act)
            total_reward += rw
            self.buff[i] = obs
            processed_before_done += 1
            if done:
                break
        pooled_frame = self.pool(self.buff[:processed_before_done], axis=0)
        return pooled_frame, total_reward, done, info