import gym
import torch
import numpy as np

import torchvision.transforms as transforms

class FireToStartWrapper(gym.Wrapper):

    def __init__(self, env=None):
        super(FireToStartWrapper, self).__init__(env)

class SkipAndPoolWrapper(gym.Wrapper):

    def __init__(self, input_shape, env=None, skip=4, pooling_function='max'):
        super(SkipAndPoolWrapper, self).__init__(env)
        if pooling_function == 'max':
            self.pool = torch.max
        elif pooling_function == 'mean':
            self.pool = torch.mean
        else:
            raise ValueError('Only accepting "max" and "mean" pooling')

        self.buff = torch.zeros(skip, *input_shape)
    
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
        pooled_frame = self.pool(self.buff[:processed_before_done], dim=0)
        return pooled_frame, total_reward, done, info


class ImageToTorchDims(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(ImageToTorch, self).__init__(env)
        shape_old = self.observation_space.shape

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(shape_old[-1], shape_old[0], shape_old[1]))

    def observation(self, obs):
        return np.moveaxis(obs, 2, 0)

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

class TorchCropAndGrayscale(TorchWrapper):

    def __init__(self, env=None, reward_dtype=torch.float32, resize_shape=[84, 84]):
        super(TorchCropAndGrayscale, self).__init__(env, obs_dtype=torch.uint8, reward_dtype=reward_dtype)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, *resize_shape), dtype=np.uint8)
        self.image_processing = transforms.Compose([transforms.Resize(resize_shape),
                                                    transforms.Grayscale()])

    def reset(self):
        obs_tensor = super(TorchCropAndGrayscale, self).reset()
        return self.image_processing(obs_tensor)
    
    def step(self, action):
        obs, r, d, i = super(TorchCropAndGrayscale, self).step(action)
        return self.image_processing(obs), r, d, i

    
