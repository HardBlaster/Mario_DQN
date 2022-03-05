import gym
import torch
import numpy as np
from torchvision import transforms


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip
        
    def step(self, action):
        total_reward = 0.
        done = False
        
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            if done:
                return obs, total_reward, done, info
            
        return obs, total_reward, done, info


class GrayScaleObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        self.gs_transform = transforms.Grayscale()
        
        obs_shape = self.observation_space.shape[:2] #without channels
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        
    def observation(self, observation):
        obs = torch.tensor(observation.copy(), dtype=torch.float32).permute(2, 0, 1) #np array -> tensor, then HxWxC -> CxHxW
        
        return self.gs_transform(obs)


class ResizeObs(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        
        self.shape = (shape, shape) if isinstance(shape, int) else shape
        self.transforms = transforms.Compose([
            transforms.Resize(self.shape), 
            transforms.Normalize(0, 255)
        ])
        
        obs_shape = self.shape + self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        
    def observation(self, observation):
        return self.transforms(observation).squeeze(0) #resize and normalize, remove channel dimension 1xHxW -> HxW
