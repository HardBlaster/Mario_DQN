import os
import torch
import random
import numpy as np

from collections import deque


class Mario:
    def __init__(self, model, action_dim, save_dir, cuda=True, memory_size=int(1e5), batch_size=32, lr=.00025, exploration_rate=1, er_dec=.001, er_min=.1, gamma=.9, random_steps=1e4, learn_freq=3, sync_freq=1e4, save_freq=5e5):
        self.cuda = cuda
        self.model = model.cuda() if cuda else model
        self.action_dim = action_dim
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=memory_size)
        
        self.exploration_rate = exploration_rate
        self.exploration_rate_decrease = er_dec
        self.exploration_rate_min = er_min
        self.gamma = gamma
        self.random_steps = random_steps
        self.learn_freq = learn_freq
        self.sync_freq = sync_freq
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.current_step = 0
        self.save_freq = save_freq
        
    def learn(self):
        if self.current_step % self.sync_freq == 0:
            self.sync_Q()

        if self.current_step % self.save_freq == 0:
            self.save()

        if self.current_step < self.random_steps:
            return None, None

        if self.current_step % self.learn_freq != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()
        if self.cuda:
            state, next_state, action, reward, done = state.cuda(), next_state.cuda(), action.cuda(), reward.cuda(), done.cuda()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q(td_est, td_tgt)

        return td_est.mean().item(), loss
        
    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
            
        else:
            state = torch.tensor(state.__array__()).unsqueeze(0)
            state = state.cuda() if self.cuda else state
            action = torch.argmax(self.model(state)).item()
            
        self.exploration_rate = max(self.exploration_rate - self.exploration_rate_decrease, self.exploration_rate_min)
        self.current_step += 1
        
        return action
    
    def cache(self, state, next_state, action, reward, done):
        self.memory.append((
            torch.tensor(state.__array__()).cpu(),
            torch.tensor(next_state.__array__()).cpu(),
            torch.tensor(action).cpu(),
            torch.tensor(reward).cpu(),
            torch.tensor(done).cpu(),
        ) if self.cuda else (
            torch.tensor(state.__array__()),
            torch.tensor(next_state.__array__()),
            torch.tensor(action),
            torch.tensor(reward),
            torch.tensor(done),
        ))
        
    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        
        return map(torch.stack, zip(*batch))
    
    def td_estimate(self, state, action):
        return self.model(state)[np.arange(0, self.batch_size), action]
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        best_next_action = torch.argmax(self.model(next_state), axis=1)
        next_Q = self.model(next_state, model="target")[np.arange(0, self.batch_size), best_next_action]
        
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def sync_Q(self):
        self.model.target.load_state_dict(self.model.online.state_dict())

    def save(self):
        save_path = os.path.join(self.save_dir, f"mario_net_{self.current_step}.chkpt")

        torch.save(
            dict(model=self.model.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.current_step}")
        