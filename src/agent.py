
import torch
from torch import nn
import random
import os
import numpy as np
from collections import deque
from network import DQN


class DQNAgent:
    def __init__(self, state_dim, action_dim, save_dir):
        # initial conditions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.use_cuda = torch.cuda.is_available()

        # load models
        self.net = DQN(self.state_dim,self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
        
        # setting exploartion rate
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.9995
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        
        # setting saving rate
        self.save_every = 100

        # cache and recall information
        self.memory = deque(maxlen=10000)
        self.batch_size = 64

        # TD settings
        self.gamma = 0.9

        # NN setting
        self.lr = 0.0001
        self. optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.SmoothL1Loss()

        # learning setting
        self.burnin = 64 #steps before training
        self.learn_every = 3 

        
    def act(self,state):
        # Exploration
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        
        # Exploitation
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state)
            action_idx = torch.argmax(action_values,axis=1).item()
        
        # decrease exploration rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # incrment step size
        self.curr_step += 1

        return action_idx

    @torch.no_grad()
    def act_no_grad(self, state):
        # Exploration
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        
        # Exploitation
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state)
            action_idx = torch.argmax(action_values,axis=1).item()
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            if self.use_cuda:
                state = torch.tensor(state)
                next_state = torch.tensor(next_state)
                action = torch.tensor([action])
                reward = torch.tensor([reward])
                done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def td_estimate(self, state, action):
        current_Q = self.net(state)[np.arange(0,self.batch_size), action] 
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state)
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state)[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self):
        save_path = (self.save_dir / f"dqn_{int(self.curr_step // self.save_every)}.chkpt")
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), save_path)
        print(f"DQN saved to {save_path} at step {self.curr_step}")
    
    def load(self, load_path):
        if os.path.isfile(load_path):
            checkpoint = torch.load(load_path)
            self.exploration_rate = checkpoint['exploration_rate']
            self.net.load_state_dict(checkpoint['model'])
            print("=> loaded checkpoint '{} ".format(load_path))
        else:
            print("=> no checkpoint found at '{}'".format(load_path))
        
    def learn(self):
        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step < self.learn_every != 0:
            return None, None
        
        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)

        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q(td_est, td_tgt)

        return (td_est.mean().item(), loss)


