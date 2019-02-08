import gym

import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayMemory:
    def __init__(self,capacity=1000):
        self.position = -1
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(args)
            self.position = (self.position + 1) % self.capacity
        else:
            self.memory[self.position] = args
            self.position = (self.position + 1) % self.capacity

    def sample(self,sample_size):
        return random.sample(self.memory, sample_size)

class Q_Net(torch.nn.Module):
    def __init__(self, lsize):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = len(lsize) - 1
        for i in range(self.n_layers):
            self.layers.append(torch.nn.Linear(lsize[i], lsize[i+1]))

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)

            if i < self.n_layers-1:
                x = F.relu(x)
        return x

    def save(self, fn):
        torch.save(self.state_dict(), fn)

class Agent:
    def __init__(self,env):
        self.env = env
        self.qf1, self.qf2 = self.make_network()
        self.RM = ReplayMemory()
        self.n_episode = 0
        self.total_step = 0
        self.u_check = 0
        self.train_reward = [0]
        self.train_step = [0]

        self.gamma = 0.9

        self.optimizer1 = optim.Adam(self.qf1.parameters(), lr=0.001, weight_decay=1e-4)
        self.optimizer2 = optim.Adam(self.qf2.parameters(), lr=0.001, weight_decay=1e-4)

        milestone = [20000, 100000, 280000, 450000, 800000, 1300000, 2000000, 3500000, 5000000]
        self.lr_scheduler1 = optim.lr_scheduler.MultiStepLR(self.optimizer1, milestone, gamma=1/np.sqrt(3))
        self.lr_scheduler2 = optim.lr_scheduler.MultiStepLR(self.optimizer2, milestone, gamma=1/np.sqrt(3))
        #self.lr_scheduler1 = optim.lr_scheduler.LambdaLR(self.optimizer1, lr_lambda = lambda epoch: 1/math.sqrt((epoch+1)/2000))
        #self.lr_scheduler2 = optim.lr_scheduler.LambdaLR(self.optimizer2, lr_lambda = lambda epoch: 1/math.sqrt((epoch+1)/2000))

    def make_network(self):
        lsize = [self.dimS(), 8,16,16, self.dimA()]
        print(lsize)
        return Q_Net(lsize), Q_Net(lsize)

    def dimS(self):
        state_space = self.env.observation_space
        return len(state_space.high)

    def dimA(self):
        action_space = self.env.action_space
        return action_space.n

    def train(self, fname):
        reward_episode = 0
        while True:
            if self.train_reward[-1] > 195 or self.n_episode == 100000:
                print(self.train_reward)
                print(self.train_step)
                break

            self.n_episode += 1
            step = 0
            state = self.env.reset()
            done = False
            while not done:
                self.total_step += 1
                self.env.render()
                action = self.DDQN_action_choice(state)
                action = np.array(action)
                next_s, reward, done, info = self.env.step(action)
                reward_episode += reward
                if done == True:
                    reward = -1
                self.RM.push(state,action,reward,next_s)
                state = next_s

                if self.total_step % 100 == 0 and self.total_step >= 1000 :
                    self.Batch_Update(100)

            if self.n_episode % 20 == 0:
                self.train_reward.append(reward_episode/20)
                self.train_step.append(self.total_step)
                reward_episode = 0

        self.save(fname)
        self.train_result()

    def DDQN_action_choice(self, state):
        state = torch.tensor(state,dtype=torch.float32)
        with torch.no_grad():
            q1 = self.qf1(state)
            q2 = self.qf2(state)
            q = (q1+q2)/2
        q = q.cpu()

        ### eps greedy action ###
        if self.u_check == 0:
            action = self.env.action_space.sample()
            return action
        else:
            p = random.random()
            eps = 1 / math.sqrt(self.u_check/2000)
            eps = min(0.5, eps)
            if p > eps:
                action = q.argmax()
            else:
                action = self.env.action_space.sample()
            return action

    def Batch_Update(self,batch_size):
        sample = self.RM.sample(batch_size)
        for i in range(len(sample)):
            s,a,r,ns = sample[i]
            self.TD_update_DQ(s,a,r,ns)
            self.u_check += 1
            if self.u_check % 100 == 0:
                print('{} updates'.format(self.u_check))
                print ('lr = %f'%((self.optimizer1.param_groups[0]['lr']+self.optimizer2.param_groups[0]['lr'])/2))
                print('Recent 3 rewards avg : ',str(sum(self.train_reward[-1:-4:-1])/3))
                print('# of Episodes',str(self.n_episode))

    def TD_update_DQ(self, s,a,r,ns):
        s = torch.tensor(s,dtype=torch.float32)
        ns = torch.tensor(ns,dtype=torch.float32)

        random_q = random.random()

        with torch.no_grad():
            if random_q < 0.5:
                t = r + self.gamma * self.qf2(ns).max()
            else:
                t = r + self.gamma * self.qf1(ns).max()

        if random_q < 0.5:
            self.lr_scheduler1.step()
            q = self.qf1(s)[a]
            #loss = F.mse_loss(q, t)
            loss = F.smooth_l1_loss(q, t)
            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()
        else:
            self.lr_scheduler2.step()
            q = self.qf2(s)[a]
            #loss = F.mse_loss(q, t)
            loss = F.smooth_l1_loss(q, t)
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step()

    def train_result(self):
        plt.plot(self.train_step,self.train_reward)
        plt.title('Double DQN')
        plt.ylabel('Reward per episode')
        plt.xlabel('# of Steps')
        plt.show()

    def save(self, fname):
        self.qf1.save(fname)

if __name__ == '__main__':
    fname = 'DDQN_result_function'

    env = gym.make('CartPole-v0')
    agent = Agent(env)
    agent.train(fname)
