# -*- coding: utf-8 -*-

import math
import random
import numpy as np
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

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

    def load(self, fn):
        self.load_state_dict(torch.load(fn))

class PlayerMLP:
    def __init__(self, casino):
        self.casino = casino
        self.nA = casino.get_action_space()
        self.nDS = casino.nDS
        self.nPS = casino.nPS
        self.nPA = casino.nPA
        self.nSnA = (self.nDS, self.nPS, self.nPA, self.nA)
        self.n_episode = 0
        self.pocket = 0
        self.make_net()

        self.optimizer = optim.Adam(self.qf.parameters(), lr=0.002, weight_decay=1e-4)
        milestone = [30000, 60000, 100000, 150000, 200000, 300000, 450000, 650000, 900000, 1200000, 1550000, 1950000, 2400000, 2900000, 3450000, 4050000, 4700000, 5400000, 6150000]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestone, gamma=1/np.sqrt(3))

        self.batch_size = 1
        self.n_idx = 4
        self.n = [0] * self.n_idx

        self.NV = np.ones([12,23,2,self.n_idx,4],dtype=np.int32)

    def make_net(self):
        H = 64
        lsize = [self.dimS(), H, H,H, self.nA]
        self.qf = Q_Net(lsize)

    def load(self, fn):
        self.qf.load(fn)

    def save(self, fn):
        self.qf.save(fn)

    def dimS(self):
        return 4

    def get_state(self):
        s = self.casino.observe()
        s = torch.tensor(s, dtype=torch.float32)
        p_idx = self.casino.peep_cpr()

        if p_idx < -0.1:
            p = 0
            self.n[0] += 1
        elif p_idx < 0.02:
            p = 1
            self.n[1] += 1
        elif p_idx < 0.15:
            p = 2
            self.n[2] += 1
        else:
            p = 3
            self.n[3] += 1
        p = torch.tensor([p], dtype=torch.float32)
        s = torch.cat((s,p))
        return s

    def get_action(self, state, greedy=True):
        if greedy:
            with torch.no_grad():
                q = self.qf(state)
            q = q.cpu()
            q = q.numpy()
            a = q.argmax()
        else:
            a = np.random.choice(self.nA)
        return a

    def UCB_action(self, state):
        with torch.no_grad():
            q = self.qf(state)
        q = q.cpu()
        tem = self.NV.flatten()
        t = sum(tem)

        state = state.to(dtype=torch.int32)
        state = state.tolist()
        ds, ps, pua, p = state
        ucb_term = np.sqrt((2*math.log(t))/self.NV[ds,ps,pua,p])

        q = q.numpy()
        q = q + ucb_term
        a = q.argmax()
        return a

    def TD_update_Q(self, s,a,r,sp):
        s = torch.tensor(s)
        q = self.qf(s)[a]
        if sp is None:
            t = torch.tensor(float(r))
        else:
            sp = torch.tensor(sp)
            with torch.no_grad():
                t = r + self.qf(sp).max()

        loss = F.mse_loss(q, t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def Batch_update(self,batch_size):
        sample = self.Trans_Memory.sample(batch_size)
        for i in range(len(sample)):
            s,a,r,sp = sample[i]
            dx,ps,pu,p,aa = np.hstack((s,a)).astype(int)
            self.NV[dx,ps,pu,p,aa] += 1
            if r > 1:
                r = r * (1+(1/math.sqrt(self.n_episode+1)))
            self.TD_update_Q(s,a,r,sp)

    def reset_episode(self):
        self.n_episode += 1

    def run_episode1(self, batch_size):
        self.lr_scheduler.step()
        self.reset_episode()
        self.casino.start_game()
        s = self.get_state()
        done = False
        while not done:
            a = self.UCB_action(s)
            done = True
            _, reward, done = self.casino.step(a)
            if done: sp = None
            else:    sp = self.get_state()
            self.Trans_Memory.push(s,a,reward,sp)
            s = sp
        if self.n_episode > 100:
            self.Batch_update(batch_size)

    def run_simulation(self, n_episode=1E7, max_time=6000000):
        stime = time.time()
        ip1 = 0
        self.Trans_Memory = ReplayMemory(1000)
        self.n_episode = 0

        while time.time() - stime < max_time:
            ip1 += 1
            if ip1 > n_episode: break
            self.run_episode1(self.batch_size)
            if ip1 % 1000 == 0:
                print (ip1, 'lr = %f'%(self.optimizer.param_groups[0]['lr']))
        print("learning complete")

    def get_all_state_tensor(self, p):
        S = torch.zeros((self.nDS * self.nPS * self.nPA, self.dimS()))
        k = 0
        for ds in range(self.nDS):
            for ps in range(self.nPS):
                for ua in range(self.nPA):
                    S[k][0] = ds
                    S[k][1] = ps
                    S[k][2] = ua
                    k += 1
        return S

    def plot_Q(self, p=None, fid=0):
        if p is None:
            p = np.zeros(10)
            p.fill(1 / 13)
            p[8] = 4 / 13
        S = self.get_all_state_tensor(p)
        with torch.no_grad():
            Q = self.qf(S)
        Q = Q.numpy()
        Q = Q.reshape(self.nSnA)
        pi = Q.argmax(-1)
        Q[0:2, :, :, :] = -2
        Q[:, 0:4, :, :] = -2
        Q[:, 22, :, :] = -2
        Q[:, 4:12, 1, :] = -2
        pi[0:2, :] = -2
        pi[:, 0:4, :] = -2
        pi[:, 22, :] = -2
        pi[:, 4:12, 1] = -2

        fig = plt.figure(fid, figsize=(7, 8), clear=True)
        for ua in range(self.nPA):
            for a in range(self.nA):
                self.plot_Qi(fig, Q, a, ua)
            self.plot_pi(fig, pi, ua)
        self.diff_Q(fig, Q, pi)
        plt.draw()
        plt.pause(1)

    def plot_Qi(self, fig, Q, a, ua):
        ax = fig.add_subplot(6, 2, 2 * a + ua + 1)
        ax.imshow(Q[:, :, ua, a], vmin=-2, vmax=1)

    def plot_pi(self, fig, pi, ua):
        ax = fig.add_subplot(6, 2, 9 + ua)
        ax.imshow(pi[:, :, ua], vmin=-2, vmax=self.nA - 1)

    def diff_Q(self, fig, Q, pi):
        if 'pi_old' in dir(self):
            PIdiff = (pi != self.pi_old)
            Qdiff = (Q - self.Q_old)
            print("PI diff = %d" % (PIdiff.sum()))
            print('Qdiff max=%.3f, min=%.3f' % (Qdiff.max(), Qdiff.min()))
            ax = fig.add_subplot(6, 2, 11)
            ax.imshow(PIdiff[:, :, 0])
            ax = fig.add_subplot(6, 2, 12)
            ax.imshow(PIdiff[:, :, 1])
        self.Q_old = Q
        self.pi_old = pi

    def update_pocket(self, reward):
        self.pocket += reward

    def play_game(self):
        self.reset_episode()
        self.casino.start_game()
        done = False
        while not done:
            s = self.get_state()
            a = self.get_action(s, greedy=True)
            _, reward, done = self.casino.step(a)
        self.update_pocket(reward)
        return reward

    def print_epg_wr(self, n_games):
        epg = self.pocket / n_games
        wr = (epg + 1) / 2
        std_wr = np.sqrt(wr * (1 - wr) / n_games)
        print("# of game=%d, player's pocket=%d, E/G=%.5f, WR=%.5f%% +- %.5f"
              % (n_games, self.pocket, epg, wr * 100, std_wr * 100))
        return wr

    def test_performance(self, n_games):
        self.pocket = 0
        n_10 = n_games / 10
        for i in range(1, n_games + 1):
            reward = self.play_game()
            if n_games > 100000 and i % n_10 == 0:
                self.print_epg_wr(i)
        print ("Final result")
        return self.print_epg_wr(n_games)
