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

def get_softmax(z, tau=1.0):
    exp_z = np.exp((z - np.max(z)) / tau)
    p = list(exp_z / exp_z.sum())
    p[-1] = 1 - sum(p[:-1])
    return p

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

        #self.optimizer1 = optim.Adam(self.qf1.parameters(), lr=0.001, weight_decay=0)
        #self.optimizer2 = optim.Adam(self.qf2.parameters(), lr=0.001, weight_decay=0)
        #self.optimizer1 = optim.Adadelta(self.qf1.parameters(), lr=0.1, weight_decay=0)
        #self.optimizer2 = optim.Adadelta(self.qf2.parameters(), lr=0.1, weight_decay=0)
        #self.optimizer1 = optim.Adagrad(self.qf1.parameters(), lr=0.01, weight_decay=0)
        #self.optimizer2 = optim.Adagrad(self.qf2.parameters(), lr=0.01, weight_decay=0)
        #self.optimizer1 = optim.SparseAdam(self.qf1.parameters(), lr=0.001)
        #self.optimizer2 = optim.SparseAdam(self.qf2.parameters(), lr=0.001)
        self.optimizer1 = optim.Adamax(self.qf1.parameters(), lr=0.002, weight_decay=0)
        self.optimizer2 = optim.Adamax(self.qf2.parameters(), lr=0.002, weight_decay=0)
        #self.optimizer1 = optim.ASGD(self.qf1.parameters(), lr=0.01, weight_decay=0)
        #self.optimizer2 = optim.ASGD(self.qf2.parameters(), lr=0.01, weight_decay=0)
        #self.optimizer1 = optim.LBFGS(self.qf1.parameters(), lr=1)
        #self.optimizer2 = optim.LBFGS(self.qf2.parameters(), lr=1)
        #self.optimizer1 = optim.RMSprop(self.qf1.parameters(), lr=0.01, weight_decay=0)
        #self.optimizer2 = optim.RMSprop(self.qf2.parameters(), lr=0.01, weight_decay=0)
        #self.optimizer1 = optim.Rprop(self.qf1.parameters(), lr=0.01)
        #self.optimizer2 = optim.Rprop(self.qf2.parameters(), lr=0.01)
        #self.optimizer1 = optim.SGD(self.qf1.parameters(), lr=0.01, nesterov = True, momentum = 0.1)
        #self.optimizer2 = optim.SGD(self.qf2.parameters(), lr=0.01, nesterov = True, momentum = 0.1)

        milestone = [5000, 10000,30000,50000,70000,90000,120000,160000,200000,250000,]

        self.lr_scheduler1 = optim.lr_scheduler.MultiStepLR(self.optimizer1, milestone, gamma=1/np.sqrt(10))
        self.lr_scheduler2 = optim.lr_scheduler.MultiStepLR(self.optimizer2, milestone, gamma=1/np.sqrt(10))
        #self.lr_scheduler1 = optim.lr_scheduler.LambdaLR(self.optimizer1, lr_lambda = lambda epoch: 1/math.sqrt((epoch+1)/1000))
        #self.lr_scheduler2 = optim.lr_scheduler.LambdaLR(self.optimizer2, lr_lambda = lambda epoch: 1/math.sqrt((epoch+1)/1000))
        #self.lr_scheduler1 = optim.lr_scheduler.ExponentialLR(self.optimizer1, gamma = 1+1e-5)
        #self.lr_scheduler2 = optim.lr_scheduler.ExponentialLR(self.optimizer2, gamma = 1+1e-5)
        #self.lr_scheduler1 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer1, T_max=1000)
        #self.lr_scheduler2 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer2, T_max=1000)

        self.batch_size = 1

        self.choice = 1

        self.test_episode = 0

    def make_net(self):
        H = 64
        lsize = [self.dimS(), H, H,H, self.nA]
        self.qf1 = Q_Net(lsize)
        self.qf2 = Q_Net(lsize)

    def load(self, fn):
        self.qf1.load(fn)

    def save(self, fn):
        self.qf1.save(fn)

    def dimS(self):
        return 13

    def get_state(self):
        s = self.casino.observe()
        s = torch.tensor(s, dtype=torch.float32)
        p = self.casino.peep()
        p = torch.tensor(p, dtype=torch.float32)
        s = torch.cat((s,p))
        return s

    def get_action(self, state, greedy=True):
        if greedy:
            with torch.no_grad():
                q = (self.qf1(state)+ self.qf2(state))/2
            q = q.cpu()
            q = q.numpy()
            a = q.argmax()
        else:
            a = np.random.choice(self.nA)
        return a

    def get_eps_greedy_action(self, state):
        eps = 1/math.sqrt(self.n_episode/10000)
        eps = min(eps, 0.5)
        p = np.random.random()
        if p > eps:
            with torch.no_grad():
                q = self.qf(state)
                q = q.cpu()
                q = q.numpy()
                a = q.argmax()
        else:
            a = np.random.choice(self.nA)
        return a

    def get_DQ_eps_action(self, state):
        eps = 1/math.sqrt(self.n_episode/10000)
        eps = min(eps, 0.5)
        p = np.random.random()
        if p > eps:
            with torch.no_grad():
                q1 = self.qf1(state)
                q2 = self.qf2(state)
                q = (q1+q2)/2
                q = q.cpu()
                q = q.numpy()
                a = q.argmax()
        else:
                a = np.random.choice(self.nA)
        return a

    def get_softmax_action(self, state,tau):
        with torch.no_grad():
            q = self.qf(state)
        q = q.cpu()
        q = q.numpy()
        soft_q = get_softmax(q,tau)
        a = np.random.choice(4,1,p=soft_q)
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
        #loss = F.mse_loss(q, t)
        loss = F.smooth_l1_loss(q, t)
        #loss = F.soft_margin_loss(q,t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def TD_update_DQ(self, s,a,r,sp):
        s = torch.tensor(s)

        if sp is None:
            t = torch.tensor(float(r))
        else:
            sp = torch.tensor(sp)
            with torch.no_grad():
                if self.choice == 1:
                    t = r + self.qf2(sp).max()
                else:
                    t = r + self.qf1(sp).max()
        if self.choice == 1:
            q = self.qf1(s)[a]

            #loss = F.l1_loss(q,t)
            loss = F.mse_loss(q, t)
            #loss = F.smooth_l1_loss(q, t)
            #loss = F.kl_div(q,t)
            #loss = F.multilabel_soft_margin_loss(q,t)

            #loss = F.poisson_nll_loss(q,t)
            #loss = F.cross_entropy(q,t)
            #loss = F.hinge_embedding_loss(q,t)
            #loss = F.multilabel_margin_loss(q,t)
            #loss = F.multi_margin_loss(q,t)
            #loss = F.nll_loss(q,t)
            #loss = F.soft_margin_loss(q,t)

            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()
        else:
            q = self.qf2(s)[a]

            #loss = F.l1_loss(q,t)
            loss = F.mse_loss(q, t)
            #loss = F.smooth_l1_loss(q, t)
            #loss = F.kl_div(q,t)
            #loss = F.multilabel_soft_margin_loss(q,t)

            #loss = F.poisson_nll_loss(q,t)
            #loss = F.cross_entropy(q,t)
            #loss = F.hinge_embedding_loss(q,t)
            #loss = F.multilabel_margin_loss(q,t)
            #loss = F.multi_margin_loss(q,t)
            #loss = F.nll_loss(q,t)
            #loss = F.soft_margin_loss(q,t)

            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step()


    def Batch_update(self,batch_size):
        sample = self.Trans_Memory.sample(batch_size)
        for i in range(len(sample)):
            s,a,r,sp = sample[i]
            self.TD_update_DQ(s,a,r,sp)

    def reset_episode(self):
        self.n_episode += 1

    def run_episode1(self, batch_size):
        p = random.random()
        if p < 0.5:
            self.lr_scheduler1.step()
            self.choice = 1
        else:
            self.lr_scheduler2.step()
            self.choice = 2
        self.n_episode += 1
        self.casino.start_game()
        s = self.get_state()
        done = False
        while not done:
            #tau = math.sqrt(1/self.n_episode)
            #a = self.get_softmax_action(s,tau)
            #a = self.get_eps_greedy_action(s)
            a = self.get_DQ_eps_action(s)
            _, reward, done = self.casino.step(a)
            if done: sp = None
            else:    sp = self.get_state()
            self.Trans_Memory.push(s,a,reward,sp)
            s = sp
        if len(self.Trans_Memory.memory) > batch_size*100:
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

            if ip1 % 10000 == 0:
                print (ip1, 'lr = %f'%((self.optimizer1.param_groups[0]['lr']+self.optimizer2.param_groups[0]['lr'])/2))
                w = self.test_performance(30000) * 100
                print(ip1, 'winning rate = ', w)
                print('--------------------------------------')
                #self.plot_Q()

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
        self.test_episode += 1
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
