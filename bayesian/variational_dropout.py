import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data


# config
c = {}
c['train_size'] = 300
c['batch_size'] = c['train_size'] // 30
c['lr'] = 1e-3

def func(x, noise):
    epsilon = np.random.normal(0, noise, len(x))
    y = x + 0.3*np.sin(2*np.pi*(x + epsilon)) + 0.3*np.sin(4*np.pi*(x + epsilon)) + epsilon
    return y.astype(np.float32)

# data
x_space = np.arange(-0.2, 1.2, 0.01).astype(np.float32)
true_y = func(x_space, noise=0)

x_train_space = x_space[(0<=x_space) & (x_space<=0.5)]
train_x = np.random.choice(x_train_space, c['train_size'], replace=True)
train_y = func(train_x, noise=0.02)


class Data(torch_data.Dataset):
    def __len__(self):
        return len(train_x)

    def __getitem__(self, index):
        return train_x[index], train_y[index]
data = Data()
data_loader = torch_data.DataLoader(data, batch_size=c['batch_size'], shuffle=True)


# model
class VariationalDropout(nn.Module):
    c1 = 1.16145124
    c2 = -1.50204118
    c3 = 0.58629921

    def __init__(self, dim):
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.log_alpha = nn.Parameter(-torch.rand(dim)) # alpha : [0, 1] => log alpha : [-inf, 0]

    def alpha(self):
        self.log_alpha.data = torch.clamp(self.log_alpha.data, max=0)
        return self.log_alpha.exp()

    def add_noise(self, x, alpha):
        epsilon = torch.randn_like(x)
        return x*(1 + alpha*epsilon)

    def forward(self, x):
        alpha = self.alpha()

        return self.add_noise(x, alpha)

    def forward_with_loss(self, x):
        alpha = self.alpha()
        kl_loss = -(0.5 * self.log_alpha + self.c1 * alpha + self.c2 * alpha ** 2 + self.c3 * alpha ** 3).sum()

        return self.add_noise(x, alpha), kl_loss


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        unit = 32
        self.network = nn.Sequential(
            nn.Linear(1, unit), VariationalDropout(unit), nn.ReLU(),
            nn.Linear(unit, unit), VariationalDropout(unit), nn.ReLU(),
            nn.Linear(unit, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=c['lr'])

    def forward(self, x, n_sampling=50):
        x_tile = x.repeat(n_sampling)
        out = self.network(x_tile[:, None])
        out = out.view(n_sampling, len(x))

        return out.mean(dim=0), out.std(dim=0)

    def forward_with_loss(self, x, y, kl_ratio=0.001):
        loss = 0
        out = x[:, None]
        for module in self.network:
            if isinstance(module, VariationalDropout):
                out, kl_loss = module.forward_with_loss(out)
                loss += kl_loss
            else:
                out = module(out)
        out = out[:, 0]

        loss = kl_ratio*loss + F.mse_loss(out, y)

        return out, loss

    def optimize(self, x, y):
        self.optimizer.zero_grad()
        _, loss = self.forward_with_loss(x, y)
        loss.backward()
        self.optimizer.step()

        return loss


# train
model = Model()
for epoch in range(100):
    for x, y in data_loader:
        loss = model.optimize(x, y)
    print("Epoch {:4d} | loss : {:f}".format(epoch, loss))


# feed
mu, sigma = model(torch.tensor(x_space))
mu = mu.detach().numpy()
sigma = sigma.detach().numpy()

# visualize
plt.scatter(train_x, train_y, marker='x', c='black', alpha=0.5, label='train data')
plt.plot(x_space, mu, label='prediction mean')
plt.fill_between(x_space, mu + sigma, mu - sigma, alpha=0.5, label='+-')
plt.plot(x_space, true_y, label='true function')
plt.legend()
plt.ylim(-0.5, 1.3)
plt.show()
