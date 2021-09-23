import torch
import torch as tor
import matplotlib.pyplot as plt
import numpy as np

seed = 0

torch.manual_seed(seed)
np.random.seed(seed)

# Problem
act_dim = 1
astar = tor.tensor([1])

# Solution
mu = tor.randn(act_dim, requires_grad=True)
sigma = tor.randn(act_dim, requires_grad=True)
opt = tor.optim.SGD([mu, sigma], lr=0.01)

dist = tor.distributions.Normal(0, 1)
zeta = dist.sample()
pol = mu + sigma * zeta

pol.backward(retain_graph=True)
true_grad_mean = mu.grad.data
mu.grad.data.zero_()
true_grad_sigma = sigma.grad.data
sigma.grad.data.zero_()

# dist = tor.distributions.MultivariateNormal(mu, tor.diag(tor.exp(sigma)))
#
# dist.mean.backward(retain_graph=True)
# true_grad_mean = mu.grad.data
# mu.grad.data.zero_()
# dist.variance.backward(retain_graph=True)
# true_grad_sigma = sigma.grad.data
# sigma.grad.data.zero_()

# Experiment
T = 10000
mus = tor.zeros(T, act_dim)
sigmas = tor.zeros(T, act_dim)
grad_mus = tor.zeros(T, act_dim)
grad_sigmas = tor.zeros(T, act_dim)

for t in range(T):
    # Interaction
    # pol = tor.distributions.MultivariateNormal(mu, tor.diag(tor.exp(log_sigma)))
    # A = pol.rsample()
    # R = -tor.norm(A - astar) ** 2 + tor.randn(1)
    #
    # # Compute loss
    # sur_obj = pol.log_prob(A) * R
    # loss = -sur_obj

    dist = tor.distributions.Normal(0, 1)
    zeta = dist.sample()
    A = mu + sigma * zeta
    R = -tor.norm(A - astar) ** 2 + tor.randn(1)

    # Compute loss
    sur_obj = tor.exp(dist.log_prob(zeta)) * R
    loss = -sur_obj
    # loss = -R

    # Update
    opt.zero_grad()
    loss.backward(retain_graph=True)
    opt.step()

    # Log
    mus[t] = mu.data.clone()
    sigmas[t] = sigma.data.clone()

    grad_mus[t] = mu.grad.data.clone()
    grad_sigmas[t] = sigma.grad.data.clone()

grad_mean_mus = np.cumsum(grad_mus.numpy()) / np.arange(1, T + 1, 1)
grad_mean_sigmas = np.cumsum(grad_sigmas.numpy()) / np.arange(1, T + 1, 1)

plt.plot(mus)
plt.figure()
plt.plot(sigmas)
plt.figure()
plt.plot([0, T], [true_grad_mean] * 2, linestyle="--", alpha=0.5, linewidth=3)
plt.plot(grad_mean_mus)
plt.figure()
plt.plot([0, T], [true_grad_sigma] * 2, linestyle="--", alpha=0.5, linewidth=3)
plt.plot(grad_mean_sigmas)
plt.show()
