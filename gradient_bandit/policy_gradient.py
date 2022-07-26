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
mu_np = np.random.randn(1)
log_sigma_np = np.random.randn(1)

# True gradient calculation

mu = tor.tensor(mu_np, requires_grad=True)
log_sigma = tor.tensor(log_sigma_np, requires_grad=True)

# Taking the gradient with respect to the true objective function calculated in Question 1
true_obj = -(mu - astar) ** 2 - tor.exp(log_sigma)
true_obj = - true_obj
true_obj.backward(retain_graph=True)
true_grad_mean = mu.grad.data.clone()
true_grad_sigma = log_sigma.grad.data

# Calculating sample gradients using stochastic gradient descent
mu = tor.tensor(mu_np, requires_grad=True)
log_sigma = tor.tensor(log_sigma_np, requires_grad=True)
opt = tor.optim.SGD([mu, log_sigma], lr=0.0005)

# Experiment
T = 10000
mus = tor.zeros(T, act_dim)
sigmas = tor.zeros(T, act_dim)
grad_mus = tor.zeros(T, act_dim)
grad_sigmas = tor.zeros(T, act_dim)

pol = tor.distributions.MultivariateNormal(mu, tor.diag(tor.exp(log_sigma)))

for t in range(T):
    # Interaction
    A = pol.sample()
    R = -tor.norm(A - astar) ** 2 + tor.randn(1)

    # Compute loss
    sur_obj = pol.log_prob(A) * R
    loss = -sur_obj

    # Update
    opt.zero_grad()
    loss.backward(retain_graph=True)

    # Log
    grad_mus[t] = mu.grad.data.clone()
    grad_sigmas[t] = log_sigma.grad.data.clone()

# Optimizing using the objective function to find the solution

for t in range(T):
    # Interaction
    pol = tor.distributions.MultivariateNormal(mu, tor.diag(tor.exp(log_sigma)))
    A = pol.sample()
    R = -tor.norm(A - astar) ** 2 + tor.randn(1)

    # Compute loss
    sur_obj = pol.log_prob(A) * R
    loss = -sur_obj

    # Update
    opt.zero_grad()
    loss.backward(retain_graph=True)
    opt.step()

    # Log
    mus[t] = mu.data.clone()
    sigmas[t] = tor.exp(log_sigma.data.clone())

grad_mean_mus = np.cumsum(grad_mus.numpy()) / np.arange(1, T + 1, 1)
grad_mean_sigmas = np.cumsum(grad_sigmas.numpy()) / np.arange(1, T + 1, 1)

plt.figure()
plt.plot(mus)
plt.title('Leaned $\mu$ over 10000 trials')
plt.xlabel('Trial')
plt.ylabel('$\mu$')

plt.figure()
plt.plot(sigmas)
plt.title('Leaned $\sigma$ over 10000 trials')
plt.xlabel('Trial')
plt.ylabel('$\sigma$')

plt.figure()
plt.plot([0, T], [true_grad_mean] * 2, linestyle="--", alpha=0.5, linewidth=3, label='True derivative')
plt.plot(grad_mean_mus, label='Estimated derivative')
plt.title('Sample avg estimate of the derivative and the true derivative')
plt.xlabel('Trial')
plt.ylabel('$\\frac{\partial J(\\theta)}{\partial \mu}$')
plt.legend()

plt.figure()
plt.plot([0, T], [true_grad_sigma] * 2, linestyle="--", alpha=0.5, linewidth=3, label='True derivative')
plt.plot(grad_mean_sigmas, label='Estimated derivative')
plt.title('Sample avg estimate of the derivative and the true derivative')
plt.xlabel('Trial')
plt.ylabel('$\\frac{\partial J(\\theta)}{\partial \log(\sigma^2)}$')
plt.legend()

plt.show()
