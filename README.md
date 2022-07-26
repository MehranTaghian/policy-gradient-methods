# Policy Gradient Methods
This repository contains the policy gradient algorithms from bandit policy gradient to 
PPO and REINFORCE. Each algorithm is explained in the following section.

### Gradient Bandit
The policy chooses actions from a normal distribution ($A \sim \mathcal{N}(\mu, \sigma^2) $) as follows:
$$\pi_\theta(A) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{1}{2\sigma^2}(A - \mu)^2\right)$$
where the policy is parametrized by $\theta = [\mu, \sigma^2]^\top$ and $\mu \in \mathbb{R}$ and 
$\sigma^2 \in \mathbb{R}^+$. 

The environment provides a reward sampled from another normal distribution 
$R\sim\mathcal{N}\left(-(A - a^*)^2, 1\right)$ where $a^*$ is the value of the optimal arm.
