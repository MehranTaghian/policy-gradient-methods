import sys
import numpy as np
# import matplotlib as mpl
# mpl.use("TKAgg")
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists('results'):
    os.makedirs('results')

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


class Reinforce:
    def __init__(self, state_dim, action_dim, gamma, lr):
        self.policy = Policy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()
        self.gamma = gamma

    def update(self, memory):
        policy_loss = []
        transitions_length = 0
        for rewards, log_probs in zip(memory['batch_rewards'], memory['batch_log_probs']):
            R = 0
            returns = []
            transitions_length += len(rewards)
            for r in rewards[::-1]:
                R = r + self.gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = (-1 / transitions_length) * torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        prob = self.policy(state)
        m = Categorical(prob)
        action = m.sample()
        return action.item(), m.log_prob(action)


def main():
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])

    # Task setup block starts
    # Do not change
    env = gym.make('CartPole-v1')
    env.seed(seed)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    # Task setup block end

    # Learner setup block
    torch.manual_seed(seed)
    ####### Start
    gamma = 0.99
    batch_size = 5
    batch_size_counter = 0
    lr = 1e-3
    memory = {'batch_rewards': [], 'batch_log_probs': []}
    episode_rewards = []
    episode_log_probs = []
    agent = Reinforce(state_dim=o_dim, action_dim=a_dim, gamma=gamma, lr=lr)
    ####### End

    # Experiment block starts
    ret = 0
    rets = []
    avgrets = []
    o = env.reset()
    num_steps = 500000
    checkpoint = 10000
    for steps in range(num_steps):
        # Select an action
        ####### Start
        # Replace the following statement with your own code for
        # selecting an action
        a, log_prob = agent.select_action(o)
        ####### End

        # Observe
        op, r, done, infos = env.step(a)
        # Learn
        ####### Start
        # Here goes your learning update
        episode_log_probs.append(log_prob)
        episode_rewards.append(r)
        if done:
            batch_size_counter += 1
            memory['batch_rewards'].append(episode_rewards)
            memory['batch_log_probs'].append(episode_log_probs)

            if batch_size_counter == batch_size:
                agent.update(memory)
                batch_size_counter = 0
                del memory['batch_rewards'][:]
                del memory['batch_log_probs'][:]
                del episode_rewards[:]
                del episode_log_probs[:]
        o = op
        ####### End

        # Log
        ret += r
        if done:
            rets.append(ret)
            ret = 0
            o = env.reset()

        if (steps + 1) % checkpoint == 0:
            avgrets.append(np.mean(rets))
            rets = []
            plt.clf()
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            plt.pause(0.001)
    name = sys.argv[0].split('.')[-2].split('_')[-1]
    data = np.zeros((2, len(avgrets)))
    data[0] = range(checkpoint, num_steps + 1, checkpoint)
    data[1] = avgrets
    np.savetxt(os.path.join('results', name + str(seed) + ".txt"), data)
    plt.show()


if __name__ == "__main__":
    main()
