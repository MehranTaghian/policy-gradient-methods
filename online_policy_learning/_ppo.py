import sys
import numpy as np
# import matplotlib as mpl
# mpl.use("TKAgg")
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists('results'):
    os.makedirs('results')


class Memory:
    def __init__(self, capacity, gamma, N, gae_lambda):
        """
        :param capacity:
        :param gamma:
        :param N: number of mini-batches
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.capacity = capacity
        self.N = N
        self.actions = []
        self.states = []
        self.values = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.values[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def sample(self):
        G_tn = []
        lambda_return = []
        discounted_reward = 0
        value_gamma = self.gamma
        index = len(self.values) - 1

        for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                G_tn = []
                value_gamma = self.gamma
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            G_tn.insert(0, discounted_reward + value_gamma * self.values[index])
            value_gamma *= self.gamma
            lambda_return.insert(0, sum(self.gae_lambda ** i * G_tn[i] for i in range(len(G_tn))))
            index -= 1

        lambda_return = (1 - self.gae_lambda) * torch.tensor(lambda_return, dtype=torch.float32).to(device)
        lambda_return = (lambda_return - lambda_return.mean()) / (lambda_return.std() + 1e-7)

        # convert list to tensor
        states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(device)
        actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(device)
        logprobs = torch.squeeze(torch.stack(self.logprobs, dim=0)).detach().to(device)

        # shuffle
        indices = np.arange(self.capacity)
        np.random.shuffle(indices)
        shuffled_states = states[indices]
        shuffled_actions = actions[indices]
        shuffled_log_probs = logprobs[indices]
        shuffled_lambda_return = lambda_return[indices]

        mini_batches = []
        mini_batch_size = int(self.capacity / self.N)

        for i in range(self.N):
            mini_batch_states = shuffled_states[i * mini_batch_size: (i + 1) * mini_batch_size]
            mini_batch_actions = shuffled_actions[i * mini_batch_size: (i + 1) * mini_batch_size]
            mini_batch_log_probs = shuffled_log_probs[i * mini_batch_size: (i + 1) * mini_batch_size]
            mini_batch_lambda_return = shuffled_lambda_return[i * mini_batch_size: (i + 1) * mini_batch_size]

            mini_batches.append((mini_batch_states,
                                 mini_batch_actions,
                                 mini_batch_log_probs,
                                 mini_batch_lambda_return))

        return mini_batches


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, buffer, lr_actor, lr_critic, epochs, eps_clip):
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.buffer = buffer

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
            state_value = self.policy.critic(state)

        self.buffer.states.append(state)
        self.buffer.values.append(state_value)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def update(self):
        mini_batches = self.buffer.sample()
        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Evaluating old actions and values
            for old_states, old_actions, old_logprobs, rewards in mini_batches:
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


def main():
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])

    # Task setup block starts
    # Do not change
    env = gym.make('CartPole-v1')
    env.seed(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    # Task setup block end

    # Learner setup block
    torch.manual_seed(seed)
    ####### Start
    epochs = 10
    gamma = 0.99
    gae_lambda = 0.95
    eps_clip = 0.2
    update_freq = 2000
    N = 20
    buffer = Memory(capacity=update_freq, gamma=gamma, N=N, gae_lambda=gae_lambda)
    agent = PPO(state_dim=state_dim,
                action_dim=action_dim,
                buffer=buffer,
                lr_actor=3e-4,
                lr_critic=3e-4,
                epochs=epochs,
                eps_clip=eps_clip)
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
        a = agent.select_action(o)
        ####### End

        # Observe
        op, r, done, infos = env.step(a)

        # Learn
        ####### Start
        # Here goes your learning update
        agent.buffer.rewards.append(r)
        agent.buffer.is_terminals.append(done)
        if (steps + 1) % update_freq == 0:
            agent.update()
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
