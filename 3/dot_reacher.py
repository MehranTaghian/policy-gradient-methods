import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt


# Environment

class Environment:
    def __init__(self, b=0.03):
        self.MIN_DIM = -1
        self.MAX_DIM = 1
        self.current_state = np.random.uniform(self.MIN_DIM, self.MAX_DIM, 2)
        self.reward = -0.01

        self.LEFT_ACTION = np.array([-b, 0])
        self.LEFT_UP_ACTION = np.array([-b, b]) / np.sqrt(2)
        self.UP_ACTION = np.array([0, b])
        self.RIGHT_UP_ACTION = np.array([b, b]) / np.sqrt(2)
        self.RIGHT_ACTION = np.array([b, 0])
        self.RIGHT_DOWN_ACTION = np.array([b, -b]) / np.sqrt(2)
        self.DOWN_ACTION = np.array([0, -b])
        self.LEFT_DOWN_ACTION = np.array([-b, -b]) / np.sqrt(2)

    def reset(self):
        self.current_state = np.random.uniform(self.MIN_DIM, self.MAX_DIM, 2)
        return self.current_state.copy()

    def step(self, action):
        self.take_action(action)
        done = self.check_termination()
        # reward = 0.0 if done else self.punish
        return self.current_state.copy(), self.reward, done

    def take_action(self, action):
        if action == 0:
            self.current_state += self.LEFT_ACTION
        elif action == 1:
            self.current_state += self.LEFT_UP_ACTION
        elif action == 2:
            self.current_state += self.UP_ACTION
        elif action == 3:
            self.current_state += self.RIGHT_UP_ACTION
        elif action == 4:
            self.current_state += self.RIGHT_ACTION
        elif action == 5:
            self.current_state += self.RIGHT_DOWN_ACTION
        elif action == 6:
            self.current_state += self.DOWN_ACTION
        elif action == 7:
            self.current_state += self.LEFT_DOWN_ACTION
        # elif action == 8:  # do nothing
        #     pass
        self.current_state = np.clip(self.current_state, self.MIN_DIM, self.MAX_DIM)

    def check_termination(self):
        return (self.current_state <= 0.2).all() and (self.current_state >= -0.2).all()


class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

        self.out.weight.data.fill_(0.0)
        self.out.bias.data.fill_(0.0)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.out(x)

    def sample(self, state):
        logits = self.forward(state)
        pol = torch.distributions.Categorical(logits=logits)
        action = pol.sample()
        log_prob = pol.log_prob(action)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.out(x)


class ActorCritic:
    def __init__(self, env, num_steps, gamma, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, device):
        self.num_steps = num_steps
        self.gamma = gamma
        self.env = env
        self.device = device

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def train(self):
        step = 0
        episode = 0
        trajectories = []
        while step < self.num_steps:
            done = False
            I = 1
            prior_steps = step
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            trajectory = []
            while not done:
                trajectory.append(state.numpy())
                action, log_prob = self.actor.sample(state)
                next_state, reward, done = self.env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
                step += 1

                v_next = self.critic(next_state).detach() if not done else 0
                delta = reward + self.gamma * v_next - self.critic(state)
                critic_loss = 1 / 2 * delta ** 2
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                actor_loss = - I * delta.detach() * log_prob
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                I = I * self.gamma
                state = next_state
                if step >= self.num_steps:
                    break

            episode += 1
            trajectories.append(trajectory)
            after_steps = step
            print(f"Episode {episode} length is:", after_steps - prior_steps)

        if len(trajectories) > 10:
            self.plot_initial_behavior(trajectories)

    def plot_initial_behavior(self, trajectories):
        plt.figure(figsize=[8, 8])
        colormap = plt.cm.get_cmap('hsv', 10)
        for t in range(10):
            trajectory = np.array(trajectories[t])
            alpha_step = 1 / trajectory.shape[0]
            alpha = 1 / trajectory.shape[0]
            for i in range(trajectory.shape[0] - 1):
                plt.plot(trajectory[i:i + 2, 1], trajectory[i:i + 2, 0], color=colormap(t),
                         alpha=alpha if alpha < 1 else 1)
                alpha += alpha_step
            # alpha = 1 / len(trajectory)
            # for step in range(len(trajectory)):
            #     line =
            #     ax.plot()

        plt.xlim([self.env.MIN_DIM, self.env.MAX_DIM])
        plt.ylim([self.env.MIN_DIM, self.env.MAX_DIM])
        plt.show()

    def plot_final_behavior(self, trajectories):
        pass

    def plot_learning_curves(self, episode_lenghts):
        pass


if __name__ == '__main__':
    # TODO: numpy seed should stay the same while torch seeds should vary
    env = Environment()
    gamma = 1
    num_steps = 30000
    hidden_dim = 10
    state_dim = 2
    action_dim = 9
    device = torch.device('cpu')

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    # actor_lr = 8e-4
    # critic_lr = 8e-4

    actor_lr = 1e-3
    critic_lr = 1e-3


    agent = ActorCritic(env, num_steps, gamma, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, device)
    agent.train()
