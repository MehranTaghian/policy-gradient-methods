import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    def train(self, limit_num_episode=None):
        step = 0
        episode = 0
        trajectories = []
        episode_lengths = []
        pbar = tqdm(total=self.num_steps) if limit_num_episode is None else tqdm(total=limit_num_episode)

        while step < self.num_steps:
            done = False
            I = 1
            prior_step = step
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

                v_next = self.critic(next_state).detach() if not done else 0.0
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
            episode_length = step - prior_step
            # print(f"Episode {episode} length is:", episode_length)
            episode_lengths.append([prior_step, episode_length])
            if limit_num_episode is None:
                # update per number of steps
                pbar.update(episode_length)
            else:
                # update one episode
                pbar.update(1)
            if limit_num_episode is not None and episode == limit_num_episode:
                break

        return np.array(episode_lengths), trajectories

    def plot_initial_behavior(self, trajectories):
        plt.figure(figsize=[8, 8])
        colormap = plt.cm.get_cmap('hsv', 10)
        for t in range(10):
            trajectory = np.array(trajectories[t])
            alpha_step = 1 / trajectory.shape[0]
            alpha = 1 / trajectory.shape[0]

            for i in range(trajectory.shape[0] - 1):
                plt.plot(trajectory[i:i + 2, 1], trajectory[i:i + 2, 0], color=colormap(t),
                         alpha=np.exp(alpha) if np.exp(alpha) < 1 else 1)
                alpha += alpha_step

        plt.xlim([self.env.MIN_DIM, self.env.MAX_DIM])
        plt.ylim([self.env.MIN_DIM, self.env.MAX_DIM])
        plt.grid(axis='both', color='gray')
        plt.show()

    def plot_final_behavior(self, trajectories):
        plt.figure(figsize=[8, 8])
        colormap = plt.cm.get_cmap('hsv', 30)
        for t in range(-30, 0):
            trajectory = np.array(trajectories[t])
            alpha_step = 1 / trajectory.shape[0]
            alpha = 1 / trajectory.shape[0]
            for i in range(trajectory.shape[0] - 1):
                plt.plot(trajectory[i:i + 2, 1], trajectory[i:i + 2, 0], color=colormap(-t),
                         alpha=alpha if alpha < 1 else 1)
                alpha += alpha_step

        plt.xlim([self.env.MIN_DIM, self.env.MAX_DIM])
        plt.ylim([self.env.MIN_DIM, self.env.MAX_DIM])
        plt.grid(axis='both', color='gray')
        plt.show()


def plot_learning_curves(episode_lengths, first_part_length):
    # Plot first part
    plt.figure(figsize=[8, 8])
    seed = 0
    for el in episode_lengths:
        indices = np.where(el[:, 0] <= first_part_length)[0]
        plt.plot(el[indices, 0], el[indices, 1], label=f'seed {seed}')
        seed += 1
    plt.xlabel("Number of time-steps")
    plt.ylabel("Episode length")
    plt.grid(axis='both', color='gray')
    plt.legend()
    plt.show()

    # plot second part
    plt.figure(figsize=[8, 8])
    seed = 0
    for el in episode_lengths:
        indices = np.where(el[:, 0] > first_part_length)[0]
        plt.plot(el[indices, 0], el[indices, 1], label=f'seed {seed}')
        seed += 1
    plt.ylim([0, 100])
    plt.xlabel("Number of time-steps")
    plt.ylabel("Episode length")
    plt.grid(axis='both', color='gray')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # TODO: numpy seed should stay the same while torch seeds should vary
    env = Environment()
    gamma = 1
    num_seeds = 10
    hidden_dim = 10
    state_dim = 2
    action_dim = 9
    device = torch.device('cpu')

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    actor_lr = 1e-3
    critic_lr = 1e-3

    num_steps = 10000
    agent = ActorCritic(env, num_steps, gamma, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, device)
    _, trajectories = agent.train()

    if len(trajectories) > 10:
        agent.plot_initial_behavior(trajectories)
        agent.plot_final_behavior(trajectories)

    num_steps = 40000
    episode_lengths = []
    # numpy's random should be initialized once so that the environment has the same random seed among all the runs.
    np.random.seed(0)
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        agent = ActorCritic(env, num_steps, gamma, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, device)
        episode_length, _ = agent.train()
        episode_lengths.append(episode_length)

    plot_learning_curves(episode_lengths, int(num_steps / 2))
