import matplotlib.pyplot as plt
import numpy as np
import os

experiments = ['ppo', 'reinforce']

PLOT_DIR = './plots'
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


def individual_plot(seeds=10):
    for e in experiments:
        fig = plt.figure(figsize=[15, 7])
        ax = fig.add_subplot(1, 1, 1)

        ax.set_ylim(0, 500)

        major_ticks_x = np.arange(0, 500001, 100000)
        minor_ticks_x = np.arange(0, 500001, 10000)

        major_ticks_y = np.arange(0, 501, 100)
        minor_ticks_y = np.arange(0, 501, 20)

        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)

        # Or if you want different settings for the grids:
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        for i in range(seeds):
            result = np.loadtxt(f'{e}{i}.txt')
            ax.plot(result[0], result[1], label=i)

        ax.set_xlabel("Number of time steps")
        ax.set_ylabel("Average reward")
        ax.set_title(f"Learning curves of the {e} algorithm on CartPole-v1")
        ax.legend(loc='lower right')
        plt.savefig(f'{PLOT_DIR}/{e}.jpg', dpi=300)
        plt.close()


def total_plot(seeds=10):
    fig = plt.figure(figsize=[15, 7])
    ax = fig.add_subplot(1, 1, 1)

    ax.set_ylim(0, 500)

    major_ticks_x = np.arange(0, 500001, 100000)
    minor_ticks_x = np.arange(0, 500001, 10000)

    major_ticks_y = np.arange(0, 501, 100)
    minor_ticks_y = np.arange(0, 501, 20)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    # Or if you want different settings for the grids:
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    for e in experiments:
        returns = []
        time_steps = None
        for i in range(seeds):
            seed_return = np.loadtxt(f'{e}{i}.txt')
            if time_steps is None:
                time_steps = seed_return[0]
            returns.append(seed_return[1])

        returns = np.array(returns)
        average_return = np.mean(returns, axis=0)
        standard_error = np.std(returns, axis=0) / np.sqrt(returns.shape[0])
        ax.plot(time_steps, average_return, label=e)
        ax.fill_between(time_steps, average_return - 2.26 * standard_error, average_return + 2.26 * standard_error,
                        alpha=0.2)

    ax.set_xlabel("Number of time steps")
    ax.set_ylabel("Average reward (averaged across 10 runs)")
    ax.set_title(f"Comparing the performance of PPO vs REINFORCE on CartPole-v1")
    ax.legend(loc='lower right')
    plt.savefig(f'{PLOT_DIR}/total_result.jpg', dpi=300)


if __name__ == '__main__':
    individual_plot()
    total_plot()
