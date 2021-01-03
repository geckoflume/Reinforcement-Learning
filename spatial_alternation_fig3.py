import numpy as np
import matplotlib.pyplot as plt
from spatial_alternation import compute

# spatial alternation task
# Figure 2: varying epsilon value from 0.1 to 1.0 and fixing alpha value to 0.1 (10 runs)

# 9 - 8 - 2 - 3 - 4
# |       |       |
# 10      1       5
# |       |       |
# 11-12 - 0 - 7 - 6

# start state : 0
# reward states : 4, 9

# parameter values for the algorithm
alphas = [0.1]
discount = 0.95
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
decay_rate = 0.25
theta = 0.01

if __name__ == '__main__':
    correct_responses = compute(alphas, discount, epsilons, decay_rate, theta)
    # correct_responses, correct_responses1 = compute(alphas, discount, epsilons, decay_rate, theta)

    '''
    plt.plot(np.average(episode_rewards, axis=0))
    plt.title('Continuous spatial alternation, 10 runs')
    plt.ylabel('Average reward')
    plt.xlabel('Step block')
    plt.show()

    plt.plot(np.average(episode_lengths, axis=0))
    plt.title('Continuous spatial alternation, 10 runs')
    plt.ylabel('Number of steps for each trial')
    plt.xlabel('Step block')
    plt.show()
    '''

    #fig, ax = plt.plot(np.average(correct_responses, axis=0))
    fig, ax = plt.subplots()
    #X = np.arange(num_episodes)
    #M =  np.mean(correct_responses, axis=0)
    #V = np.var(correct_responses, axis=0)
    #ax.plot(X, M)
    #ax.fill_between(X, M+V, M-V, alpha=0.1)

    '''
    for i in range(num_runs):
        ax.plot(correct_responses[i][:], alpha=0.1)
    '''

    for alpha_run in range(len(alphas)):                # for each alpha
        for epsilon_run in range(len(epsilons)):        # for each epsilon
            ax.plot(np.average(correct_responses[alpha_run][epsilon_run], axis=0), label="with memory α=" + str(alphas[alpha_run]) + ", ε=" + str(epsilons[epsilon_run]))      # with memory
            # ax.plot(np.average(correct_responses1[alpha_run][epsilon_run], axis=0), label="without memory α=" + str(alphas[alpha_run]) + ", ε=" + str(epsilons[epsilon_run]))  # without memory
    ax.set_title('Continuous spatial alternation')
    ax.set_ylabel('Performance')
    ax.set_xlabel('Step block')
    plt.legend(loc="upper right")
    plt.show()
