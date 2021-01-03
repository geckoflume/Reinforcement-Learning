import numpy as np
import matplotlib.pyplot as plt

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

if __name__ == '__main__':
    correct_responses = np.load('data.npy')
    fig, ax = plt.subplots()

    for alpha_run in range(len(alphas)):                # for each alpha
        for epsilon_run in range(len(epsilons)):        # for each epsilon
            ax.plot(np.average(correct_responses[alpha_run][epsilon_run], axis=0), label="with memory α=" + str(alphas[alpha_run]) + ", ε=" + str(epsilons[epsilon_run]))      # with memory
    ax.set_title('Continuous spatial alternation')
    ax.set_ylabel('Performance')
    ax.set_xlabel('Step block')
    plt.legend(loc="upper right")
    plt.show()
