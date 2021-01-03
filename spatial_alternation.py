import numpy as np

# spatial alternation task

# 9 - 8 - 2 - 3 - 4
# |       |       |
# 10      1       5
# |       |       |
# 11-12 - 0 - 7 - 6

# start state : 0
# reward states : 4, 9

# action dictionary
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
UPDATE = 4

# epsilon greedy policy for the actor
def policy(Q_values, epsilon, nA):
    random_prob = np.random.uniform(0, 1)
    if(random_prob < epsilon) or (not np.any(Q_values)):
        action = np.random.randint(low=0, high=nA)
    else:
        action = np.argmax(Q_values)
    return action


def take_action(loc_state, action):
    location_transition = [(1, 0, 0, 0),
                           (2, 1, 1, 1),
                           (2, 2, 3, 8),
                           (3, 3, 4, 3),
                           (4, 5, 4, 4),
                           (5, 6, 5, 5),
                           (6, 6, 7, 6),
                           (7, 7, 0, 7),
                           (8, 8, 8, 9),
                           (9, 10, 9, 9),
                           (10, 11, 10, 10),
                           (11, 11, 11, 12),
                           (12, 12, 12, 0)]
    next_loc_state = location_transition[loc_state][action]
    if next_loc_state != loc_state:
        reward = -0.05
    else:
        reward = -1
    return next_loc_state, reward


def compute(alphas, discount, epsilons, decay_rate, theta):
    n_states = 13
    n_actions = 5
    actions_dict = {UP: 'up', DOWN: 'down', RIGHT: 'right', LEFT: 'left', UPDATE: 'update'}

    num_episodes = 70
    num_runs = 10
    n_steps = 2000

    episode_rewards = np.zeros((num_runs, num_episodes))
    episode_lengths = np.zeros((num_runs, num_episodes))
    correct_responses = np.zeros((len(alphas), len(epsilons), num_runs, num_episodes))

    for alpha_run in range(len(alphas)):
        for epsilon_run in range(len(epsilons)):
            print("α={} ({}/{}), ε={} ({}/{})".format(alphas[alpha_run], alpha_run + 1, len(alphas), epsilons[epsilon_run], epsilon_run + 1, len(epsilons)))
            for i_run in range(num_runs):

                state_values = np.zeros((n_states, n_states+1))
                action_values = np.zeros((n_states, n_states+1, n_actions))

                state_history = []

                # sample trial for each run
                state = [0, -1]  # location state, memory state

                for t in range(n_steps):

                    # actor, take action with highest probability
                    loc_state = state[0]
                    mem_state = state[1]
                    action = policy(action_values[state[0]][state[1]], epsilons[epsilon_run], n_actions)
                    if loc_state == 2:
                        remember = action
                    if action != 4:
                        next_loc_state, reward = take_action(loc_state, action)
                        next_state = [next_loc_state, mem_state]
                        if (loc_state == 3 and next_loc_state == 4) or (loc_state == 8 and next_loc_state == 9):
                            reward = 9.5
                    else:
                        if mem_state == loc_state:
                            reward = -1
                        else:
                            reward = -0.05
                        next_mem_state = loc_state
                        next_state = [loc_state, next_mem_state]

                    #print(state, actions_dict[action], next_state, reward)

                    # critic, caluculate td error
                    td_error = reward + discount * state_values[next_state[0]][next_state[1]] - state_values[state[0]][state[1]]
                    state_values[state[0]][state[1]] += alphas[alpha_run]*td_error
                    action_values[state[0]][state[1]][action] += alphas[alpha_run]*td_error

                    state = next_state
                    if state[0] == 0 and t > 7:
                        break

                #print("action taken", actions_dict[remember])
                # for the sample trial, going right or left obtains a reward

                for i_episode in range(num_episodes):
                    # each episode block is of lenght 2000  steps
                    ncr = 0
                    tr = 1
                    state = [0, -1]
                    print("\rEpisode {}/{}".format(i_episode, num_episodes))
                    if actions_dict[remember] == "left":
                        reward_at = "right"
                    elif actions_dict[remember] == "right":
                        reward_at = "left"
                    #print("reward now at ", reward_at)
                    total_reward = 0

                    for t in range(n_steps):
                        # actor, take action with highest probability
                        loc_state = state[0]
                        mem_state = state[1]
                        action = policy(action_values[state[0]][state[1]], epsilons[epsilon_run], n_actions)
                        if loc_state == 2:
                            remember = action
                        if action != 4:
                            next_loc_state, reward = take_action(loc_state, action)
                            next_state = [next_loc_state, mem_state]
                            if (reward_at == "right" and loc_state == 3 and next_loc_state == 4) or (reward_at == "left" and loc_state == 8 and next_loc_state == 9):
                                reward = 9.5
                                ncr += 1
                            if (reward_at == "left" and loc_state == 3 and next_loc_state == 4) or (reward_at == "right" and loc_state == 8 and next_loc_state == 9):
                                reward = -6

                        else:
                            if mem_state == loc_state:
                                reward = -1
                            else:
                                reward = -0.05
                            next_mem_state = loc_state
                            next_state = [loc_state, next_mem_state]
                        total_reward += reward
                        #print(state, actions_dict[action], next_state, reward, action_values[state[0]][state[1]])

                        # critic, caluculate td error
                        td_error = reward + discount * state_values[next_state[0]][next_state[1]] - state_values[state[0]][state[1]]
                        state_values[state[0]][state[1]] += alphas[alpha_run]*td_error
                        action_values[state[0]][state[1]][action] += alphas[alpha_run]*td_error

                        if next_state[0] == 0 and state[0] != 0:
                            tr += 1
                            #print("reward was at ", reward_at, "response was ", actions_dict[remember], "memory state ", state[1])
                            if actions_dict[remember] == "left":
                                reward_at = "right"
                            elif actions_dict[remember] == "right":
                                reward_at = "left"
                        state = next_state
                        if t == n_steps-1:
                            correct_responses[alpha_run][epsilon_run][i_run][i_episode] = ncr/tr
                            episode_rewards[i_run][i_episode] = total_reward/tr
                            episode_lengths[i_run][i_episode] = n_steps/tr
                            print("trials = ", tr, " correct responses = ", ncr)
    '''
    # without working memory

    n_states = 13
    n_actions = 4
    actions_dict = {UP: 'up', DOWN: 'down', RIGHT: 'right', LEFT: 'left', UPDATE: 'update'}

    num_episodes = 70
    num_runs = 10
    n_steps = 2000

    episode_rewards1 = np.zeros((num_runs, num_episodes))
    episode_lengths1 = np.zeros((num_runs, num_episodes))
    correct_responses1 = np.zeros((len(alphas), len(epsilons), num_runs, num_episodes))

    for alpha_run in range(len(alphas)):
        for epsilon_run in range(len(epsilons)):
            print("α={} ({}/{}), ε={} ({}/{})".format(alphas[alpha_run], alpha_run + 1, len(alphas), epsilons[epsilon_run], epsilon_run + 1, len(epsilons)))
            for i_run in range(num_runs):

                state_values = np.zeros(n_states)
                action_values = np.zeros((n_states, n_actions))

                # sample trial for each run
                state = 0  # location state, memory state

                for t in range(n_steps):

                    # actor, take action with highest probability
                    loc_state = state
                    action = policy(action_values[loc_state], epsilons[epsilon_run], n_actions)
                    if loc_state == 2:
                        remember = action
                    next_loc_state, reward = take_action(loc_state, action)
                    if (loc_state == 3 and next_loc_state == 4) or (loc_state == 8 and next_loc_state == 9):
                        reward = 9.5

                    #print(state, actions_dict[action], next_state, reward)

                    # critic, caluculate td error
                    td_error = reward + discount * state_values[next_loc_state] - state_values[state]
                    state_values[state] += alphas[alpha_run]*td_error
                    action_values[state][action] += alphas[alpha_run]*td_error

                    state = next_loc_state
                    if state == 0 and t > 7:
                        break

            # print("action taken", actions_dict[remember])
                # for the sample trial, going right or left obtains a reward

                for i_episode in range(num_episodes):
                    # each episode block is of lenght 2000  steps
                    ncr = 0
                    tr = 1
                    state = 0
                    print("\rEpisode {}/{}".format(i_episode, num_episodes))
                    if actions_dict[remember] == "left":
                        reward_at = "right"
                    elif actions_dict[remember] == "right":
                        reward_at = "left"
                    #print("reward now at ", reward_at)
                    total_reward = 0

                    for t in range(n_steps):
                        # actor, take action with highest probability
                        loc_state = state
                        action = policy(action_values[loc_state], epsilons[epsilon_run], n_actions)
                        if loc_state == 2:
                            remember = action
                        next_loc_state, reward = take_action(loc_state, action)
                        if (reward_at == "right" and loc_state == 3 and next_loc_state == 4) or (reward_at == "left" and loc_state == 8 and next_loc_state == 9):
                            reward = 9.5
                            ncr += 1
                        if (reward_at == "left" and loc_state == 3 and next_loc_state == 4) or (reward_at == "right" and loc_state == 8 and next_loc_state == 9):
                            reward = -6

                        total_reward += reward
                        #print(state, actions_dict[action], next_state, reward, action_values[state[0]][state[1]])

                        # critic, caluculate td error
                        td_error = reward + discount * state_values[next_loc_state] - state_values[state]
                        state_values[state] += alphas[alpha_run]*td_error
                        action_values[state][action] += alphas[alpha_run]*td_error

                        if next_loc_state == 0 and state != 0:
                            tr += 1
                            #print("reward was at ", reward_at, "response was ", actions_dict[remember], "memory state ", state[1])
                            if actions_dict[remember] == "left":
                                reward_at = "right"
                            elif actions_dict[remember] == "right":
                                reward_at = "left"
                        state = next_loc_state
                        if t == n_steps-1:
                            correct_responses1[alpha_run][epsilon_run][i_run][i_episode] = ncr/tr
                            episode_rewards1[i_run][i_episode] = total_reward/tr
                            episode_lengths1[i_run][i_episode] = n_steps/tr
                            print("trials = ", tr, " correct responses = ", ncr)
    '''
    return correct_responses #, correct_responses1
