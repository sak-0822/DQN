import numpy as np
from rat_class import Rat


def random_action():
    return np.random.choice([0, 1])

def get_action(next_state, episode):
    epsilon = 0.5 * (1 / (episode + 1))

    if epsilon <= np.random.uniform(0, 1):
        a = np.where(q_table[next_state]==q_table[next_state].max())[0]
        next_action = np.random.choice(a)

    else:
        next_action = random_action()

    return next_action

def update_Qtable(q_table, state, action, reward, next_state):
    gamma = 0.9
    alpha = .5
    next_maxQ = max(q_table[next_state])

    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * next_maxQ)

    return q_table

max_number_of_steps = 5
num_episodes = 10
q_table = np.zeros((2,2))
env = Rat()

for episode in range(num_episodes):
    state = 0
    episode_reward = 0

    for t in range(max_number_of_steps):
        action = get_action(state, episode)
        next_state, reward = env.step(action)
        print(state, action, reward)
        episode_reward += reward
        q_table = update_Qtable(q_table, state, action, reward, next_state)
        state = next_state


    print('episode : %d total reward %d' %(episode+1, episode_reward))
    print(q_table)





