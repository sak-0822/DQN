import gym
from gym import wrappers
import numpy as np
import time
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

class Qfunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super(Qfunction, self).__init__()
        with self.init_scope():
            self.l0=L.Linear(obs_size, n_hidden_channels)
            self.l1=L.Linear(n_hidden_channels, n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels, n_actions)
    def __call__(self, x, test=False):
        h = F.tanh(self.l0(x))
        h = F.tanh(self.l1(h))
        y = chainerrl.action_value.DiscreteActionValue(self.l2(h))
        return y

env = gym.make('CartPole-v0')


gamma = 0.9
alpha = 0.5
max_number_of_steps = 200
num_episodes = 300


q_func = Qfunction(env.observation_space.shape[0], env.action_space.n)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.1, decay_steps=num_episodes, random_action_func=env.action_space.sample)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)
agent = chainerrl.agents.DQN(
        q_func, optimizer, replay_buffer, gamma, explorer, replay_start_size=500, update_interval=1, target_update_interval=100, phi=phi)


for episode in range(num_episodes):
    observation = env.reset()
    done = False 
    reward = 0
    R = 0

    
    for t in range(max_number_of_steps):
        if episode%100:
            env.render()
        action = agent.act_and_train(observation, reward)
        observation, reward, done, info = env.step(action)
        R += reward
        if done:
            break
    agent.stop_episode_and_train(observation, reward, done)
    if episode % 10 == 0:
        print('episode :',episode, 'total reward ', R,'statistics:', agent.get_statistics())
#agent.save('agent')
