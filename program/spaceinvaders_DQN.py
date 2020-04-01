import gym
from gym import wrappers
import numpy as np
import time
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

class Qfunction(chainer.Chain):
    def __init__(self):
        super(Qfunction, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(3, 16, (11,9), 1, 0)
            self.conv2=L.Convolution2D(16, 32, (11,9), 1, 0)
            self.conv3=L.Convolution2D(32, 64, (11,9), 1, 0)
            self.l4 = L.Linear(14976, 6)


    def __call__(self, x, test=False):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x), ksize=2, stride=2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1), ksize=2, stride=2)
        h3 = F.max_pooling_2d(F.relu(self.conv3(h2), ksize=2, stride=2)
        return chainerrl.action_value.DiscreteActionValue(self.l4(h3))


def random_action():
    return np.random.choice([0, 1, 2, 3, 4, 5])




env = gym.make('CartPole-v0')


def main():
    gamma = 0.9
    alpha = 0.5
   # max_number_of_steps = 200
    num_episodes = 300


    q_func = Qfunction()
    optimizer = chainer.optimizers.Adam(eps=1e-2)
    optimizer.setup(q_func)
    explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.1, decay_steps=num_episodes, random_action_func=env.action_space.sample)
    replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
    phi = lambda x: x.astype(np.float32, copy=False)
    agent = chainerrl.agents.DQN(
           q_func, optimizer, replay_buffer, gamma, explorer, replay_start_size=5000, update_interval=50, target_update_interval=2000, phi=phi)

    
    outdir = 'result'
    env = gym.make('SpaceInvaders-v'
    chainerrl.misc.env_modifiers.make_reward_filtered(env.lambda x: x * 0.01)


    for episode in range(num_episodes):
        observation = env.reset()
        done = False 

    
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
