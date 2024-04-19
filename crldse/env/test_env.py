import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(10):
    act = env.action_space.sample()
    print(act)
    observation, reward, done, info = env.step(act) # take a random action
    print('observation:{}, reward:{}, done:{}, info:{}'.format(observation, reward, done, info))
env.close()