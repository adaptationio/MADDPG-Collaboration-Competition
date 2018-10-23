from unityagents import UnityEnvironment
import numpy as np
import random
import time  
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
#%matplotlib inline

env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

from agent import Agent

agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=4)
agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=4)


def ddpg(n_episodes=1000000, max_steps=10000, multi_agent=False, multi_replay=False, split_replay=False):
    scores_mean = deque(maxlen=100)
    scores = []
    best_score = 0
    best_average_score = 0
    for i_episode in range(1, n_episodes+1):                       
        average_score = 0
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations            
        scores_agents = np.zeros(num_agents)             
        score = 0
        if multi_agent:
            agent1.reset()
            agent2.reset()
        else:
            agent1.reset()

        for step in range(max_steps):
            if multi_agent:
                actions = np.random.randn(num_agents, action_size)
                actions[0] = agent1.act(states[0])                  # select an action (for each agent)
                actions[1] = agent2.act(states[1])
            else: 
                actions = agent1.act(states)
                env_info = env.step(actions)[brain_name]     
                next_states = env_info.vector_observations   
                rewards = env_info.rewards                   
                dones = env_info.local_done
            if multi_replay:
                if split_replay:
                    agent1.step(states[0], actions[0], rewards[0], next_states[0], dones[0], step)
                    agent2.step(states[1], actions[1], rewards[1], next_states[1], dones[1], step)
                else:
                    agent1.step(states, actions, rewards, next_states, dones, step)
                    agent2.step(states, actions, rewards, next_states, dones, step)
                
            else:             
                agent1.step(states, actions, rewards, next_states, dones, step)
                states = next_states
                scores_agents += rewards
            if np.any(dones):
                break

        score = scores_agents.max()
        scores_mean.append(score)
        average_score = np.mean(scores_mean)
        scores.append(score)
        if score > best_score:
            best_score = score
        if average_score > best_average_score:
            best_average_score = average_score
        if i_episode % 100 == 0:
            print("Episode:{}, Low Score:{:.2f}, High Score:{:.2f}, Score:{:.2f}, Best Score:{:.2f}, Average Score:{:.2f}, Best Avg Score:{:.2f}".format(i_episode, scores_agents.min(), scores_agents.max(), score, best_score, average_score, best_average_score))
        if average_score > 0.5:
            print("Episode:{}, Low Score:{:.2f}, High Score:{:.2f}, Score:{:.2f}, Best Score:{:.2f}, Average Score:{:.2f}, Best Avg Score:{:.2f}".format(i_episode, scores_agents.min(), scores_agents.max(), score, best_score, average_score, best_average_score))
            torch.save(agent1.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent1.critic_local.state_dict(), 'checkpoint_critic.pth') 
            if multi_agent:
                torch.save(agent1.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent1.critic_local.state_dict(), 'checkpoint_critic.pth')

            break
    return scores

scores = ddpg()