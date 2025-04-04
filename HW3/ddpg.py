# Spring 2024, 535514 Reinforcement Learning
# HW3: DDPG
# pip install numpy==1.24
# pip install gym=0.25.2
import sys
import warnings
import gym
import numpy as np
import os
import time
import random
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_Pendulum300")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network
        self.fc1 = nn.Sequential(
            nn.Linear(num_inputs, 400, device=device),
            nn.ReLU(),
            nn.Linear(400, 300, device=device),
            nn.ReLU(),
            nn.Linear(300, num_outputs, device=device),
            nn.Tanh()
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(400, 300, device=device),
        #     nn.ReLU()
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Linear(300, num_outputs, device=device),
        #     nn.Tanh()
        # )
        ########## END OF YOUR CODE ##########
        
    def forward(self, inputs):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        x = self.fc1(inputs)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own critic network
        self.fc1 = nn.Sequential(
            nn.Linear(num_inputs, 400, device=device),
            nn.ReLU()
        )   
        self.fc2 = nn.Sequential(
            nn.Linear(400 + num_outputs, 300, device=device),
            nn.ReLU(),
            nn.Linear(300, 1, device=device)
        )
        #self.fc3 = nn.Linear(300, 1, device=device)
        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network
        x = self.fc1(inputs)
        x = self.fc2(torch.cat([x, actions], 1))
        #x = self.fc3(x)
        return x
        ########## END OF YOUR CODE ##########        
        

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.995, tau=0.0005, hidden_size=128, lr_a=1e-4, lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor) 
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor((Variable(state.to(device))))
        mu = mu.data

        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed 
        # add noise to action
        if action_noise is not None:
            mu += torch.tensor(action_noise).to(device)

        # clip action, set action between -1 and 1
        return torch.clamp(mu, -2, 2).cpu()
        ########## END OF YOUR CODE ##########


    def update_parameters(self, batch):
        state_batch = Variable(batch.state)
        action_batch = Variable(batch.action)
        reward_batch = Variable(batch.reward)
        mask_batch = Variable(batch.mask)
        next_state_batch = Variable(batch.next_state)
        
        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        # compute TD target, set Q_target to 0 if next state is terminal
        with torch.no_grad():    
            Q_targets = reward_batch + (self.gamma * next_state_action_values * (1 - mask_batch))

        # predict Q-value in current state
        state_action_batch = self.critic(state_batch, action_batch)
        
        # compute critic loss (MSE loss)
        value_loss = F.mse_loss(state_action_batch, Q_targets)

        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # predict action in current state
        actions_pred = self.actor(state_batch)
        
        # compute actor loss (policy gradient)
        policy_loss = -self.critic(state_batch, actions_pred).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        ########## END OF YOUR CODE ########## 

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        local_time = time.localtime()
        timestamp = time.strftime("%m%d%Y_%H%M%S", local_time)
        if not os.path.exists('preTrained/'):
            os.makedirs('preTrained/')

        if actor_path is None:
            actor_path = "preTrained/ddpg_actor_{}_{}_{}".format(env_name, timestamp, suffix) 
        if critic_path is None:
            critic_path = "preTrained/ddpg_critic_{}_{}_{}".format(env_name, timestamp, suffix) 
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

def train():    
    num_episodes = 1000
    gamma = 0.995
    tau = 0.002
    hidden_size = 128
    noise_scale = 0.3
    replay_size = 100000
    batch_size = 256
    updates_per_step = 1
    print_freq = 1
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    
    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size)
    ounoise = OUNoise(env.action_space.shape[0])
    memory = ReplayMemory(replay_size)
    
    for i_episode in range(num_episodes):
        
        ounoise.scale = noise_scale
        ounoise.reset()
        
        state = torch.Tensor(env.reset())

        episode_reward = 0
        value_loss, policy_loss = 0, 0
        while True:
            
            ########## YOUR CODE HERE (15~25 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic

            # select action and interact with the environment
            # add noise to action for exploration
            action = agent.select_action(state, ounoise.noise())
            next_state, reward, done, _ = env.step(action.numpy())
            
            # add sample to replay buffer
            # convert to numpy array, since replay buffer only accepts numpy array
            memory.push(state.numpy(), action.numpy(), done, next_state, reward)

            # update the actor and the critic
            if memory.__len__() > batch_size:
                experiences_batch = memory.sample(batch_size)

                # convert to Transition object
                # Since the replay buffer stores numpy array, we need to convert them to torch tensor
                # and move them to GPU
                experiences_batch = Transition(state=torch.from_numpy(np.vstack([i.state for i in experiences_batch])).to(torch.float32).to(device),
                                               action=torch.from_numpy(np.vstack([i.action for i in experiences_batch])).to(torch.float32).to(device),
                                               mask=torch.from_numpy(np.vstack([i.mask for i in experiences_batch])).to(torch.uint8).to(device),
                                               next_state=torch.from_numpy(np.vstack([i.next_state for i in experiences_batch])).to(torch.float32).to(device),
                                               reward=torch.from_numpy(np.vstack([i.reward for i in experiences_batch])).to(torch.float32).to(device))
                
                # update the actor and the critic
                value_loss, policy_loss = agent.update_parameters(experiences_batch)
            
            # update the state
            state = torch.Tensor(next_state).clone()
            episode_reward += reward

            if done:
                break

            ########## END OF YOUR CODE ########## 
            

        rewards.append(episode_reward)
        t = 0
        if i_episode % print_freq == 0:
            state = torch.Tensor([env.reset()])
            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, done, _ = env.step(action.numpy()[0])
                
                #env.render()
                
                episode_reward += reward

                next_state = torch.Tensor([next_state])

                state = next_state
                
                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)           
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))

            # write results to tensorboard
            writer.add_scalar('Reward/ewma', ewma_reward, i_episode)
            writer.add_scalar('Reward/ep_reward', rewards[-1], i_episode)
            writer.add_scalar('Loss/value', value_loss, i_episode)
            writer.add_scalar('Loss/policy', policy_loss, i_episode)
    
    env_name = 'Pendulum-v1'
    agent.save_model(env_name, suffix="DDPG")      
 
def test():
    num_episodes = 10
    render = True
    env = gym.make('Pendulum-v1')
    agent = DDPG(env.observation_space.shape[0], env.action_space)
    agent.load_model(actor_path='./preTrained/ddpg_actor_Pendulum-v1_05082024_030212_DDPG',
                        critic_path='./preTrained/ddpg_critic_Pendulum-v1_05082024_030212_DDPG')
    for i_episode in range(1, num_episodes+1):
        state = torch.Tensor([env.reset()])
        episode_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action.numpy()[0])
            # if render:
            #     env.render()
            episode_reward += reward
            next_state = torch.Tensor([next_state])
            state = next_state
            if done:
                break
        print("Episode: {}, reward: {:.2f}".format(i_episode, episode_reward))


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10
    env = gym.make('Pendulum-v1')
    env.seed(random_seed)
    #train()
    test()