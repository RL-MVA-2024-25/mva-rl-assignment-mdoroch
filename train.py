from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import os
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_HIV_population
import torch.nn as nn
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import random
from tqdm import tqdm 
from IPython.display import clear_output

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

import numpy as np
import os
import pickle
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

STATE_SIZE = 6                  
ACTION_SIZE = 4                   
GAMMA = 0.98                     
ALPHA = 0.001
EPSILON = 1 #1.0             
EPSILON_DECAY = 0.98    
EPSILON_MIN = 0.15                
MEMORY_SIZE = 1000000  
BATCH_SIZE = 256                 


def plot_rewards(rews, loss_history):
    clear_output(wait=True)
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 8)) 
    
    axs[0].plot(np.arange(len(rews)), rews, marker='o', color='b', label='Rewards')
    axs[0].set_title('Total Reward')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(np.arange(len(loss_history)), loss_history, marker='x', color='r', label='Loss')
    axs[1].set_title('Loss History')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Loss')
    axs[1].grid()
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
def add_features(arr):
    new_features = np.array([arr[0] / arr[1],
                            arr[2] / arr[3],
                            arr[4] / arr[5],
                            arr[0] / arr[2],
                            arr[1] / arr[3]])
    
    return np.concatenate((arr, new_features))

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        # self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(256, 256)
        self.relu3 = nn.ReLU()
        # self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(256, 256)
        self.relu4 = nn.ReLU()
        
        self.fc5 = nn.Linear(256, 256)
        self.relu5 = nn.ReLU()
        
        self.fc6 = nn.Linear(256, action_size)        
    def forward(self, x):
        # print(x.shape, self.fc1(x).shape)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        return self.fc6(x)
    
class ProjectAgent:
    
    def __init__(self):
        self.model = DQNNetwork(STATE_SIZE, ACTION_SIZE).to(device)
        self.target_model = DQNNetwork(STATE_SIZE, ACTION_SIZE).to(device)
        
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=ALPHA)
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.criterion = nn.HuberLoss() #nn.MSELoss()  
        self.epsilon = EPSILON
        self.loss_history = []
        
    def act(self, observation, use_random=False):

        #self.model.eval()
    
        observation = torch.FloatTensor(observation).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(observation)
        return torch.argmax(q_values).item()
    
    def random_action(self):
        return np.random.choice(ACTION_SIZE)
    
    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))
        
    
    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # self.model.train()
        
        
        q_values = self.model(states).gather(1, actions).squeeze(1)


        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + GAMMA * next_q_values #* (1 - dones)

        
        loss = self.criterion(q_values, target_q_values)
        
        # print(loss.shape, q_values.shape, target_q_values.shape)
        
        
        self.loss_history.append(loss.item())
        
        # print(f"Q-values: {q_values}, Target Q-values: {target_q_values}, Loss: {loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_model(self):

        self.target_model.load_state_dict(self.model.state_dict())
        
    def soft_update(self, tau=0.001):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filepath = f'model_best_2_v8_179_19792695857.256184.pth'):

        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath = f'model_best_2_v8_179_19792695857.256184.pth'):

        self.model.load_state_dict(torch.load(filepath, map_location = torch.device('cpu')))
        self.model.eval()  
        
