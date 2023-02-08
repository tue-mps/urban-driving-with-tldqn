import random
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch import optim
import os
from collections import deque
import math

# the defination of the hyper parameters
BATCH_SIZE = 64  # batch size of the training data
LR = 0.0005 # learning rate
EPSILON = 0.6  # greedy algorithm
GAMMA = 0.9  # reward discount
TARGET_UPDATE = 100  # update the target network after training
MEMORY_CAPACITY = 5000  # the capacity of the memory
# N_STATE = 4  # the number of states that can be observed from environment
# N_ACTION = 3  # the number of action that the agent should have
# N_CHANNEL = 6
# decide the device used to train the network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the structure of the network


class Linear_Net(nn.Module):
    def __init__(self, n_state=None, n_action=None):
        super(Linear_Net, self).__init__()
        self.fc1 = nn.Linear(n_state, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(64, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(256, n_action)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = x.to(device)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)
        return out


class CNN(nn.Module):
    def __init__(self, n_channel=None, out_channel=None):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channel, out_channels=16, kernel_size=3, stride=4, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(50 * 50 * 64, 256)
        self.fc2 = nn.Linear(256, out_channel)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).float().to(device)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        out = self.tanh(self.fc2(x))

        return out


class DDQN(object):
    def __init__(self, n_state = None, n_action = None, test=False, var_eps=True, agent_name=None, threshold=0.2):
        self.policy_net, self.target_net = Linear_Net(n_state, n_action).to(device), \
                                           Linear_Net(n_state, n_action).to(device)
        self.n_state = n_state
        self.n_action = n_action
        self.learn_step = 0
        self.memory_counter = 0
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.test_mode = test
        self.var_eps = var_eps
        self.agent_name = agent_name
        self.eval_model_load_path = "test"
        self.q_threshold = threshold

    def action_value(self, state):
        state = torch.unsqueeze(torch.tensor(state), dim=0)
        # here directly output the action value
        return self.policy_net.forward(state)

    def TLQ_action_selection(self, action_set, state):
        state = torch.unsqueeze(torch.tensor(state), dim=0)
        #means current objective have no choice but only choose this action
        if len(action_set) == 1:
            return action_set[0], action_set
        else:
            q_value = self.action_value(state).data.cpu().numpy()[0][0]
            q_action_set = q_value[action_set]
            max_action_value = np.max(q_action_set)
            max_action = action_set[np.argmax(q_action_set)]
            low_bound = max_action_value - self.q_threshold
            new_action_set = action_set[q_action_set >= low_bound]
            new_action = random.choice(new_action_set)
        return max_action, new_action_set

    def find_action_range(self, pre_action_range = None, batch_s_=None):
        #return the action value from the batch
        action_value = self.action_value(batch_s_)
        if pre_action_range is not None:
            new_action_value = torch.squeeze(action_value, dim=0)
            max_list = []
            #here will find the max action value from pre_action_set
            condition = torch.squeeze(pre_action_range == 1, dim=0)
            #find the max value in the action value,  shape:(1, batch_size)
            for i in range(len(new_action_value)):
                max_value = torch.max(new_action_value[i][condition[i]])
                max_list.append(max_value)
            max_value_tensor = torch.unsqueeze(torch.tensor(max_list).to(device), dim=0)
        else:
            max_value_tensor = torch.max(action_value, dim=2).values
        #make a array have same shape with max value, used to calculate low bound of q value
        threshold_list = (torch.ones(1, 64) * self.q_threshold).to(device)
        # lowest acceptable q value for the action set selection
        low_bound = max_value_tensor - threshold_list
        low_bound = torch.unsqueeze(low_bound, dim=2)
        # the action value above the low bound will be marked by one
        action_range = torch.where(action_value >= low_bound, 1, 0)
        if pre_action_range is not None:
            action_range = action_range & pre_action_range
        else:
            pass
        return action_range


    def select_action(self, state):
        state = torch.unsqueeze(torch.tensor(state), dim=0)
        p = np.random.random()

        if os.path.exists(self.eval_model_load_path):
            E_thresh = EPS_END
        else:
            E_thresh = EPS_END + (EPS_START - EPS_END) * \
                       math.exp(-1. * self.learn_step / EPS_DECAY)
        if self.test_mode:
            E_thresh = 0.0
        if p > E_thresh:
            actions_value = self.policy_net.forward(state)
            return torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:
            return np.random.randint(0, self.n_action)

    def store_transition(self, s, a, r, s_, done):
        transition = [s, a, r, s_, done]
        self.memory.append(transition)
        self.memory_counter += 1


    def optimize_model(self, pre_DQN_list = None):

        if self.learn_step % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.learn_step += 1

        # get the samples to train the policy net
        sample_index = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
        if pre_DQN_list is None:
            pass
        else:
            action_range = None
            for i in range(len(pre_DQN_list)):
                # here should sample s_ from each DQN memory
                batch_s_ = torch.FloatTensor(np.array([transition[3] for transition in [pre_DQN_list[i].memory[j] for j in sample_index]])).to(device)
                action_range = pre_DQN_list[i].find_action_range(action_range, batch_s_)
            condition = torch.squeeze(action_range == 1, dim=0)
        # here is the sample from current DQN
        sample_batch = [self.memory[i] for i in sample_index]
        batch_s = torch.FloatTensor(np.array([transition[0] for transition in sample_batch])).to(device)
        batch_a = torch.LongTensor(np.array([transition[1] for transition in sample_batch])).unsqueeze(dim=1).to(device)
        batch_r = torch.FloatTensor(np.array([transition[2] for transition in sample_batch])).to(device)
        batch_s_ = torch.FloatTensor(np.array([transition[3] for transition in sample_batch])).to(device)
        # calculate the q_value
        policy_out_next = self.policy_net(batch_s_)
        if pre_DQN_list is None:
            max_a_batch = torch.argmax(policy_out_next, dim=1).unsqueeze(dim=1)
        else:
            max_a_batch = []
            for i in range(len(policy_out_next)):
                max_value = torch.max(policy_out_next[i][condition[i]])
                max_action = (policy_out_next[i] == max_value).nonzero(as_tuple=True)[0]
                if len(max_action) > 1:
                    # if several actions have same action value, choose the first one
                    max_action = torch.tensor([max_action[0]])
                max_a_batch.append(max_action)
            max_a_batch = torch.unsqueeze(torch.tensor(max_a_batch).to(device, dtype=torch.int64), dim=1)
        q_next = self.target_net(batch_s_).gather(1, max_a_batch)  # use detach to avoid the backpropagation during the training
        q_target = []
        for index, (s, a, r, s_, done) in enumerate(sample_batch):
            if not done:
                q_target_value = batch_r.squeeze()[index] + GAMMA * q_next[index]
            else:
                q_target_value = batch_r.squeeze()[index]
            q_target.append(q_target_value)
        q_target = torch.tensor(q_target).to(device)
        policy_out_put = self.policy_net(batch_s)
        q_eval = policy_out_put.gather(1, batch_a)
        loss = self.loss_func(q_eval.squeeze(), q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, episode):
        torch.save(self.policy_net.state_dict(), './model_' + self.agent_name + '/policy_net_' + str(episode) + '.pt')
        torch.save(self.target_net.state_dict(), './model_' + self.agent_name + '/target_net_' + str(episode) + '.pt')

    def load_model(self, episode):
        self.policy_net.load_state_dict(torch.load('./model_' + self.agent_name + '/policy_net_' + str(episode) + '.pt'))
        self.target_net.load_state_dict(torch.load('./model_' + self.agent_name + '/target_net_' + str(episode) + '.pt'))
