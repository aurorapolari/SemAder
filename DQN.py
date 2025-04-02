import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import os
from config import *

class DQN:
    def __init__(self, model_path=None):
        self.memory = deque(maxlen=10000)

        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.min_lr = 0.00001
        self.learning_decay = 0.5
        self.tau = 0.125

        self._action_space = action_set

        self.init_model()
        if model_path:
            self.load_model(model_path)

    def init_model(self):
        self.model, self.model_optimizer = self.create_model()
        self.target_model, self.target_model_optimizer = self.create_model()

    def load_model(self, model_path):
        try:
            self.model, _ = self.create_model()
            dummy_input = torch.zeros((1, 1, embedding_len))
            _ = self.model(dummy_input)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        except Exception as e:
            print(f'Cannot load model from {model_path}')
            print(str(e))
            self.init_model()

    def create_model(self):
        class DQNModel(nn.Module):
            def __init__(self, input_size, output_size):
                super(DQNModel, self).__init__()
                self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=512, batch_first=True, bidirectional=True)
                self.lstm2 = nn.LSTM(input_size=1024, hidden_size=512, batch_first=True)
                self.fc1 = nn.Linear(512, 512)
                self.fc2 = nn.Linear(512, output_size)

            def forward(self, x):
                x, _ = self.lstm1(x)
                x, _ = self.lstm2(x)
                x = x[:, -1, :]
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        input_size = embedding_len
        output_size = len(action_set)
        model = DQNModel(input_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    def set_lr(self, lr):
        current_lr = self.model_optimizer.param_groups[0]['lr']
        print(f'Set learning rate from {current_lr} to {lr}')
        self.learning_rate = lr
        for param_group in self.model_optimizer.param_groups:
            param_group['lr'] = self.learning_rate
        for param_group in self.target_model_optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def reduce_lr(self):
        if self.learning_rate == self.min_lr:
            return
        new_learning_rate = self.learning_rate * self.learning_decay
        if new_learning_rate < self.min_lr:
            new_learning_rate = self.min_lr
        print(f'Reduce learning rate from {self.learning_rate} to {new_learning_rate}')
        self.learning_rate = new_learning_rate
        self.set_lr(new_learning_rate)

    def act(self, state, eval_tag=False):
        if not eval_tag:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)
            if np.random.random() < self.epsilon:
                #sample_index = self._action_space.sample()
                return random.choice(self._action_space)# action_set[sample_index]
        state = torch.tensor(state, dtype=torch.float32)    # state:3
        with torch.no_grad():
            q_values = self.model(state)
        action_index = torch.argmax(q_values).item()
        return action_set[action_index]
        #return torch.argmax(q_values).item()

    def remember(self, state, action, reward, new_state, done):
        action_index = action_set.index(action)
        self.memory.append((state, action_index, reward, new_state, done))

    def replay(self):
        batch_size = 128
        if len(self.memory) < batch_size:
            print("replay error")
            return

        samples = random.sample(self.memory, batch_size)
        states, targets = [], []

        for state, action, reward, new_state, done in samples:
            state = torch.tensor(state, dtype=torch.float32)
            new_state = torch.tensor(new_state, dtype=torch.float32)
            target = self.target_model(state).clone().detach()
            if done:
                target[0][action] = reward
            else:
                Q_future = torch.max(self.target_model(new_state)).item()
                target[0][action] = reward + Q_future * self.gamma
            states.append(state)
            targets.append(target)

        states = torch.cat(states, dim=0)
        targets = torch.cat(targets, dim=0)

        self.model.train()
        self.model_optimizer.zero_grad()
        predictions = self.model(states)
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        self.model_optimizer.step()
        print('Training...')

    def target_train(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, fn, iteration):
        if not os.path.isdir(fn):
            os.makedirs(fn)
        file_path = os.path.join(fn, f'dqn_model-{iteration}.pt')
        torch.save(self.model.state_dict(), file_path)
