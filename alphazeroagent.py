import torch
import torch.nn as nn
import torch.optim as optim

class AlphaZeroNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(AlphaZeroNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.res_block1 = self._build_residual_block(hidden_size)
        self.res_block2 = self._build_residual_block(hidden_size)
        
        self.policy_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.policy_head = nn.Linear(hidden_size // 2, action_size)
        
        self.value_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.value_head = nn.Linear(hidden_size // 2, 1)
        
    def _build_residual_block(self, hidden_size):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        
        residual = x
        x = torch.relu(self.res_block1(x) + residual)
        residual = x
        x = torch.relu(self.res_block2(x) + residual)
        
        policy = torch.relu(self.policy_fc(x))
        policy = torch.softmax(self.policy_head(policy), dim=-1)
        
        value = torch.relu(self.value_fc(x))
        value = torch.tanh(self.value_head(value))
        
        return policy, value
