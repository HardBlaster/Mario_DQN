import copy
import torch.nn as nn


class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        
        self.online = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        self.target = copy.deepcopy(self.online)
        
        for p in self.target.parameters():
            p.requires_grad = False
            
    def forward(self, x, model="online"):
        return self.online(x) if model == "online" else self.target(x)
