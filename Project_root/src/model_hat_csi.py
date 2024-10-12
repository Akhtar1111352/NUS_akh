#CODE BY M M AKHTAR
import torch
import torch.nn as nn
import torch.nn.functional as F


class HAT_CNN(nn.Module):
    def __init__(self, num_tasks):
        super(HAT_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)  

        
        self.masks = nn.ParameterList([nn.Parameter(torch.ones(128), requires_grad=True) for _ in range(num_tasks)])

    def forward(self, x, task_id):
        
        mask = torch.sigmoid(self.masks[task_id])

        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        
        x = x * mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
