import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import copy

# from lightly.models.utils import deactivate_requires_grad
# from lightly.models.utils import update_momentum

class SimSiamModel(nn.Module):
    def __init__(self):
        super(SimSiamModel, self).__init__()
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(*[
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        ])
    
    def forward(self, x):
        return self.projector(self.backbone(x))

class SimCLRModel(nn.Module):
    def __init__(self):
        super(SimCLRModel, self).__init__()
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(*[
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        ])
    
    def forward(self, x):
        return F.normalize(self.projector(self.backbone(x)))

class DCLWModel(nn.Module):
    def __init__(self):
        super(DCLWModel, self).__init__()
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(*[
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        ])
    
    def forward(self, x):
        return F.normalize(self.projector(self.backbone(x)))

class VICRegModel(nn.Module):
    def __init__(self):
        super(VICRegModel, self).__init__()
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(*[
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096)
        ])

    def forward(self, x):
        return self.projector(self.backbone(x))

class BarlowModel(nn.Module):
    def __init__(self):
        super(BarlowModel, self).__init__()
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(*[
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096)
        ])
        self.bn = nn.BatchNorm1d(4096, affine=False)

    def forward(self, x):
        return self.bn(self.projector(self.backbone(x)))

class TiCoModel(nn.Module):
    def __init__(self):
        super(TiCoModel, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.projection_head = nn.Sequential(*[
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, 4096)
        ])
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        self.deactivate_requires_grad(self.backbone_momentum)
        self.deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

    def deactivate_requires_grad(self, module):
        for param in module.parameters():
            param.requires_grad = False
    
    def update_momentum(self, model, model_ema, m):
        for model_ema, model in zip(model_ema.parameters(), model.parameters()):
            model_ema.data = model_ema.data * m + model.data * (1.0 - m)

    def schedule_momentum(self, iter, max_iter, m=0.99):
        return m + (1 - m)*np.sin((np.pi/2)*iter/(max_iter-1))

