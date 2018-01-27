from torchvision import  models
from data.CIFAR10_utils import get_dataloader
from training.training_utils import train_model
import torch.nn as nn

"""
model = models.alexnet(pretrained=False, num_classes=6)
"""
model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)

train_model(model_ft, model_ft.fc, get_dataloader())

