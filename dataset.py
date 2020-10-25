import torch
import torchvision
import torchvision.transforms as transforms
import argparse

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
BATCH_SIZE = 4
#准备数据
def Dataset(train = True, batch_size = BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    ]) 
    dataset = CIFAR10(root="./data",train= train,transform=transform,download=False)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return dataloader

