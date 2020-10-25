import torch
import torch.nn as nn
import os
from lenet import LeNet
from dataset import Dataset
import torch.optim as optim

#实例化模型、loss、优化器
net = LeNet()
criterion = nn.CrossEntropyLoss()
optimzer = optim.Adam(net.parameters(), lr = 0.001)
if os.path.exists("./model/model.pkl"):
    net.load_state_dict(torch.load("./model/model.pkl"))
    optimzer.load_state_dict(torch.load("./model/optimizer.pkl"))

def train(Epoch):
    net.train()
    sum_loss = 0
    total = 0
    predict = 0
    train_dataset = Dataset()
    for epoch in range(Epoch):
        for i, (inputs,labels) in enumerate(train_dataset):
            optimzer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimzer.step()

            sum_loss += loss.item()
            total += labels.size(0)
            pre = outputs.max(-1)[-1]
            predict += pre.eq(labels).float().sum()

            print("[{}/{}] : avg_loss = {}  avg_accuracy = {}".format(epoch,Epoch,sum_loss / (i+1),predict / total))
            torch.save(net.state_dict(),"./model/model.pkl")
            torch.save(optimzer.state_dict(),"./model/optimizer.pkl")
train(2)


