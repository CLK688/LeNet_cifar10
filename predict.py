import torch
import torch.nn as nn
from lenet import LeNet
from dataset import Dataset

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#加载模型
net = LeNet()
net.load_state_dict(torch.load("./model/model.pkl"))
def predict():
    net.eval()
    class_correct = list(0 for i in range(10))
    class_total = list(0 for i in range(10))
    predict_dataset = Dataset(train=False,batch_size=4)
    for i, (data,labels) in enumerate(predict_dataset):
        with torch.no_grad():
            outputs = net(data)
            pre = outputs.max(-1)[-1]
            c = pre.eq(labels).float().squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] +=1

    
    for i in range(10):

        print("Accuracy of %5s : %2d %%"%(classes[i], 100*class_correct[i]/class_total[i]))



predict()