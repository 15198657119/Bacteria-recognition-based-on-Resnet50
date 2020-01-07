# -*- coding: utf-8 -*-
# @Author  : 胡子旋
# @Email   ：1017190168@qq.com


import torch
from PIL import Image
from torch import nn, optim
from torchvision import transforms
from torchvision.models import resnet18
import time

from torch.utils.data import DataLoader

classes = ('Boletus', 'Chickenchop', 'Cookedbacteria', 'Greenbackterium')
device = torch.device('cuda')
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def prediect(img_path):
    #net = torch.load('model.pkl')
    net = torch.load('models/jun_data_model_30.pt')
    net = net.to(device)
    torch.no_grad()
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print('this picture maybe :', classes[predicted[0]])


if __name__ == '__main__':
    t1=time.time()
    prediect('predict/timg.jpeg')
    t2=time.time()
    print("time",t2-t1)
