import PIL
import torch
#from PIL.Image import Image
#from torch.distributions import transforms
import os
import numpy as np
from PIL import Image
#from cv2.cv2 import transform
from bokeh.transform import transform
from mpmath.tests.test_functions2 import V
#from torch.autograd.grad_mode import T
#

from torchvision.transforms import transforms
from keras.applications import ResNet50
from torchvision import datasets, models, transforms
# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((1.1618,), (1.1180,))])
#
# def get_files(directory):
#     return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
#             if os.path.isfile(os.path.join(directory, f))]
# images = np.array([])
# file = get_files('predict/')
# for i, item in enumerate(file):
#     print('Processing %i of %i (%s)' % (i+1, len(file), item))
#     image = transform(Image.open(item).convert('L'))
#     images = np.append(images, image.numpy())
#
# img = images.reshape(-1, 1, 256,256)
# img = torch.from_numpy(img).float()
# label = torch.ones(5,1).long()


def predict(img_path):
    net = torch.load('models/jun_data_model_30.pt')
    #net = net.to(torch.device)
    torch.no_grad()
    img_ = PIL.Image.open(img_path)
    img_ = transform(img_).unsqueeze(0)
    img_ = img_.to(torch.device)
    img_=torch.tensor(img_)
    outputs = net(img_)
    _, predicted = torch.max(outputs, 1)
    print(predicted)

predict("predict/Boletus_4.jpg")

# from PIL import Image
# import torchvision.transforms as T
#
# from torch.autograd import Variable as V
# import torch as
#
# trans = T.Compose([
#     transforms.Scale(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
# ])
#
# # 读入图片
# img = Image.open('predict/Boletus_4.jpg')
# input = trans(img)  # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
# #img = img.unsqueeze(0)  # 增加一维，输出的img格式为[1,C,H,W]
# resnet50 = models.resnet50(pretrained=True)
# model = resnet50.to('cuda:0') # 导入网络模型
# model.eval()
# model.load_state_dict(torch.t.load('models/jun_data_model_30.pt'))  # 加载训练好的模型文件
#
# input = V(img.cuda())
# score = model(input)  # 将图片输入网络得到输出
# probability = torch.t.nn.functional.softmax(score, dim=1) # 计算softmax，即该图片属于各类的概率
# max_value, index = torch.t.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
# print(index)
