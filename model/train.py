import torch.optim as optim
import UNET as UNET
from torch.utils.data import random_split
from Dataset import FramesDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

unet = UNET(in_channels=6, out_channels=3)
optimizer = optim.Adam(unet.parameters(), lr= 0.0001)

dataset = FramesDataset(dir = r'C:\Users\Mau\Desktop\proyectos\Proyecto\dataset', transform=transforms.ToTensor())

trainset, testset = random_split(dataset, [4000,700])

epochs = 3

for epoch in range(epochs):
    for data in trainset:
        x, y = data