import torch.optim as optim
import torch.cuda as cuda
import torch.nn.functional as F
import torch
from torch import cat
from UNET import UNET
from torch.utils.data import random_split
from Dataset import FramesDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#iniciar parametros
BATCH = 5
EPOCHS = 1
device = "cuda" if cuda.is_available() else "cpu"
unet = UNET(in_channels=6, out_channels=3).to(device)
optimizer = optim.Adam(unet.parameters(), lr= 0.001)
dataset = FramesDataset(dir = r'C:\Users\Mau\Desktop\proyectos\Proyecto\dataset', transform=transforms.ToTensor())

#dividir el dataset
len = dataset.__len__()
test = len//10
train = len - test 
trainset, testset = random_split(dataset,[train,test])
trainset = DataLoader(trainset, batch_size=BATCH, shuffle=True)
testset = DataLoader(testset, batch_size=BATCH, shuffle=False)

#entrenar el modelo
loss = None
for epoch in range(EPOCHS):
    for data in trainset:
        print(loss)
        (F1,F3) ,F2 = data
        input = cat((F1,F3),1)
        output = unet(input)
        optimizer.zero_grad()
        loss = F.mse_loss(output , F2)
        loss.backward()
        optimizer.step()
    print(loss)

#guardar el modelo
torch.save(unet.state_dict(), 'weights.pth')