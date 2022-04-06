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
BATCH = 1
EPOCHS = 10
device = "cuda" if cuda.is_available() else "cpu"
dataset = FramesDataset(dir = r'C:\Users\Mau\Desktop\proyectos\Proyecto\dataset', transform=transforms.ToTensor())
unet = UNET(in_channels=6, out_channels=3).to(device)
optimizer = optim.Adam(unet.parameters(), lr= 0.000001)

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
        output = unet(F1, F3)
        optimizer.zero_grad()
        loss  = F.mse_loss(output , F2)
        loss.backward()
        optimizer.step()
    print(loss)

#guardar el modelo
torch.save(unet.state_dict(), 'weights.pth')