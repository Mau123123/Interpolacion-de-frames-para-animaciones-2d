import os
from torch import cat
import numpy as np
from torch.utils.data import Dataset
from cv2 import imread, imshow

class FramesDataset(Dataset):
    
    def __init__(self, dir, transform = None):
        self.transform = transform
        self.dir = dir #'C:\Users\Mau\Desktop\proyectos\Proyecto\dataset'
        self.frames = self.frameList()
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self,index):
        ((frame1,frame3),frame2) =  self.frames[index]
        frame1 = imread(frame1)
        frame2 = imread(frame2)
        frame3 = imread(frame3)
        if self.transform is None:
            return ((frame1, frame3), frame2)
        frame1 = self.transform(frame1)
        frame2 = self.transform(frame2)
        frame3 = self.transform(frame3)
        return ((frame1.cuda(),frame3.cuda()), frame2.cuda())
    
    def frameList(self):
        frames = []
        for directories in os.listdir(self.dir):
            images =  os.listdir(self.dir + '\\' + directories)
            images.sort()
            for i in range(len(images)-2):
                img1 = self.dir + '\\' +directories +'\\' + images[i]
                img2 = self.dir + '\\' +directories +'\\' + images[i+1]
                img3 =self.dir + '\\' +directories +'\\' +  images[i+2]
                input = (img1,img3)
                output = (img2)
                frames.append((input,output))
        return frames