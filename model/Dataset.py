import os
import torch.cuda as cuda
import numpy as np
from torch.utils.data import Dataset
from cv2 import imread, calcOpticalFlowFarneback, CV_16S,remap,INTER_LINEAR,imwrite,cvtColor,COLOR_BGR2GRAY


class FramesDataset(Dataset):
    
    def __init__(self, dir, transform = None):
        self.device = "cuda" if cuda.is_available() else "cpu"
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
        return ((frame1.to(self.device),frame3.to(self.device)), frame2.to(self.device))
    
    def frameList(self):
        frames = []
        output = []
        for directories in os.listdir(self.dir):
            images =  os.listdir(self.dir + '\\' + directories)
            images.sort()
            directory_frames = []
            for i in range(len(images)-2):
                img1 = self.dir + '\\' +directories +'\\' + images[i]
                img2 = self.dir + '\\' +directories +'\\' + images[i+1]
                img3 =self.dir + '\\' +directories +'\\' +  images[i+2]
                input = (img1,img3)
                output.append(img2)
                directory_frames.append(input)
            
            if not os.path.exists(self.dir + '\\' +directories + '\\'+ "warped"):
                os.mkdir(self.dir + '\\' +directories + '\\'+ "warped")
                os.mkdir(self.dir + '\\' +directories + '\\'+"warped"+ '\\'  + "f1")
                os.mkdir(self.dir + '\\' +directories + '\\'+"warped"+ '\\' + "f2")
                idx = 0
                for (img1 , img2) in directory_frames:
                    img1  = imread(img1 )
                    img2  = imread(img2 )
                    img1gray = cvtColor(img1, COLOR_BGR2GRAY)
                    img2gray = cvtColor(img2, COLOR_BGR2GRAY)
                    flow12 = calcOpticalFlowFarneback(img1gray, img2gray,CV_16S, 0.2, 5, 2, 3, 5, 1.1, 0)
                    flow21 = calcOpticalFlowFarneback(img2gray, img1gray,CV_16S, 0.2, 5, 2, 3, 5, 1.1, 0)
                    h, w = flow12.shape[:2]
                    flow12 = -flow12*0.5
                    flow12[:,:,0] += np.arange(w)
                    flow12[:,:,1] += np.arange(h)[:,np.newaxis]
                    flow21 = -flow21*0.5
                    flow21[:,:,0] += np.arange(w)
                    flow21[:,:,1] += np.arange(h)[:,np.newaxis]
                    f1 = remap(img1, flow12, None, INTER_LINEAR)
                    f2 = remap(img2, flow21, None, INTER_LINEAR)
                    f1name = self.dir + '\\'+directories + '\\' +"warped"+ '\\' + "f1/frame%d.jpg" % (idx+10000)
                    f2name = self.dir + '\\'+directories + '\\' +"warped"+ '\\' + "f2/frame%d.jpg" % (idx+10000)
                    imwrite(f1name, f1)
                    imwrite(f2name, f2)
                    frames.append(((f1name,f2name), output[idx]))
                    
                    idx += 1
            else:
                F1frames = os.listdir(self.dir + '\\' +directories + '\\'+"warped"+ '\\'  + "f1")                
                F2frames = os.listdir(self.dir + '\\' +directories + '\\'+"warped"+ '\\'  + "f2")     
                for idx in range(len(F1frames)):
                    f1 = F1frames[idx]
                    f2 = F2frames[idx]
                    f1 = self.dir + '\\'+directories + '\\' +"warped"+ '\\' + "f1/"+f1
                    f2 = self.dir + '\\'+directories + '\\' +"warped"+ '\\' + "f1/"+f2     
                    frames.append(((f1,f2), output[idx]))     
        return frames 