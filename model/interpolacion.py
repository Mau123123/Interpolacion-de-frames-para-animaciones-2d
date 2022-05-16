from GMA.core.network import RAFTGMA
from GMA.core.utils.utils import InputPadder
from model.UNET import UNET 
from model.resNet import resNet 
from model.convNet import convNet 
import cv2
from os.path import exists
import torch
import torch.cuda as cuda
from torchvision import transforms
import numpy as np
import argparse
from PIL.Image import open
import torchvision.transforms as transforms


class interpolacion():
    
    def __init__(self):
        super().__init__()
        self.args = self.loadArgs()
        self.device = "cuda" if cuda.is_available() else "cpu"
        #self.model = UNET(in_channels=3, out_channels=3).to(self.device)
        self.model = resNet(in_channels=6, out_channels=3).to(self.device)
        #self.model = convNet(in_channels=6, out_channels=3).to(self.device)
        if(exists("./model/weights/resNet.pth")):
            self.model.load_state_dict(torch.load('./model/weights/resNet.pth'))
        self.flow = torch.nn.DataParallel(RAFTGMA(self.args))
        self.flow.load_state_dict(torch.load(self.args.model))
        self.flow = self.flow.module
        self.flow.to(self.device)
        self.convert_tensor = transforms.ToTensor()
        self.flow.eval()
        self.transform =  transforms.ToTensor()
        
    def loadArgs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', help="restore checkpoint", default="GMA/checkpoints/gma-things.pth")
        parser.add_argument('--model_name', help="define model name", default="GMA")
        parser.add_argument('--path', help="dataset for evaluation")
        parser.add_argument('--num_heads', default=1, type=int,
                            help='number of heads in attention and aggregation')
        parser.add_argument('--position_only', default=False, action='store_true',
                            help='only use position-wise attention')
        parser.add_argument('--position_and_content', default=False, action='store_true',
                            help='use position and content-wise attention')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        return parser.parse_args(args=[])
    
    def loadImage(self,imfile):
        img = open(imfile)
        img = self.transform(img)
        return img[None].to(self.device)
    
    def calFlow(self, img1, img2):
        shape = img1.shape
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        with torch.no_grad():
            flow_low, flow_up = self.flow(img1, img2, iters=10, test_mode=True)
        flow_up = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()  
        if(img1.shape != shape):
            flow_up = flow_up[2:-2,:,:]  
        h, w = flow_up.shape[:2]
        flow = flow_up*0.5
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        flow = np.float32(flow)
        return flow
    

    def warp(self, img, flow):
        img = np.float32(img.permute(0, 2, 3, 1).cpu())[0]
        warped = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
        warped = warped[...,::-1].copy()
        warped = torch.unsqueeze(self.convert_tensor(warped).to(self.device),0)
        return warped
    
    def generatedFrame(self, f1, f2):
        input = torch.cat([f1,f2], dim=1)
        with torch.no_grad():
            tensor_image = self.model(input).cpu().detach().numpy()[0]
        tensor_image = np.transpose(tensor_image, (1,2,0))    
        return tensor_image
    
    def refineWarp(self, image,  img1, img2 ):
        image = torch.clone(image)
        image1 = torch.clone(img1)
        image2 = torch.clone(img2)
        image1 = np.transpose(image1.cpu().detach().numpy()[0], (1,2,0))
        image2 = np.transpose(image2.cpu().detach().numpy()[0], (1,2,0))
        image = image[0].permute(1, 2, 0).cpu().detach().numpy()
        diference = np.absolute(image1-image)
        diference = diference[:,:,0] + diference[:,:,1]+ diference[:,:,2]
        diferenceMap1 = 1*(diference<100)
        diferenceMap2 = 1*(diference>100)
        image1[:,:,0] = image1[:,:,0] * diferenceMap1
        image1[:,:,1] = image1[:,:,1] * diferenceMap1
        image1[:,:,2] = image1[:,:,2] * diferenceMap1
        image2[:,:,0] = image2[:,:,0] * diferenceMap2
        image2[:,:,1] = image2[:,:,1] * diferenceMap2
        image2[:,:,2] = image2[:,:,2] * diferenceMap2
        image1 = image1 + image2
        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        return image1[None].to(self.device)
               
    def forward(self, img1, img3):
        flow = self.calFlow(img1, img3)
        flow2 = self.calFlow(img3, img1)
        f1 = self.warp(img1, flow)
        f2 = self.warp(img3, flow2)
        f11 = self.refineWarp(img1, f1 ,f2)
        f22 = self.refineWarp(img3, f2, f1)
        newFrame = self.generatedFrame(f11, f22)
        return np.abs(newFrame)
        
        