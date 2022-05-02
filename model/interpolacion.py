from PIL import Image
from GMA.core.network import RAFTGMA
from GMA.core.utils.utils import InputPadder
from model.UNET import UNET 
from warp import Warp
import cv2
import torch
import torch.cuda as cuda
from torchvision import transforms
import numpy as np
import argparse


class interpolacion():
    
    def __init__(self):
        super().__init__()
        self.args = self.loadArgs()
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = UNET(in_channels=3, out_channels=3).to(self.device)
        self.model.load_state_dict(torch.load('model/weights.pth'))
        self.flow = torch.nn.DataParallel(RAFTGMA(self.args))
        self.flow.load_state_dict(torch.load(self.args.model))
        self.flow = self.flow.module
        self.flow.to(self.device)
        self.convert_tensor = transforms.ToTensor()
        self.flow.eval()
        
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
        return parser.parse_args()
    
    def calFlow(self, img1, img2):
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        flow_low, flow_up = self.flow(img1, img2, iters=8, test_mode=True)
        flow_up = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()    
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
    
    def generatedFrame(self, f1, f2 , img):
        tensor_image = self.model(f1,f2, img).cpu().detach().numpy()[0]
        tensor_image = np.transpose(tensor_image, (1,2,0))    
        return tensor_image
        
    def forward(self, img1, img2):
        flow = self.calFlow(img1, img2)
        f1 = self.warp(img1, flow)
        f2 = self.warp(img2, flow)
        cv2.imshow("f1", np.transpose(f1.cpu().detach().numpy()[0], (1,2,0))/255)
        cv2.imshow("f2", np.transpose(f2.cpu().detach().numpy()[0], (1,2,0))/255)
        newFrame = self.generatedFrame(f1, f2 ,f1)
        return newFrame
        
        