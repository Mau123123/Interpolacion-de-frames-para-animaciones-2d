import numpy as np
import argparse
from cv2 import remap, INTER_LINEAR
from torch.nn import DataParallel
from torch import load, from_numpy
from model.GMA.core.network import RAFTGMA
from model.GMA.core.utils.utils import InputPadder

class Warp():
    def __init__(self):
        self.device = "cuda"
        self.args = self.loadArgs()
        self.flow = DataParallel(RAFTGMA(self.args))
        self.flow.load_state_dict(load(self.args.model))
        self.flow = self.flow.module
        self.flow.to(self.device)
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
        
    def warpImage(self, image, img1, img2):
        padder = InputPadder(img1.shape)
        img1, img2 = padder.pad(img1, img2)
        flow_low, flow_up = self.flow(img1, img2, iters=6, test_mode=True)
        flow_up = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()
        h, w = flow_up.shape[:2]
        flow = -flow_up*0.5
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        warped = remap(image, flow, None, INTER_LINEAR)
        warped = warped[2:h-2, ::]
        print(warped.shape)
        return warped