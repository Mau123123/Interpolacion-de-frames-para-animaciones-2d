import os
from PIL import Image
from torch import from_numpy
from numpy import array, uint8
from warp import Warp
import shutil
from cv2 import imread, imwrite, IMREAD_UNCHANGED

def load_image(imfile):
        img = array(Image.open(imfile)).astype(uint8)
        img = from_numpy(img).permute(2, 0, 1).float()
        return img[None].to("cuda")

def frameList(dir , Warp):
        for directories in os.listdir(dir):
           if os.path.exists(dir + '\\' +directories + '\\'+ "warped"):
                shutil.rmtree(dir + '\\' +directories + '\\'+ "warped")

                    
dir = r'C:\Users\Mau\Desktop\proyectos\Proyecto\test_2k_540p'
warp = Warp()
frameList(dir, warp)
