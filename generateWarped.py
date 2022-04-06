import os
from PIL import Image
from torch import from_numpy
from numpy import array, uint8
from warp import Warp
from cv2 import imread, imwrite, IMREAD_UNCHANGED

def load_image(imfile):
        img = array(Image.open(imfile)).astype(uint8)
        img = from_numpy(img).permute(2, 0, 1).float()
        return img[None].to("cuda")

def frameList(dir , Warp):
        for directories in os.listdir(dir):
            images =  os.listdir(dir + '\\' + directories)
            images.sort()
            directory_frames = []
            for i in range(len(images)-2):
                img1 = dir + '\\' +directories +'\\' + images[i]
                img2 = dir + '\\' +directories +'\\' + images[i+1]
                img3 =dir + '\\' +directories +'\\' +  images[i+2]
                input = (img1,img3)
                directory_frames.append(input)
            
            if not os.path.exists(dir + '\\' +directories + '\\'+ "warped"):
                os.mkdir(dir + '\\' +directories + '\\'+ "warped")
                os.mkdir(dir + '\\' +directories + '\\'+"warped"+ '\\'  + "f1")
                os.mkdir(dir + '\\' +directories + '\\'+"warped"+ '\\' + "f2")
                idx = 0
                for (img1 , img2) in directory_frames:
                    print(img1)
                    image1  = imread(img1, IMREAD_UNCHANGED )
                    image2  = imread(img2, IMREAD_UNCHANGED )
                    img1 = load_image(img1)
                    img2 = load_image(img2)
                    f1 = Warp.warpImage(image1,img1, img2)
                    f2 = Warp.warpImage(image2,img2, img1)
                    f1name = dir + '\\'+directories + '\\' +"warped"+ '\\' + "f1/frame%d.jpg" % (idx+10000)
                    f2name = dir + '\\'+directories + '\\' +"warped"+ '\\' + "f2/frame%d.jpg" % (idx+10000)
                    imwrite(f1name, f1)
                    imwrite(f2name, f2)
                    idx += 1
                    
dire = r'C:\Users\Mau\Desktop\proyectos\Proyecto\test_2k_540p'
warp = Warp()
frameList(dire, warp)
