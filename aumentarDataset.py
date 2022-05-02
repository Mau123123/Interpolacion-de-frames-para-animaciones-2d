import cv2
import os
import shutil
import generateWarped
from warp import Warp

vid = "simpson2"
dataset = "dataset"

def ob_img(video_dir, name, dim):
    print(video_dir)
    vidcap = cv2.VideoCapture(video_dir)
    success,image = vidcap.read()
    count = 100000
    if os.path.exists(f"./{dataset}/{name}"):
        return
    os.mkdir(f"./{dataset}/{name}")
    os.chdir(f"./{dataset}/{name}")
    while success:
        if dim: 
            image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)
        imgname = "%d.jpg" % count
        cv2.imwrite(imgname, image)
        success,image = vidcap.read()
        count += 1
    os.chdir(f"../..")

DIM = (400,300)      
directory = os.getcwd()+"/{dataset}"
for video in os.listdir("video"):
    dir = os.getcwd()+r"\video"+ "\\" + video
    name = video.split(".")[0]
    ob_img(dir,name, DIM)
